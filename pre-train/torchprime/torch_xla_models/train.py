import importlib
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from collections import OrderedDict
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

from dotenv import load_dotenv
load_dotenv()

import datasets
import hydra
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
import wandb
import webdataset as wds
from omegaconf import ListConfig, DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import Dataset as HuggingFaceDataset
from datasets.distributed import split_dataset_by_node
from torch_xla._internal.jax_workarounds import jax_env_context
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from transformers import (
  AutoTokenizer,
  default_data_collator,
  get_scheduler,
  set_seed,
)
from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer
from transformers.optimization import Adafactor
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version
from transformers import PreTrainedTokenizerBase

from torchprime.data.dataset import make_huggingface_dataset, make_gcs_dataset, make_gcs_pretokenized_dataset
from torchprime.data.webdataset import make_webdataset, create_webdataset_collate_fn
from torchprime.data.sft_data_collator import SFTDataCollator, make_sft_dataset
from torchprime.layers.sequential import HomogeneousSequential
from torchprime.metrics.metrics import MetricsLogger
from torchprime.metrics.mfu import compute_mfu
from torchprime.metrics.step_duration import step_duration_from_latest_profile
from torchprime.sharding.shard_model import (
  shard_torch_xla_model_from_config,
  wrap_module,
)
from torchprime.torch_xla_models import offloading, remat_all, scan_layers
from torchprime.torch_xla_models.topology import (
  get_mesh,
  get_num_slices,
  is_1d_sharding,
)
from torchprime.torch_xla_models.model_utils import (
  initialize_model_class,
  set_default_dtype,
  load_hf_model,
  get_model_dtype,
)
from torchprime.torch_xla_models.masking_scheduler import MaskingScheduler
from torchprime.utils.retry import retry

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True

if not dist.is_initialized():
  dist.init_process_group(backend='gloo', init_method='xla://')

def is_main_process():
  """Check if this is the main process (rank 0)."""
  return xr.process_index() == 0


MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)
GCS_PREFIX = "gs://sfr-text-diffusion-model-research/"

class Trainer:
  """The trainer."""

  minibatch: bool

  def __init__(
    self,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    train_dataset: Dataset | IterableDataset | None = None,
    eval_dataset: Dataset | IterableDataset | None = None,
  ):
    self.config = config
    self.device = torch_xla.device()
    self.global_batch_size = self.config.global_batch_size
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.tokenizer = tokenizer

    # Set up SPMD mesh and shard the model
    mesh = get_mesh(self.config)
    xs.set_global_mesh(mesh)

    # TODO(https://github.com/pytorch/xla/issues/8696): Minibatch only works in 1D sharding.
    minibatch = is_1d_sharding(tuple(config.ici_mesh.values()))
    self.minibatch = minibatch
    if is_main_process():
      logger.info(f"Logical mesh shape: {mesh.shape()}")
      logger.info(f"Logical mesh device assignments: {mesh.device_ids}")
      logger.info(f"Minibatch dataloading: {minibatch}")

    # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
    self.input_sharding_spec = xs.ShardingSpec(
      mesh, (("data", "fsdp"), None), minibatch=minibatch
    )

    # Recursively replace `nn.Linear` layers with einsum operations in the model.
    # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
    # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
    model = apply_xla_patch_to_nn_linear(model)

    # Annotate model weights and activations with sharding constraints to distribute
    # the training across devices following the SPMD paradigm.
    sharding_config = OmegaConf.to_container(self.config.model.sharding, resolve=True)
    assert isinstance(sharding_config, dict), (
      f"Sharding config {sharding_config} must be a dict"
    )
    model = shard_torch_xla_model_from_config(model, config=sharding_config)

    # Rematerialize forward computation during the backward pass if requested.
    model = self._add_checkpoint_offload_scan_model(model)
    model = self._add_optimization_barrier_model(model)
    self.model = model

    # Set up optimizers
    self.optimizer = Adafactor(
      params=model.parameters(),
      lr=self.config.optimizer.learning_rate,
      relative_step=False,
      scale_parameter=False,
    )

    self.lr_scheduler = get_scheduler(
      name=self.config.lr_scheduler.type,
      optimizer=self.optimizer,
      num_warmup_steps=self.config.lr_scheduler.warmup_steps,
      num_training_steps=self.config.max_steps,
    )

    # Initialize masking scheduler
    scheduler_config = self.config.model.masking_scheduler
    # Convert mask_block_sizes to proper list format
    mask_block_sizes = self.config.model.mask_block_sizes
    if mask_block_sizes is not None:
      # Convert to list (handles OmegaConf ListConfig)
      mask_block_sizes = list(mask_block_sizes)
      # If it's a list of lists, ensure inner lists are also converted
      if len(mask_block_sizes) > 0 and isinstance(mask_block_sizes[0], (list, type(self.config.model.mask_block_sizes))):
        mask_block_sizes = [list(inner) for inner in mask_block_sizes]
    
    self.masking_scheduler = MaskingScheduler(
      schedule_type=scheduler_config.schedule_type,
      max_schedule_steps=scheduler_config.max_schedule_steps,
      prefix_probability=self.config.model.prefix_probability,
      truncate_probability=self.config.model.truncate_probability,
      block_masking_probability=self.config.model.block_masking_probability,
      mask_block_sizes=mask_block_sizes,
      total_training_steps=self.config.max_steps,
    )

    # Initialize checkpoint manager
    # Use GCS for checkpoints with proper path handling
    self.checkpoint_load_dir = self.config.checkpoint_load_dir # NOTE: config.checkpoint_load_dir always used for loading checkpoints
    if self.checkpoint_load_dir is not None:
      self.checkpoint_load_manager = CheckpointManager(path=self.checkpoint_load_dir, save_interval=config.save_steps)
    else:
      self.checkpoint_load_manager = None
    if self.config.checkpoint_save_dir is not None:
      self.checkpoint_save_dir = self.config.checkpoint_save_dir
      self.checkpoint_save_manager = CheckpointManager(path=self.checkpoint_save_dir, save_interval=self.config.save_steps)
    else:
      self.checkpoint_save_dir = self.checkpoint_load_dir
      self.checkpoint_save_manager = self.checkpoint_load_manager
    self.start_step = 0

    # Execute all initialization work queued so far before starting training.
    torch_xla.sync()


  def _load_checkpoint(self):
    """Load optimizer, scheduler, and training state from checkpoint."""
    tracked_steps = self.checkpoint_load_manager.all_steps()
    if not tracked_steps:
      logger.warning("No checkpoint steps found. Starting from scratch.")
      return
    state_dict = {
      "model": self.model.state_dict(), # NOTE: torch_xla has problem loading state dict with 2d sharding
    }
    if self.config.resume_from_checkpoint:
      # self.optimizer = prime_optimizer(self.optimizer) # NOTE: needed to create the dummy state dict for the optimizer
      state_dict.update(
        {
          # "optimizer": self.optimizer.state_dict(), # NOTE: torch_xla has problem loading optimizer state dict with 2d sharding
          "scheduler": self.lr_scheduler.state_dict(),
          "masking_scheduler": self.masking_scheduler.state_dict(),
          "step": self.start_step,
        }
      )
    checkpoint_load_step = self.config.checkpoint_load_step
    if checkpoint_load_step in tracked_steps:
      if is_main_process():
        logger.info(f"Loading checkpoint from step {checkpoint_load_step}")
      if self.config.model.model_class == "coda.Qwen2ForCausalLM":
        load_dir = os.path.join(MOUNTED_GCS_DIR, self.checkpoint_load_dir.split(GCS_PREFIX)[1], f"unsharded_state_dict_{checkpoint_load_step}.pt")
        unsharded_state_dict = torch.load(load_dir)
        self.model.load_state_dict(unsharded_state_dict, strict=False)
        state_dict["model"] = {name: param for name, param in self.model.named_parameters() if name not in unsharded_state_dict}
      self.checkpoint_load_manager.restore(checkpoint_load_step, state_dict)
    elif checkpoint_load_step == "latest":
      last_step = max(tracked_steps)
      if is_main_process():
        logger.warning(f"Checkpoint step {checkpoint_load_step} not found in tracked steps {tracked_steps}. Loading from latest checkpoint {last_step}.")
      if self.config.model.model_class == "coda.Qwen2ForCausalLM":
        load_dir = os.path.join(MOUNTED_GCS_DIR, self.checkpoint_load_dir.split(GCS_PREFIX)[1], f"unsharded_state_dict_{checkpoint_load_step}.pt")
        unsharded_state_dict = torch.load(load_dir)
        self.model.load_state_dict(unsharded_state_dict, strict=False)
        state_dict["model"] = {name: param for name, param in self.model.named_parameters() if name not in unsharded_state_dict}
      self.checkpoint_load_manager.restore(last_step, state_dict)
    else:
      raise ValueError(f"Invalid checkpoint step: {checkpoint_load_step}. Must be one of {tracked_steps} or 'latest'.")

    self.model.load_state_dict(state_dict["model"], strict=False)
    if self.config.resume_from_checkpoint:
      # self.optimizer.load_state_dict(state_dict["optimizer"])
      self.lr_scheduler.load_state_dict(state_dict["scheduler"])
      if "masking_scheduler" in state_dict:
        self.masking_scheduler.load_state_dict(state_dict["masking_scheduler"])
      self.start_step = state_dict["step"]

  def _get_dataloader(self, dataset: IterableDataset | wds.WebDataset | Dataset) -> pl.MpDeviceLoader:
    num_replicas = xr.process_count()
    if is_main_process():
      logger.info(f"Num replicas: {num_replicas}") # 64 for v5p-512

    per_worker_batch_size = self.global_batch_size // num_replicas
    if isinstance(dataset, IterableDataset) or isinstance(dataset, wds.WebDataset):
      # For IterableDataset, don't use DistributedSampler as it doesn't have len()
      sampler = None
      if is_main_process():
        logger.info("Using IterableDataset or WebDataset without DistributedSampler")
    else:
      sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=xr.process_index(),
        drop_last=True, # It's crucial to drop last to ensure all batches are even
      )
    # Choose appropriate data collator based on training mode
    # For pre-training, use default data collator
    collate_fn = default_data_collator

    if isinstance(dataset, wds.WebDataset):
      if self.config.training_mode == "sft":
        columns = ["input_ids", "src_mask"]
      else:
        columns = None
      dataloader = wds.WebLoader(
        dataset,
        collate_fn=create_webdataset_collate_fn(columns),
        batch_size=per_worker_batch_size,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
        drop_last=True,
      )
    else:
      dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=per_worker_batch_size, # <-- Use the smaller, per-worker batch size
        sampler=sampler,
        drop_last=True, # Sampler also has drop_last=True for safety
      )
    loader = pl.MpDeviceLoader(
      dataloader, self.device, input_sharding=self.input_sharding_spec
    )
    return loader

  def _get_eval_dataloader(self):
    if self.eval_dataset is None:
      raise ValueError("Trainer: evaluation requires a eval_dataset.")
    return self._get_dataloader(self.eval_dataset)

  def _get_train_dataloader(self):
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")
    return self._get_dataloader(self.train_dataset)

  def _add_checkpoint_offload_scan_model(self, model: nn.Module):
    remat_classes = self._get_classes_by_names(
      model, self.config.model.remat.get("activation_checkpoint_layers", [])
    )
    layers_to_scan = self.config.model.remat.get("scan_layers", None)
    offload_tensors = self.config.model.remat.get("offload_tensors", [])

    # Checking preconditions and logging.
    if remat_classes and is_main_process():
      logger.info(f"Enabling activation checkpointing on {remat_classes}")
    if layers_to_scan:
      assert isinstance(layers_to_scan, str)
      if is_main_process():
        logger.info(f"Compiling module `{layers_to_scan}` with scan")
    if len(offload_tensors):
      if is_main_process():
        logger.info(f"Will offload these tensors to host RAM: {offload_tensors}")
      if layers_to_scan is None:
        raise NotImplementedError("Host offloading requires scan")
      if len(remat_classes) != 1:
        raise NotImplementedError(
          "Host offloading requires checkpointing exactly one layer"
        )

    def maybe_checkpoint(mod, _name):
      if isinstance(mod, tuple(remat_classes)):
        return checkpoint_module(mod)
      return mod

    if layers_to_scan is None:
      # Implement activation checkpointing without scan by wrapping modules.
      if not remat_classes:
        return model
      return wrap_module(model, maybe_checkpoint)

    if not remat_classes:
      # Scan without activation checkpointing.
      return scan_layers.compile(model, layers_to_scan)

    # Implement activation checkpointing and host offloading under scan via
    # a graph partitioner instead of `checkpoint_module`.
    seq = model.get_submodule(layers_to_scan)
    assert isinstance(seq, HomogeneousSequential)
    if len(remat_classes) != 1 or list(remat_classes)[0] != seq.repeated_layer:
      raise NotImplementedError(
        f"When compiling decoder layers with scan and \
          activation checkpointing is also requested, we only support \
          checkpointing {seq.repeated_layer} i.e. the layer being scanned."
      )
    if not len(offload_tensors):
      partition_fn = remat_all.remat_all_partition_fn
    else:
      partition_fn = partial(
        offloading.remat_all_and_offload_these_inputs,
        names_to_offload=offload_tensors,
      )
    return scan_layers.compile(model, layers_to_scan, partition_fn=partition_fn)

  def _add_optimization_barrier_model(self, model: nn.Module):
    classes = self._get_classes_by_names(
      model, self.config.model.remat.get("optimization_barrier_layers", [])
    )
    if not classes:
      return model

    if is_main_process():
      logger.info(f"Adding backward optimization barriers to {classes}")

    def maybe_add_barrier(mod, _name):
      if isinstance(mod, tuple(classes)):
        # Register a backward hook to place optimization barrier to prevent
        # gigantic fusions on syncing the gradients.
        xs.apply_backward_optimization_barrier(mod)
        return mod
      return mod

    return wrap_module(model, maybe_add_barrier)

  def _get_classes_by_names(self, model, activation_checkpoint_layers: list[str]):
    classes_to_checkpoint = set()
    for layer_class in activation_checkpoint_layers:
      cls = get_module_class_from_name(model, layer_class)
      if cls is None:
        raise Exception(
          f"Could not find the transformer layer class {layer_class} in the model."
        )
      else:
        classes_to_checkpoint.add(cls)
    return tuple(classes_to_checkpoint)

  def train_loop(self):
    if self.config.checkpoint_load_step is not None:
      self._load_checkpoint()
    self.model.train()
    self.model.zero_grad()
    max_step = self.config.max_steps
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)

    metrics_logger = MetricsLogger(self.config.model)
    if is_main_process():
      logger.info("Starting training")
      logger.info(f"    Max step: {max_step}")
      logger.info(f"    Global batch size: {self.global_batch_size}")
      if hasattr(self, 'start_step') and self.start_step > 0:
        logger.info(f"    Resuming from step: {self.start_step}")

      wandb.login(key=os.environ.get("WANDB_API_KEY"), host="https://salesforceairesearch.wandb.io")
      run_name = self.config.run_name if hasattr(self.config, "run_name") and self.config.run_name is not None else self.config.model.model_class
      wandb.init(project="text-diffusion-model-research-qwen2_5-1_5b-pretrain", name=run_name)
      # Log the configuration to wandb
      wandb.config.update(OmegaConf.to_container(self.config, resolve=True))
      # Set wandb step to start_step if resuming from checkpoint
      if self.start_step > 0:
        wandb.log({}, step=self.start_step-1)  # Set the initial step counter
    # Initialize epoch and step counters, accounting for checkpoint loading
    epoch = 0
    start_step = self.start_step
    # Skip batches for partial file processing when resuming from checkpoint
    if self.config.checkpoint_load_step is not None and self.config.steps_to_skip == 0 and is_main_process():
      logger.warning("steps_to_skip is 0, but checkpoint_load_step is not None. This will cause the trainer to start from the beginning of the dataset. Please check the logs to see if this is expected.")
    if self.config.steps_to_skip > 0:
      if is_main_process():
        logger.info(f"Skipping {self.config.steps_to_skip} batches for partial file processing...")
      for _ in range(self.config.steps_to_skip):
        try:
          next(train_iterator)
        except StopIteration:
          epoch += 1
          train_iterator = iter(train_loader)
          next(train_iterator)

    if self.config.training_mode == "sft" and self.config.progress_src_mask:
      progress_src_mask_ratio = self.config.progress_src_mask_ratio
      progress_src_mask_steps = max(1.0, float(max_step) * progress_src_mask_ratio)
    else:
      progress_src_mask_steps = 0

    for step in range(start_step, max_step):
      data_load_start_time = timer()
      try:
        batch = next(train_iterator)
      except StopIteration:
        if is_main_process():
          logger.warning(f"DataLoader exhausted at step {step}, reset iterator")
        epoch += 1

        # If we just finished the resuming epoch and have all_data_files, recreate dataset with full data
        if hasattr(self.config, 'is_resuming_epoch') and self.config.is_resuming_epoch and hasattr(self.config, 'all_data_files'):
          if is_main_process():
            logger.info("Finished resuming epoch, switching to full dataset for subsequent epochs")
          self.config.is_resuming_epoch = False

          # Recreate dataset with all files
          use_webdataset = hasattr(self.config.data, 'use_webdataset') and self.config.data.use_webdataset
          if use_webdataset:
            self.train_dataset = retry(
              lambda: make_webdataset(
                self.config.data.dataset_name,
                shard_urls=self.config.all_data_files,
                seed=self.config.seed + epoch + step + start_step,
                checkpoint_dir=None
              )
            )
          else:
            self.train_dataset = retry(
              lambda: make_gcs_pretokenized_dataset(
                self.config.dataset_name,
                data_files=self.config.all_data_files,
                seed=self.config.seed + epoch + step + start_step,
                checkpoint_dir=None
              )
            )
          if isinstance(self.train_dataset, IterableDataset) and not isinstance(self.train_dataset, wds.WebDataset):
            try:
              logger.info(f"Applying split_dataset_by_node for device {xr.process_index()}/{xr.process_count()}")
              self.train_dataset = split_dataset_by_node(self.train_dataset, xr.process_index(), xr.process_count())
              logger.info(f"Dataset split successful for device {xr.process_index()}")
            except Exception as e:
              logger.warning(f"Dataset splitting failed: {e}. This may cause data duplication across devices.")  
        else:
          if isinstance(self.train_dataset, wds.WebDataset):
            self.train_dataset = self.train_dataset.shuffle(
              size=32768,
              seed=self.config.seed + epoch + step + start_step,
            )
          elif isinstance(self.train_dataset, IterableDataset):
            self.train_dataset = self.train_dataset.shuffle(
              buffer_size=32768,
              seed=self.config.seed + epoch + step + start_step
            )
          elif isinstance(self.train_dataset, HuggingFaceDataset):
            self.train_dataset = self.train_dataset.shuffle(
              seed=self.config.seed + epoch + step + start_step
            )
        # Recreate dataloader with the full dataset
        train_loader = self._get_train_dataloader()
        xm.wait_device_ops()
        torch_xla.sync()

        train_iterator = iter(train_loader)
        batch = next(train_iterator)
      data_load_end_time = timer()

      if step == 0 and is_main_process():
        logger.info(f"Batch shape: {batch['input_ids'].shape}")

      trace_start_time = timer()
      # Preprocess batch
      if self.config.training_mode == "sft":
        self._validate_sft_batch(batch)
        if self.config.progress_src_mask:
          ratio = step / progress_src_mask_steps
          ratio = max(0.0, min(1.0, ratio))
          if ratio <= 0.0:
            # All False at the very beginning
            batch["src_mask"] = torch.zeros_like(batch["src_mask"], dtype=torch.bool)
          elif ratio < 1.0:
            # Keep only the first floor(ratio * original_true_count) Trues per row,
            # turning the trailing Trues into False.
            batch_sz, seq_len_mask = batch["src_mask"].shape
            true_counts = batch["src_mask"].sum(dim=1)  # [batch]
            keep_len = torch.floor(true_counts.to(torch.float32) * ratio).to(torch.long)  # [batch]
            idx = torch.arange(seq_len_mask, device=batch["src_mask"].device).unsqueeze(0).expand(batch_sz, -1)
            batch["src_mask"] = idx < keep_len.unsqueeze(1)
      else:
        if self.config.reshape_context:
          batch["input_ids"] = batch["input_ids"].reshape(-1, 2048)
          if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].reshape(-1, 2048)

        # Create segment_ids from input_ids if in pretrain mode and segment_ids is None
        # Create segment_ids by looking at EOS_TOKEN_ID positions
        # NOTE: hardcode eos token id because the pretokenized dataset used this id
        EOS_TOKEN_ID = 151645 
        if self.config.seg_attn:
          eos_mask = torch.where(batch["input_ids"] == EOS_TOKEN_ID, 1, 0).int().to(batch["input_ids"].device)

          # Compute cumulative sum of EOS tokens to get segment IDs
          # Each EOS token increments the segment ID for subsequent tokens
          segment_ids = eos_mask.cumsum(dim=1)

          # Shift segment_ids to the right by 1 position so tokens before first EOS are segment 0
          # and tokens after each EOS get incremented segment IDs
          segment_ids = torch.cat(
              [torch.zeros_like(segment_ids[:, :1]), segment_ids[:, :-1]], dim=1
          )
          # NOTE: haolin
          # Convert to float to work around scan limitation with integer tensors
          # See: https://github.com/pytorch/xla/issues/8783
          segment_ids = segment_ids.float().requires_grad_(False)
          batch["segment_ids"] = segment_ids

      loss = self.train_step(batch)
      trace_end_time = timer()

      if step % self.config.logging_steps == 0:
        def step_closure(epoch, step, loss, trace_start_time, trace_end_time, data_load_start_time, data_load_end_time):
          loss = loss.detach().item()
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")
          if is_main_process():
            logger.info(
              f"Epoch: {epoch}, step: {step}, loss: {loss:0.4f}, "
              f"trace time: {(trace_end_time - trace_start_time) * 1000:0.2f} ms, "
              f"data load time: {(data_load_end_time - data_load_start_time) * 1000:0.2f} ms"
            )
            wandb.log(
              {
                "train/loss": loss,
                "train/ppl": math.exp(loss),
                "train/step_time": (trace_end_time - trace_start_time) * 1000,
                "train/data_load_time": (data_load_end_time - data_load_start_time) * 1000,
                "train/epoch": epoch,
                "train/step": step,
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/total_tokens": self.config.data.block_size * (step + 1) * self.global_batch_size,
              },
              step=step  # Explicitly set the wandb global step
            )
        xm.add_step_closure(
          step_closure,
          args=(epoch, step, loss, trace_start_time, trace_end_time, data_load_start_time, data_load_end_time),
          run_async=True,
        )
      if step > self.start_step and step % self.config.save_steps == 0:
        # NOTE: currently we save the checkpoint synchronously
        xm.wait_device_ops()  # Wait for all XLA operations to complete
        if is_main_process():
          logger.info(f"Processing unsharded tensors for checkpoint saving")
        unsharded_state_dict = {}
        for name, param in self.model.named_parameters():
          # Example logic to identify unsharded parameters
          # This may need to be adapted based on your specific sharding setup
          if param.ndim == 1:  # Assuming 1D tensors are unsharded
            unsharded_state_dict[name] = param.cpu() # Move to CPU for safety

        if is_main_process():
          logger.info(f"Processing sharded tensors for checkpoint saving")
        state_dict = {
          "model": self.model.state_dict(),
          "optimizer": self.optimizer.state_dict(),
          "scheduler": self.lr_scheduler.state_dict(),
          "masking_scheduler": self.masking_scheduler.state_dict(),
          "step": step,
        }
        try:
          # logger.info(f"model.state_dict().keys() before saving: {self.model.state_dict().keys()}")
          self.checkpoint_save_manager.save(step, state_dict, force=True)
          if is_main_process():
            logger.info(f"Unsharded state dict: {unsharded_state_dict.keys()}")
            mounted_save_dir = os.path.join(MOUNTED_GCS_DIR, self.checkpoint_save_dir.split(GCS_PREFIX)[1])
            torch.save(unsharded_state_dict, os.path.join(mounted_save_dir, f"unsharded_state_dict_{step}.pt"))
            logger.info(f"Checkpoint saved at step {step} to {self.checkpoint_save_dir}")
        except Exception as e:
          logger.error(f"Failed to save checkpoint at step with ckpt_mgr {step}: {e}")
        xm.wait_device_ops()

      # Capture profile at the prefer step
      if step == self.config.profile_step:
        # Wait until device execution catches up to tracing before triggering the profile. This will
        # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
        # for the step, the interruption will be minimal.
        xm.wait_device_ops()
        xp.trace_detached(
          "127.0.0.1:9012",
          self.config.profile_dir,
          self.config.profile_duration,
        )

    xm.wait_device_ops()
    logger.info("Finished training run")

    if self.config.profile_step >= 0:
      # Analyze the step duration from the latest profile
      step_duration = step_duration_from_latest_profile(self.config.profile_dir)
      metrics_logger.log_step_execution_time(step_duration)

      tpu_name = os.environ.get("TORCHPRIME_TPU_TYPE", None)
      if tpu_name:
        # Add "torch_dtype" in model config
        model_config_for_mfu = OmegaConf.to_container(self.config.model, resolve=True)
        model_config_for_mfu["torch_dtype"] = str(
          get_model_dtype(self.model)
        ).removeprefix("torch.")

        # Compute MFU
        mfu = compute_mfu(
          config=model_config_for_mfu,
          batch_size=self.config.global_batch_size,
          step_duration=step_duration,
          tpu_name=tpu_name,
          num_slices=get_num_slices(),
          sequence_length=self.config.block_size,
        )
        metrics_logger.log_mfu(mfu.mfu)

    # Print and save metrics
    metrics = metrics_logger.finalize()
    logger.info("***** train metrics *****\n%s", metrics)
    metrics.save(Path(self.config.output_dir) / "train_metrics.json")

  def _validate_sft_batch(self, batch):
    """Validate SFT batch before training step."""
    if "src_mask" not in batch:
      raise ValueError("src_mask not found in batch for SFT training")
    
    src_mask = batch["src_mask"]
    input_ids = batch["input_ids"]
    
    # Validate src_mask shape and content
    if src_mask.shape != input_ids.shape:
      raise ValueError(f"src_mask shape {src_mask.shape} doesn't match input_ids shape {input_ids.shape}")
    
    # Ensure we have at least some instruction tokens
    if src_mask.sum() == 0:
      raise ValueError("src_mask has no True values - no instruction tokens found")
    
    return True

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch):
    # Get current masking probabilities from scheduler
    masking_schedule = self.masking_scheduler.get_schedule()
    # logger.info(f"step: {self.masking_scheduler.current_step}, masking_schedule: {masking_schedule}")
    if self.config.training_mode == "sft":
      # For SFT, src_mask should already be in the batch from data collator
      _logits, loss = self.model(
        input_ids=batch["input_ids"],
        src_mask=batch["src_mask"],
        training_mode="sft",
        masking_schedule=masking_schedule
      )
    else:
      # Pre-training mode (original behavior)
      _logits, loss = self.model(
        **batch,
        masking_schedule=masking_schedule
      )
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.masking_scheduler.step()
    self.model.zero_grad()
    return loss


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):  
  # Configure logging (only on main process to avoid duplicate logs)
  if is_main_process():
    print(OmegaConf.to_yaml(config))  # Print the config for debugging
  
  log_level = logging.INFO
  logger.setLevel(log_level)
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Initialize distributed process group for XLA
  # torch.distributed.init_process_group('gloo', init_method='xla://')

  set_seed(config.seed)
  torch_xla.manual_seed(config.seed)
  
  # Start profiling server (only on main process)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = retry(lambda: AutoTokenizer.from_pretrained(tokenizer_name))

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
  # NOTE: read HF model from GCS bucket if checkpoint is not provided, otherwise read from checkpoint_load_dir/checkpoint_load_step in _load_checkpoint()
  load_from_checkpoint = config.checkpoint_load_dir is not None and config.checkpoint_load_step is not None
  with set_default_dtype(torch.bfloat16), torch_xla.device():
    model = initialize_model_class(config.model, load_from_hf=not load_from_checkpoint)
    

  n_params = sum([p.numel() for p in model.parameters()])
  if is_main_process():
    if load_from_checkpoint:
      logger.info(f"Continuing training on previous model checkpoint - Total size={n_params} params")
    else:
      logger.info(f"Training from scratch on pretrained model - Total size={n_params} params")

  if config.training_mode == "sft":
    # SFT mode: load instruction-response dataset
    if config.data.dataset_name:
      checkpoint_save_dir = os.path.join(MOUNTED_GCS_DIR, config.checkpoint_save_dir.split(GCS_PREFIX)[1])
      use_webdataset = hasattr(config.data, 'use_webdataset') and config.data.use_webdataset
      if use_webdataset:
        data = make_webdataset(
          config.data.dataset_name,
          seed=config.seed,
          checkpoint_dir=checkpoint_save_dir,
          columns=["input_ids", "src_mask"]
        )
      else:
        if isinstance(config.data.dataset_name, ListConfig):
          dataset_names = OmegaConf.to_container(config.data.dataset_name)
        else:
          dataset_names = config.data.dataset_name
        data = make_sft_dataset(
          dataset_names=dataset_names,
          tokenizer=tokenizer,
          block_size=config.data.block_size,
          seed=config.seed,
        )
    else:
      raise ValueError("No dataset provided for SFT")
  else:
    # Pre-training mode (original behavior)
    if config.data.dataset_name:
      # Downloading and loading a dataset from the hub.
      dataset_name = config.data.dataset_name
      if dataset_name.startswith(GCS_PREFIX):
        checkpoint_save_dir = os.path.join(MOUNTED_GCS_DIR, config.checkpoint_save_dir.split(GCS_PREFIX)[1])
        use_webdataset = hasattr(config.data, 'use_webdataset') and config.data.use_webdataset
        dataset_name = os.path.join(MOUNTED_GCS_DIR, dataset_name.split(GCS_PREFIX)[1])
        if not config.resume_from_checkpoint:
          if is_main_process():
            logger.info(f"Training from scratch, loading all data files from {dataset_name}")
          if use_webdataset:
             if is_main_process():
               logger.info(f"Building webdataset using {config.data.dataset_name}")
             data = retry(
               lambda: make_webdataset(
                 config.data.dataset_name,
                 seed=config.seed,
                 checkpoint_dir=checkpoint_save_dir
               )
             )
          else:
            data = retry(
              lambda: make_gcs_pretokenized_dataset(dataset_name, seed=config.seed, checkpoint_dir=checkpoint_save_dir)
            )
          # No additional steps to skip when starting fresh
          # Store dataset info for multi-epoch training
          config.all_data_files = None  # Will be set from data_files.json if exists
          config.is_resuming_epoch = False
        else:
          if is_main_process():
            logger.info(f"Resuming from checkpoint {config.checkpoint_load_step}, will recompute the data files to skip")
          # Calculate which files to skip based on checkpoint step and batch size
          samples_per_file = config.data.samples_per_file if hasattr(config.data, 'samples_per_file') and config.data.samples_per_file is not None else 100000
          total_samples_processed = config.checkpoint_load_step * config.global_batch_size
          files_to_skip, num_samples_processed_in_current_file = divmod(total_samples_processed, samples_per_file)
          if is_main_process():
            logger.info(f"Resuming from checkpoint step {config.checkpoint_load_step}")
            logger.info(f"Total samples processed: {total_samples_processed}")
            logger.info(f"Remaining samples in current file: {num_samples_processed_in_current_file}")

          # Calculate additional steps to skip within the current file
          steps_to_skip = num_samples_processed_in_current_file // config.global_batch_size
          if is_main_process():
            logger.info(f"Additional steps to skip in current file: {steps_to_skip}")
          config.steps_to_skip = steps_to_skip

          # Read data files from checkpoint directory
          checkpoint_load_dir = os.path.join(MOUNTED_GCS_DIR, config.checkpoint_load_dir.split(GCS_PREFIX)[1])
          data_files_path = os.path.join(checkpoint_load_dir, "data_files.json")

          with open(data_files_path, "r") as f:
            all_data_files = json.load(f)
          assert isinstance(all_data_files, list), "data_files.json should be a list"

          # Store all data files for subsequent epochs
          config.all_data_files = all_data_files
          config.dataset_name = dataset_name
          config.is_resuming_epoch = True

          # Skip the appropriate number of files for the current epoch
          files_to_skip = files_to_skip % len(all_data_files)
          remaining_files = all_data_files[files_to_skip:]
          if is_main_process():
            logger.info(f"Files to skip: {files_to_skip}")
            logger.info(f"Total data files: {len(all_data_files)}")
            logger.info(f"Remaining files after skipping: {len(remaining_files)}")

          # Load dataset starting from the appropriate files
          if use_webdataset:
            if is_main_process():
              logger.info(f"Building webdataset using {config.data.dataset_name}, loading remaining {len(remaining_files)} files from {remaining_files}")
            data = retry(
              lambda: make_webdataset(
                config.data.dataset_name,
                shard_urls=remaining_files,
                seed=config.seed,
                checkpoint_dir=None
              )
            )
          else:
            data = retry(
              lambda: make_gcs_pretokenized_dataset(
                dataset_name,
                data_files=remaining_files,
                seed=config.seed,
                checkpoint_dir=None  # Don't save data_files.json again
              )
            )
      else:
        # Initialize multi-epoch training flags for non-GCS datasets
        config.is_resuming_epoch = False
        config.all_data_files = None
        
        data = retry(
          lambda: make_huggingface_dataset(
            name=config.data.dataset_name,
            config_name=config.data.dataset_config_name,
            split="train",
            cache_dir=config.data.cache_dir,
            tokenizer=tokenizer,
            block_size=config.data.block_size,
          )
        )
    elif config.data.gcs_dataset_names:
      # Downloading and loading a dataset from GCS bucket.
      data = retry(
        lambda: make_gcs_dataset(
          names=config.data.gcs_dataset_names,
          weights=config.data.weights,
          tokenizer=tokenizer,
          seed=config.seed,
          block_size=config.data.block_size,
        )
      )
    else:
      raise ValueError("No dataset provided")

  if isinstance(data, IterableDataset) and not isinstance(data, wds.WebDataset):
    try:
      logger.info(f"Applying split_dataset_by_node for device {xr.process_index()}/{xr.process_count()}")
      data = split_dataset_by_node(data, xr.process_index(), xr.process_count())
      logger.info(f"Dataset split successful for device {xr.process_index()}")
    except Exception as e:
      logger.warning(f"Dataset splitting failed: {e}. This may cause data duplication across devices.")

  trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=data,
  )

  # Synchronize all processes before starting training
  xm.wait_device_ops()  # Wait for all XLA operations to complete
  if is_main_process():
    logger.info("All processes synchronized, starting training")

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    trainer.train_loop()


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
