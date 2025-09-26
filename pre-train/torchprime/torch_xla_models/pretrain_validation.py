"""
RUN ON v4-8 ONLY
"""
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import tempfile

import datasets
import hydra
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import torch_xla.experimental.distributed_checkpoint as xc
import transformers
import wandb
import math
import webdataset as wds
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch_xla._internal.jax_workarounds import jax_env_context
from transformers import AutoTokenizer
from transformers.utils import check_min_version
from torchprime.utils.retry import retry
from torchprime.torch_xla_models.train import Trainer, initialize_model_class
from torchprime.torch_xla_models.model_utils import save_sharded_safetensors_by_layer
from torchprime.data.dataset import make_huggingface_dataset, make_gcs_dataset, make_gcs_pretokenized_dataset
from torchprime.data.webdataset import make_webdataset, webdataset_collate_fn


MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)
GCS_PREFIX = "gs://sfr-text-diffusion-model-research/"

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True

if not dist.is_initialized():
  dist.init_process_group(backend='gloo', init_method='xla://')

def is_main_process():
  """Check if this is the main process (rank 0)."""
  return xr.process_index() == 0


@contextmanager
def set_default_dtype(dtype):
  # Get the current default dtype
  previous_dtype = torch.get_default_dtype()
  # Set the new default dtype
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    # Revert to the original default dtype
    torch.set_default_dtype(previous_dtype)


class ValidationTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  @torch_xla.compile(full_graph=True)
  def validation_step(self, batch):
    _, loss = self.model(**batch)
    return loss

  def validation_loop(self):
    if self.config.checkpoint_load_step is not None:
      self._load_checkpoint()
    self.model.train()
    self.model.zero_grad()

    max_step = self.config.max_steps
    eval_loader = self._get_eval_dataloader()
    eval_iterator = iter(eval_loader)

    if is_main_process():
      wandb.login(key=os.environ.get("WANDB_API_KEY"), host="https://salesforceairesearch.wandb.io")
      run_name = self.config.run_name if hasattr(self.config, "run_name") and self.config.run_name is not None else f"{self.config.model.model_class}_validation"
      wandb.init(project="text-diffusion-model-research-qwen2_5-1_5b-pretrain", name=run_name)
      wandb.config.update(OmegaConf.to_container(self.config, resolve=True))

    epoch = 0
    accumulated_loss = 0
    for step in range(max_step):
      try:
        batch = next(eval_iterator)
      except StopIteration:
        epoch += 1
        if is_main_process():
          logger.warning(f"DataLoader exhausted at step {step}, reset iterator")
        eval_iterator = iter(eval_loader)
        batch = next(eval_iterator)
      

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

      loss = self.validation_step(batch)
      accumulated_loss += loss.detach().item()

      if step % self.config.logging_steps == 0:
        def step_closure(epoch, step, loss):
          loss = loss.detach().item()
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")
          if is_main_process():
            logger.info(
              f"Epoch: {epoch}, step: {step}, loss: {loss:0.4f}, "
            )
            wandb.log(
              {
                "validation/loss": loss,
                "validation/ppl": math.exp(loss),
                "validation/epoch": epoch,
                "validation/step": step,
              },
              step=step  # Explicitly set the wandb global step
            )

        xm.add_step_closure(
          step_closure,
          args=(epoch, step, loss),
          run_async=True,
        )

      xm.wait_device_ops()

    xm.wait_device_ops()
    if is_main_process():
      logger.info("Finished validation run")
      logger.info(f"Average loss over {max_step} steps: {accumulated_loss / max_step}")
      wandb.log(
        {
          "validation/average_loss": accumulated_loss / max_step,
        },
      )


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


  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = retry(lambda: AutoTokenizer.from_pretrained(tokenizer_name))

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
    # NOTE: read HF model from GCS bucket if checkpoint_load_step is not provided, otherwise read from checkpoint_dir in _load_checkpoint()
  load_from_checkpoint = hasattr(config, 'checkpoint_load_step') and config.checkpoint_load_step is not None
  with set_default_dtype(torch.bfloat16), torch_xla.device():
    model = initialize_model_class(config.model, load_from_hf=not load_from_checkpoint)

  if config.data.dataset_name:
    # Downloading and loading a dataset from the hub.
    dataset_name = config.data.dataset_name
    use_webdataset = hasattr(config.data, 'use_webdataset') and config.data.use_webdataset
    dataset_name = os.path.join(MOUNTED_GCS_DIR, dataset_name.split(GCS_PREFIX)[1])
    if use_webdataset:
      data = retry(
        lambda: make_webdataset(config.data.dataset_name, seed=config.seed, checkpoint_dir=None)
      )
    else:
      data = retry(
        lambda: make_gcs_pretokenized_dataset(dataset_name, seed=config.seed, checkpoint_dir=None)
      )
  else:
    raise ValueError("No dataset provided")



  trainer = ValidationTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    eval_dataset=data,
  )

  # Synchronize all processes before starting training
  xm.wait_device_ops()  # Wait for all XLA operations to complete
  if is_main_process():
    logger.info("All processes synchronized, starting validation")

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    trainer.validation_loop()


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
