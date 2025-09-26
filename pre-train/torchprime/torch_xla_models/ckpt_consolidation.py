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
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch_xla._internal.jax_workarounds import jax_env_context
from transformers import AutoTokenizer
from transformers.utils import check_min_version

from torchprime.utils.retry import retry
from torchprime.torch_xla_models.train import Trainer, initialize_model_class
from torchprime.torch_xla_models.model_utils import save_sharded_safetensors_by_layer

MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)

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

  trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=None,
  )

  # Synchronize all processes before starting training
  xm.wait_device_ops()  # Wait for all XLA operations to complete
  if is_main_process():
    logger.info("All processes synchronized, starting checkpoint consolidation")

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    gcs_prefix = "gs://sfr-text-diffusion-model-research/"
    
    # Parse checkpoint_load_step as comma-separated list
    if isinstance(config.checkpoint_load_step, str):
      checkpoint_steps = [step.strip() for step in config.checkpoint_load_step.split(',')]
    elif isinstance(config.checkpoint_load_step, list):
      checkpoint_steps = config.checkpoint_load_step
    elif isinstance(config.checkpoint_load_step, ListConfig):
      checkpoint_steps = list(config.checkpoint_load_step)
    else:
      checkpoint_steps = [config.checkpoint_load_step]
    
    if is_main_process():
      logger.info("Consolidating %d checkpoints: %s", len(checkpoint_steps), checkpoint_steps)
    
    model_sd = model.state_dict()
    
    # Consolidate each checkpoint separately
    for i, checkpoint_step in enumerate(checkpoint_steps):
      if is_main_process():
        logger.info("Consolidating checkpoint %d/%d: step %s", i+1, len(checkpoint_steps), checkpoint_step)
      
      # Create separate output directory for each checkpoint
      save_dir = Path(MOUNTED_GCS_DIR) / config.checkpoint_load_dir.split(gcs_prefix)[1].replace("checkpoints", "consolidated_checkpoints") / f"{checkpoint_step}"
      
      possible_unsharded_model_sd_path = Path(MOUNTED_GCS_DIR) / config.checkpoint_load_dir.split(gcs_prefix)[1] / f"unsharded_state_dict_{checkpoint_step}.pt"
      unsharded_model_sd = {}
      for name, param in model_sd.items():
        if param.ndim == 1:
          unsharded_model_sd[name] = param.cpu()
      if unsharded_model_sd and possible_unsharded_model_sd_path.exists():
        unsharded_model_sd = torch.load(possible_unsharded_model_sd_path)

      if possible_unsharded_model_sd_path.exists():
        reload_sd = {
          "model": {
            name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            for name, tensor in model_sd.items() if name not in unsharded_model_sd
          }
        }
      else:
        reload_sd = {"model": model_sd}
      trainer.checkpoint_load_manager.restore(checkpoint_step, reload_sd)
      if possible_unsharded_model_sd_path.exists():
        reload_sd["model"].update(unsharded_model_sd)
      cpu_state = {k.replace("._orig_mod", ""): v for k, v in reload_sd["model"].items()}

      if is_main_process():
        logger.info("Checkpoint %s loaded, starting consolidation", checkpoint_step)
      
      torch_xla.sync()
      xm.wait_device_ops()
      
      if is_main_process():
        try:
          tmp_dir = tempfile.mkdtemp(dir="/mnt/localssd")
          logger.info("Using local SSD for safetensors shards: %s", tmp_dir)
        except (FileNotFoundError, PermissionError):
          tmp_dir = tempfile.mkdtemp()
          logger.info("Using default temp directory for safetensors shards: %s", tmp_dir)

        save_sharded_safetensors_by_layer(cpu_state, str(save_dir), tmp_dir=tmp_dir)
        logger.info("Safetensors shards + index written to %s", save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("Checkpoint %s consolidation complete", checkpoint_step)
      
      xm.rendezvous(f"checkpoint_consolidation_barrier_{checkpoint_step}")
    
    if is_main_process():
      logger.info("All checkpoint consolidations complete")

if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
