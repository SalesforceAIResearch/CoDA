"""
Calculate Model FLOPs Utilization (MFU).
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MFU:
  model_tflops: float
  """The number of floating point operations in the model, in teraflops."""

  hardware_tflops_per_step: float
  """The theoretical hardware floating point throughput during one training step, in teraflops."""

  per_chip_tflops_per_sec: float
  """The realized floating point throughput during one second in each chip, in teraflops."""

  mfu: float
  """Model FLOPs Utilization. Fraction of hardware FLOPs the model uses, from 0 to 1."""


@dataclass
class Config:
  per_device_batch_size: int
  max_target_length: int
  mlp_dim: int
  emb_dim: int
  mlp_activations: list[Any]
  num_experts: int
  num_experts_per_tok: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  num_decoder_layers: int
  vocab_size: int
  gradient_accumulation_steps: int


# Taken from https://github.com/google/maxtext/blob/bd50865c7c66bcf7d12870b8919e59e2a8ebb906/MaxText/maxtext_utils.py#L123
def calculate_tflops_training_per_device(config: Config, log=True):
  """Calculate training TFLOP"""
  ffn1_flops = (
    2
    * config.per_device_batch_size
    * config.max_target_length
    * config.mlp_dim
    * config.emb_dim
    * len(config.mlp_activations)
  )
  ffn2_flops = (
    2
    * config.per_device_batch_size
    * config.max_target_length
    * config.mlp_dim
    * config.emb_dim
  )
  total_ffn_flops = ffn1_flops + ffn2_flops

  if config.num_experts > 1:
    # MoE: brute force implementation
    gate_flops = (
      2
      * config.per_device_batch_size
      * config.max_target_length
      * config.emb_dim
      * config.num_experts
    )
    total_ffn_flops = gate_flops + config.num_experts_per_tok * total_ffn_flops

  qkv_flops = (
    2
    * config.per_device_batch_size
    * config.max_target_length
    * config.emb_dim
    * (config.num_query_heads + 2 * config.num_kv_heads)
    * config.head_dim
  )
  attention_flops = (
    4
    * config.per_device_batch_size
    * config.max_target_length**2
    * config.num_query_heads
    * config.head_dim
  )
  projection_flops = (
    2
    * config.per_device_batch_size
    * config.max_target_length
    * config.emb_dim
    * config.num_query_heads
    * config.head_dim
  )
  embedding_flops = (
    2
    * config.per_device_batch_size
    * config.max_target_length
    * config.emb_dim
    * config.vocab_size
  )

  # Multiply by 3 for both feed forward and back propagation flops.
  # In most transformer training implementations, the backward pass has roughly
  # twice the flops of the forward pass, because the backward pass computes
  # gradients with respect to both the weights and the activations.
  learnable_weight_tflops = (
    (
      (total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers
      + embedding_flops
    )
    * 3
    / 10**12
  )
  # Megatron tflops calculation does not account for causality in attention
  attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12

  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps
  total_tflops = learnable_weight_tflops + attention_tflops

  if log:
    print(
      "Per train step:\n",
      f"Total TFLOPs: {total_tflops:.2f} \n",
      f"split as {100 * learnable_weight_tflops / total_tflops:.2f}% learnable weight flops",
      f"and {100 * attention_tflops / total_tflops:.2f}% attention flops",
    )
  return total_tflops


def compute_mfu(
  config: dict,
  batch_size: int,
  sequence_length: int,
  step_duration: float,
  tpu_name: str,
  num_slices: int = 1,
  gradient_accumulation_steps: int = 1,
) -> MFU:
  """
  Calculate MFU of a training config on some TPU hardware.

  Args:

    config: a dictionary representing a decoded JSON HuggingFace-style model config.

    batch_size: global batch size.

    sequence_length: number of tokens in each training example.

    step_duration: duration of one trthroughput_per_deviceaining step.

    tpu_name: accelerator type (e.g. `v5p-128`).

    gradient_accumulation_steps: how many dataloader iterations per optimizer iteration. See \
      https://huggingface.co/docs/accelerate/v0.11.0/en/gradient_accumulation. Defaults to 1.

  """
  total_tflops = calculate_tflops_training_per_device(
    Config(
      per_device_batch_size=batch_size,
      max_target_length=sequence_length,
      mlp_dim=int(config["intermediate_size"]),
      emb_dim=int(config["hidden_size"]),
      mlp_activations=["silu", "linear"],
      num_experts=int(config.get("num_local_experts", 1)),
      num_experts_per_tok=int(config.get("num_experts_per_tok", 1)),
      num_query_heads=int(config["num_attention_heads"]),
      num_kv_heads=int(config["num_key_value_heads"]),
      head_dim=int(config["hidden_size"] / config["num_attention_heads"]),
      num_decoder_layers=int(config["num_hidden_layers"]),
      vocab_size=int(config["vocab_size"]),
      gradient_accumulation_steps=gradient_accumulation_steps,
    ),
    log=False,
  )

  dtype = config.get("torch_dtype")
  assert dtype == "bfloat16", f"Unsupported dtype {dtype}"

  chip_count_per_slice, tflops_per_chip = get_num_chips_and_tflops_per_chip(tpu_name)

  chip_count = chip_count_per_slice * num_slices
  hw_tflops = step_duration * chip_count * tflops_per_chip
  return MFU(
    model_tflops=total_tflops,
    hardware_tflops_per_step=hw_tflops,
    per_chip_tflops_per_sec=total_tflops / step_duration / chip_count,
    mfu=total_tflops / hw_tflops,
  )


def get_num_chips_and_tflops_per_chip(tpu_name: str) -> tuple[int, int]:
  """
  Determines the number of chips and TFLOPs per chip for a given TPU type.

  Args:
    tpu_name: The name of the TPU (e.g., "v4-8", "v5p-256").

  Returns:
    A tuple containing:
      - chip_count (int): The number of physical TPU chips.
      - tflops_per_chip (int): The peak TFLOPs (BF16) per chip.
  """
  version, core_count = parse_tpu_name(tpu_name)
  match version:
    case "v4":
      # See https://cloud.google.com/tpu/docs/v4
      chip_count = core_count / 2
      tflops_per_chip = 275
    case "v5p":
      # See https://cloud.google.com/tpu/docs/v5p
      chip_count = core_count / 2
      tflops_per_chip = 459
    case "v5e":
      # See https://cloud.google.com/tpu/docs/v5e
      chip_count = core_count
      tflops_per_chip = 197
    case "v6e":
      # See https://cloud.google.com/tpu/docs/v6e
      chip_count = core_count
      tflops_per_chip = 918
    case _:
      raise ValueError(f"Unsupported accelerator type {tpu_name}")
  return chip_count, tflops_per_chip


def parse_tpu_name(s) -> tuple[str, int]:
  match = re.search(r"(v..)-(\d+)", s)
  if match:
    return match.group(1), int(match.group(2))
  match = re.search(r"(v4)-(\d+)", s)
  if match:
    return match.group(1), int(match.group(2))
  raise ValueError(f"No matching pattern found in {s}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate MFU (CLI example).")

  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Model config path (it should be a HuggingFace-style JSON file)",
  )
  parser.add_argument("--batch-size", type=int, required=True, help="Size of the batch")
  parser.add_argument(
    "--step-duration", type=float, required=True, help="Duration of one step in seconds"
  )
  parser.add_argument(
    "--seq-len",
    type=float,
    required=True,
    help="Number of tokens in each training example",
  )
  parser.add_argument(
    "--tpu-name", type=str, required=True, help="Name of the TPU (e.g. v5p-128)"
  )

  args = parser.parse_args()

  global_batch_size = args.batch_size
  seq_len = args.seq_len
  step_duration = args.step_duration

  mfu = compute_mfu(
    config=json.loads((Path(args.config)).read_text()),
    batch_size=global_batch_size,
    sequence_length=seq_len,
    step_duration=step_duration,
    tpu_name=args.tpu_name,
  )

  print(f"Model teraflops: {mfu.model_tflops}", file=sys.stderr)
  print(f"Hardware teraflops: {mfu.hardware_tflops_per_step}", file=sys.stderr)

  print(mfu.mfu)
