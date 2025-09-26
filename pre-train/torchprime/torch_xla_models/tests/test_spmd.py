import copy
import functools
import unittest

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
from omegaconf import OmegaConf

from torchprime.torch_xla_models.llama import LlamaForCausalLM
from torchprime.torch_xla_models.llama.model import LlamaDecoderLayer
from torchprime.torch_xla_models.mixtral import MixtralForCausalLM
from torchprime.torch_xla_models.mixtral.model import MixtralDecoderLayer


class TestConfigSpmd(unittest.TestCase):
  """
  Test that the config based sharder has identical behavior to FSDPv2.

  Specifically:
  - Model weights have the same sharding spec
  - Outputs have the same sharding spec and are numerically close
  - Gradients have the same sharding spec and are numerically close
  """

  @classmethod
  def setUpClass(cls):
    import torch_xla.runtime as xr

    xr.use_spmd()

    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    torch_xla._XLAC._xla_set_mat_mul_precision("highest")

  def test_llama_config_sharding_against_fsdp_v2(self):
    import numpy as np
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
    # work on PyTorch/XLA. We must run this on the TPU for now.
    if xr.device_type() != "TPU":
      pytest.skip("This test only works on TPU")

    super().setUp()
    vocab_size = 128256
    torchprime_config = OmegaConf.create(
      {
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "initializer_range": 0.02,
        "rms_norm_eps": 1.0e-05,
        "attention_dropout": False,
        "attention_bias": False,
        "flash_attention": True,
        "rope_theta": 500000.0,
      }
    )
    # Place model on CPU device first
    with torch.device("cpu"):
      model = LlamaForCausalLM(torchprime_config)

    # Define mesh for test
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices, 1, 1)
    assert num_devices > 1, "The TPU VM should have more than 1 device for SPMD testing"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("data", "fsdp", "tensor", "expert"))
    xs.set_global_mesh(mesh)

    # Create random input and label of batch size 8, sequence length 256.
    input = torch.randint(vocab_size, ((8, 256)), device=torch_xla.device())
    xs.mark_sharding(input, mesh, ("fsdp", None))
    labels = torch.randint(vocab_size, ((8, 256)), device=torch_xla.device())
    xs.mark_sharding(labels, mesh, ("fsdp", None))
    torch_xla.sync()

    # Shard our model with config based sharding
    sharding_config = {
      # Weights
      "model.embed_tokens.weight": ["fsdp", None],
      "model.layers.*.self_attn.q_proj.weight": ["fsdp", None],
      "model.layers.*.self_attn.k_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.v_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.o_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.gate_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.up_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.down_proj.weight": [None, "fsdp"],
      "model.layers.*.input_layernorm.weight": ["fsdp"],
      "model.layers.*.post_attention_layernorm.weight": ["fsdp"],
      "model.norm.weight": ["fsdp"],
      "lm_head.weight": ["fsdp", None],
      # Activations
      "model.layers.*": ["fsdp", None, None],
      "lm_head": ["fsdp", None, None],
    }
    from torchprime.sharding.shard_model import shard_torch_xla_model_from_config

    model_config_sharded = shard_torch_xla_model_from_config(
      copy.deepcopy(model).to("xla"), config=sharding_config
    )
    torch_xla.sync()

    # Shard model with FSDPv2
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
      SpmdFullyShardedDataParallel as FSDPv2,
    )

    auto_wrap_policy = functools.partial(
      transformer_auto_wrap_policy,
      # Transformer layer class to wrap
      transformer_layer_cls={LlamaDecoderLayer},
    )
    model_fsdp_v2_sharded = FSDPv2(
      copy.deepcopy(model),
      shard_output=shard_output,
      auto_wrap_policy=auto_wrap_policy,
    )
    torch_xla.sync()

    assert_same_output_weights_grad(
      model_config_sharded, model_fsdp_v2_sharded, input, labels
    )

  def test_mixtral_config_sharding_against_fsdp_v2(self):
    import numpy as np
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
    # work on PyTorch/XLA. We must run this on the TPU for now.
    if xr.device_type() != "TPU":
      pytest.skip("This test only works on TPU")

    super().setUp()
    vocab_size = 32000
    torchprime_config = OmegaConf.create(
      {
        "vocab_size": vocab_size,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "num_hidden_layers": 2,
        "max_position_embeddings": 32768,
        "num_attention_heads": 32,
        "num_experts_per_tok": 2,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.02,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "flash_attention": True,
        "moe_implementation": "gmm",
      }
    )
    # Place model on CPU device first
    with torch.device("cpu"):
      model = MixtralForCausalLM(torchprime_config)

    # Define mesh for test
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices, 1, 1)
    assert num_devices > 1, "The TPU VM should have more than 1 device for SPMD testing"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("data", "fsdp", "tensor", "expert"))
    xs.set_global_mesh(mesh)

    # Create random input and label of batch size 8, sequence length 256.
    input = torch.randint(vocab_size, ((8, 256)), device=torch_xla.device())
    xs.mark_sharding(input, mesh, ("fsdp", None))
    labels = torch.randint(vocab_size, ((8, 256)), device=torch_xla.device())
    xs.mark_sharding(labels, mesh, ("fsdp", None))
    torch_xla.sync()

    # Shard our model with config based sharding
    sharding_config = {
      # Weights
      "model.embed_tokens.weight": ["fsdp", None],
      "model.layers.*.self_attn.q_proj.weight": ["fsdp", None],
      "model.layers.*.self_attn.k_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.v_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.o_proj.weight": ["fsdp", None],
      "model.layers.*.block_sparse_moe.gate.weight": [None, "fsdp"],
      "model.layers.*.block_sparse_moe.experts.w1": [None, None, "fsdp"],
      "model.layers.*.block_sparse_moe.experts.w2": [None, "fsdp", None],
      "model.layers.*.block_sparse_moe.experts.w3": [None, None, "fsdp"],
      "model.layers.*.input_layernorm.weight": ["fsdp"],
      "model.layers.*.post_attention_layernorm.weight": ["fsdp"],
      "model.norm.weight": ["fsdp"],
      "lm_head.weight": ["fsdp", None],
      # Activations
      # Shard the first element of decoder output
      "model.layers.*[0]": ["fsdp", None, None],
      "lm_head": ["fsdp", None, None],
    }
    from torchprime.sharding.shard_model import shard_torch_xla_model_from_config

    model_config_sharded = shard_torch_xla_model_from_config(
      copy.deepcopy(model).to("xla"), config=sharding_config
    )
    torch_xla.sync()

    # Shard model with FSDPv2
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
      SpmdFullyShardedDataParallel as FSDPv2,
    )

    auto_wrap_policy = functools.partial(
      transformer_auto_wrap_policy,
      # Transformer layer class to wrap
      transformer_layer_cls={MixtralDecoderLayer},
    )
    model_fsdp_v2_sharded = FSDPv2(
      copy.deepcopy(model),
      shard_output=shard_output,
      auto_wrap_policy=auto_wrap_policy,
    )
    torch_xla.sync()

    assert_same_output_weights_grad(
      model_config_sharded, model_fsdp_v2_sharded, input, labels
    )


def assert_same_output_weights_grad(
  model_config_sharded, model_fsdp_v2_sharded, input, labels
):
  """
  Asserts that two models have the same output, weights, and gradients, in terms of
  numerics and sharding specs.
  """
  # Run the model and backwards
  config_logits, config_loss = model_config_sharded(
    input, labels=labels, attention_mask=torch.ones_like(input)
  )
  config_loss.backward()
  torch_xla.sync()

  fsdp_logits, fsdp_loss = model_fsdp_v2_sharded(
    input, labels=labels, attention_mask=torch.ones_like(input)
  )
  fsdp_loss.backward()
  torch_xla.sync()

  # Check sharding and numeric accuracy.
  assert_same_value_and_sharding(
    config_logits,
    fsdp_logits,
    msg="Config sharded and FSDP v2 sharded logits are not equal",
  )
  assert_same_value_and_sharding(
    config_loss,
    fsdp_loss,
    msg="Config sharded and FSDP v2 sharded loss are not equal",
  )

  # Check model weights and gradients.
  for (p1_name, p1), (p2_name, p2) in zip(
    model_config_sharded.named_parameters(),
    model_fsdp_v2_sharded.named_parameters(),
    strict=True,
  ):
    # Because both config sharding and FSDPv2 adds wrapper modules, the module
    # tree might be different. we should at least assert that the last name
    # component matches.
    assert p1_name.split(".")[-1] == p2_name.split(".")[-1]
    assert_same_value_and_sharding(p1, p2, msg=f"{p1_name} and {p2_name} are not equal")

    assert p1.grad is not None
    assert p2.grad is not None
    assert_same_value_and_sharding(
      p1.grad,
      p2.grad,
      msg="Config sharded and FSDP v2 sharded gradients are not equal",
    )


def assert_same_value_and_sharding(actual, expected, msg):
  torch.testing.assert_close(
    actual.cpu(),
    expected.cpu(),
    msg=msg,
  )
  actual_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(actual)
  expected_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(expected)
  assert actual_sharding_spec == expected_sharding_spec


def shard_output(output, mesh):
  real_output = None
  if isinstance(output, torch.Tensor):
    real_output = output
  elif isinstance(output, tuple):
    real_output = output[0]
  else:
    raise RuntimeError("Unsupported")
  xs.mark_sharding(real_output, mesh, ("fsdp", None, None))


if __name__ == "__main__":
  unittest.main()
