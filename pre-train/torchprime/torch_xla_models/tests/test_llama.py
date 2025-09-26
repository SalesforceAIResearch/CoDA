import copy
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.test
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import LlamaForCausalLM as HfLlamaForCausalLM

from torchprime.torch_xla_models.llama import LlamaForCausalLM


@dataclass
class LlamaFixture:
  vocab_size: int
  hf_model: HfLlamaForCausalLM
  model: LlamaForCausalLM


def get_llama_3_8b() -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config = AutoConfig.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    head_dim=64,
    num_hidden_layers=1,
    num_attention_heads=8,
    hidden_size=512,
    intermediate_size=64,
    vocab_size=vocab_size,
  )
  config.flash_attention = False
  torchprime_config = OmegaConf.create(
    {
      "vocab_size": 128,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 1,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "hidden_act": "silu",
      "max_position_embeddings": 8192,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": None,
      "rope_theta": 500000.0,
    }
  )
  # Place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfLlamaForCausalLM(config)
    model = LlamaForCausalLM(torchprime_config)
    model.load_state_dict(hf_model.state_dict())
  return LlamaFixture(vocab_size, hf_model, model)


def get_llama_3_1_405b() -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 256
  config = AutoConfig.from_pretrained(
    "meta-llama/Meta-Llama-3.1-405B",
    head_dim=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    hidden_size=512,
    intermediate_size=64,
    vocab_size=vocab_size,
  )
  config.flash_attention = False
  torchprime_config = OmegaConf.create(
    {
      "vocab_size": 256,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "hidden_act": "silu",
      "max_position_embeddings": 131072,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": None,
      "rope_theta": 500000.0,
      "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
      },
    }
  )
  # Place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfLlamaForCausalLM(config)
    model = LlamaForCausalLM(torchprime_config)
    # Assert that the `inv_freq` values are the same
    assert isinstance(model.model.layers[0].self_attn, nn.Module)
    assert isinstance(hf_model.model.layers[0].self_attn, nn.Module)
    assert isinstance(model.model.rotary_emb, nn.Module)
    assert isinstance(hf_model.model.rotary_emb, nn.Module)
    torch.testing.assert_close(
      model.model.rotary_emb.inv_freq,
      hf_model.model.rotary_emb.inv_freq,
    )
    # In this simplified model architecture, hidden_size 512 / num_attention_heads 8 = 64 head dim,
    # and the inv_freq size is half of the head dim.
    assert model.model.rotary_emb.inv_freq.shape == (32,)
    model.load_state_dict(hf_model.state_dict())
  return LlamaFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_llama_3_8b, get_llama_3_1_405b],
  ids=["Llama 3.0 8B", "Llama 3.1 405B"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
def test_forward_our_model_against_hf_model(fixture, transform):
  # Arrange
  fixture = fixture()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  input_sizes = [8, 128, 256]
  for input_size in input_sizes:
    input = torch.randint(fixture.vocab_size, ((2, input_size // 2))).to(device)
    # Act
    hf_output = hf_model_xla(input, labels=input, attention_mask=torch.ones_like(input))
    llama_xla_logits, llama_xla_loss = model_xla(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    torch_xla.sync()
    # Assert
    torch.testing.assert_close(
      hf_output.logits,
      llama_xla_logits,
      atol=1e-6,
      rtol=1e-9,
      msg="logits are not equal",
    )
    torch.testing.assert_close(
      hf_output.loss, llama_xla_loss, atol=1e-6, rtol=1e-9, msg="loss is not equal"
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_3_8b, get_llama_3_1_405b],
  ids=["Llama 3.0 8B", "Llama 3.1 405B"],
)
def test_forward_torch_xla_against_native(fixture):
  # Arrange
  fixture = fixture()
  input_size = 8
  device = torch.device("cpu")
  input = torch.randint(fixture.vocab_size, ((2, input_size // 2)))
  # Act
  llama_native_logits, llama_native_loss = fixture.model(
    input, labels=input, attention_mask=torch.ones_like(input)
  )

  device = torch_xla.device()
  input = input.to(device)
  model_xla = copy.deepcopy(fixture.model).to(device)
  torch_xla.sync()

  llama_xla_logits, llama_xla_loss = model_xla(
    input, labels=input, attention_mask=torch.ones_like(input)
  )
  torch_xla.sync()
  # Assert
  torch.testing.assert_close(
    llama_native_logits,
    llama_xla_logits.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    llama_native_loss,
    llama_xla_loss.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run loss is not equal",
  )
