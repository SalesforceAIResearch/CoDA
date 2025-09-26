import copy
from dataclasses import dataclass

import pytest
import torch
import torch.test
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import MixtralForCausalLM as HfMixtralForCausalLM

from torchprime.torch_xla_models.mixtral import MixtralForCausalLM


@dataclass
class MixtralFixture:
  vocab_size: int
  hf_model: HfMixtralForCausalLM
  model: MixtralForCausalLM


def get_mixtral_8x7b() -> MixtralFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config = AutoConfig.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
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
      "vocab_size": vocab_size,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 1,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "max_position_embeddings": 32768,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "num_experts_per_tok": 2,
      "num_local_experts": 8,
      "rope_theta": 1000000.0,
      "router_aux_loss_coef": 0.02,
      "attention_dropout": 0.0,
      "attention_bias": False,
      "attention_kernel": None,
      "moe_implementation": "static",
    }
  )
  # place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfMixtralForCausalLM(config)
    model = MixtralForCausalLM(torchprime_config)
    model.load_state_dict(hf_model.state_dict())

  return MixtralFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_mixtral_8x7b],
  ids=["Mixtral 8x7B"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
def test_forward_our_model_against_hf_model(fixture, transform):
  fixture = fixture()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  input_sizes = [8, 128, 256]
  for input_size in input_sizes:
    input = torch.randint(128, ((2, input_size // 2))).to(device)
    hf_output = hf_model_xla(input, labels=input, attention_mask=torch.ones_like(input))
    mixtral_xla_logits, mixtral_xla_loss = model_xla(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    torch_xla.sync()
    torch.testing.assert_close(
      hf_output.logits,
      mixtral_xla_logits,
      atol=1e-6,
      rtol=1e-9,
      msg="logits are not equal",
    )
    torch.testing.assert_close(
      hf_output.loss, mixtral_xla_loss, atol=1e-6, rtol=1e-9, msg="loss is not equal"
    )


@pytest.mark.parametrize(
  "fixture",
  [get_mixtral_8x7b],
  ids=["Mixtral 8x7B"],
)
def test_forward_torch_xla_against_native(fixture):
  fixture = fixture()
  input_size = 8
  device = torch.device("cpu")
  input = torch.randint(fixture.vocab_size, ((2, input_size // 2)))
  mixtral_native_logits, mixtral_native_loss = fixture.model(
    input, labels=input, attention_mask=torch.ones_like(input)
  )

  device = torch_xla.device()
  input = input.to(device)
  model_xla = copy.deepcopy(fixture.model).to(device)
  torch_xla.sync()

  mixtral_xla_logits, mixtral_xla_loss = model_xla(
    input, labels=input, attention_mask=torch.ones_like(input)
  )
  torch_xla.sync()
  torch.testing.assert_close(
    mixtral_native_logits,
    mixtral_xla_logits.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    mixtral_native_loss,
    mixtral_native_loss.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run loss is not equal",
  )
