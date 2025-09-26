import copy
from dataclasses import dataclass

import pytest
import torch
import torch.test
import torch_xla
import transformers.models.llama4.modeling_llama4 as hf_modeling_llama4
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig
from transformers import Llama4ForCausalLM as HfLlama4ForCausalLM

import torchprime.torch_xla_models.llama4.model as modeling_llama4
from torchprime.torch_xla_models.llama4 import Llama4TextForCausalLM


@dataclass
class LlamaFixture:
  vocab_size: int
  hf_model: HfLlama4ForCausalLM
  model: Llama4TextForCausalLM
  model_config: DictConfig


def get_llama_4_text_dummy_model() -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config = AutoConfig.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E",
  )
  config.text_config.num_hidden_layers = 1
  config.text_config.vocab_size = vocab_size
  config.text_config.head_dim = 64
  config.text_config.num_attention_heads = 8
  config.text_config.hidden_size = 512
  config.text_config.intermediate_size = 64
  config.text_config.pad_token_id = 127
  config.text_config.use_cache = False
  config.flash_attention = False

  torchprime_config = OmegaConf.create(config.to_dict())
  torchprime_config.vision_config = None
  del torchprime_config.text_config.rope_scaling.original_max_position_embeddings
  del torchprime_config.text_config.rope_scaling.rope_type
  torchprime_config.text_config.attention_kernel = None

  # Place model on CPU device first
  with torch.device("cpu"):
    hf_text_model = HfLlama4ForCausalLM(config.text_config)
    hf_text_model.init_weights()
    text_model = Llama4TextForCausalLM(torchprime_config.text_config)
    text_model.load_state_dict(hf_text_model.state_dict())
  return LlamaFixture(vocab_size, hf_text_model, text_model, torchprime_config)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
def test_forward_our_model_against_hf_model(fixture, transform):
  # Arrange
  fixture = fixture()
  device = torch.device("xla")
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
      atol=1e-5,
      rtol=1e-9,
      msg="logits are not equal",
    )
    torch.testing.assert_close(
      hf_output.loss, llama_xla_loss, atol=1e-6, rtol=1e-9, msg="loss is not equal"
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
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

  device = torch.device("xla")
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


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
)
def test_rotarty_embedding_our_model_against_hf_model_native(fixture):
  fixture = fixture()
  device = torch.device("xla")
  model_xla = copy.deepcopy(fixture.model).to(device)
  torch_xla.sync()
  hf_model = fixture.hf_model
  input_sizes = [8, 128, 256]
  for seq_len in input_sizes:
    batch_size = 2
    in_embedding = torch.randn(
      (batch_size, seq_len // batch_size, fixture.model.config.hidden_size)
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hf_position_embedings = hf_model.model.rotary_emb(in_embedding, position_ids)
    num_attention_heads = fixture.model_config.text_config.num_attention_heads
    head_dim = fixture.model_config.text_config.head_dim
    xq = torch.rand(batch_size, seq_len, num_attention_heads, head_dim)
    xk = torch.rand(batch_size, seq_len, num_attention_heads, head_dim)

    hf_xq_out, hf_xk_out = hf_modeling_llama4.apply_rotary_emb(
      xq, xk, hf_position_embedings
    )

    position_embedings = model_xla.model.rotary_emb(
      in_embedding.to(device), position_ids.to(device)
    )
    xq_out, xk_out = modeling_llama4.apply_rotary_emb(
      xq.to(device), xk.to(device), position_embedings
    )
    torch_xla.sync()
    torch.testing.assert_close(
      hf_xq_out,
      xq_out.to("cpu"),
      atol=1e-3,
      rtol=1e-6,
      msg="xq after apply_rotary_emb is not equal",
    )
    torch.testing.assert_close(
      hf_xk_out,
      xk_out.to("cpu"),
      atol=1e-3,
      rtol=1e-6,
      msg="xk after apply_rotary_emb is not equal",
    )
