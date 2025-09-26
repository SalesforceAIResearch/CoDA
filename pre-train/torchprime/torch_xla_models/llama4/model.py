# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA 4 model.

Modelled after:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
"""

import torch
import torch_xla.debug.profiler as xp
from omegaconf import DictConfig
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import RopeScaling, llama3_rope_frequencies
from torchprime.torch_xla_models import offloading
from torchprime.torch_xla_models.attention import AttentionModule
from torchprime.torch_xla_models.loss import cross_entropy_loss

logger = logging.get_logger(__name__)


class Llama4TextExperts(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.num_experts = config.num_local_experts
    self.intermediate_size = config.intermediate_size
    self.hidden_size = config.hidden_size
    self.expert_dim = self.intermediate_size
    self.gate_up_proj = nn.Parameter(
      torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
    )
    self.down_proj = nn.Parameter(
      torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
    )
    self.act_fn = ACT2FN[config.hidden_act]

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    This should really not be run on a single machine, as we are reaching compute bound:
    - the inputs are expected to be "sorted" per expert already.
    - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

    Args:
        hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
        selected_experts (torch.Tensor): (batch_size * token_num, top_k)
        routing_weights (torch.Tensor): (batch_size * token_num, top_k)
    Returns:
        torch.Tensor
    """
    hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
    gate_up = torch.bmm(hidden_states, self.gate_up_proj)
    gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
    next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
    next_states = next_states.view(-1, self.hidden_size)
    return next_states


# Phi3MLP
class Llama4TextMLP(nn.Module):
  def __init__(self, config, intermediate_size=None):
    super().__init__()

    if intermediate_size is None:
      intermediate_size = config.intermediate_size

    self.config = config
    self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
    self.activation_fn = ACT2FN[config.hidden_act]

  @xp.trace_me("Llama4TextMLP")
  def forward(self, x):
    down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
    return self.down_proj(down_proj)


class Llama4TextL2Norm(torch.nn.Module):
  def __init__(self, eps: float = 1e-6):
    super().__init__()
    self.eps = eps

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    return self._norm(x.float()).type_as(x)

  def extra_repr(self):
    return f"eps={self.eps}"


class Llama4TextRMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-5):
    """
    Llama4RMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Llama4TextMoe(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.hidden_dim = config.hidden_size
    self.num_experts = config.num_local_experts
    self.experts = Llama4TextExperts(config)
    self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
    self.shared_expert = Llama4TextMLP(config)

  @xp.trace_me("Llama4TextMoe")
  def forward(self, hidden_states):
    batch, seq_len, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, self.hidden_dim)
    router_logits = self.router(hidden_states)
    tokens_per_expert = batch * seq_len

    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
    router_scores = (
      torch.full_like(router_logits, float("-inf"))
      .scatter_(1, router_indices, router_top_value)
      .transpose(0, 1)
    )
    # We do this to make sure we have -inf for non topK tokens before going through the !
    # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
    router_indices = (
      torch.arange(tokens_per_expert, device=hidden_states.device)
      .view(1, -1)
      .expand(router_scores.size(0), -1)
    )
    router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

    router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
    routed_in = torch.gather(
      input=hidden_states,
      dim=0,
      index=router_indices,
    ).to(hidden_states.device)
    # we gather inputs corresponding to each expert based on the router indices
    routed_in = routed_in * router_scores.reshape(-1, 1)
    routed_out = self.experts(routed_in)
    out = self.shared_expert(hidden_states)
    # now that we finished expert computation -> we scatter add because we gathered previously
    # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
    # this scales a lot better if you do EP!
    out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim))
    return out, router_scores


class Llama4TextRotaryEmbedding(nn.Module):
  inv_freq: nn.Buffer

  def __init__(
    self,
    head_dim,
    rope_theta,
    scaling: RopeScaling | None = None,
  ):
    super().__init__()
    inv_freq = llama3_rope_frequencies(head_dim, theta=rope_theta, scaling=scaling)
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  @torch.no_grad()
  def forward(self, x, position_ids):
    inv_freq_expanded = (
      self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    with torch.autocast(device_type=x.device.type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
        1, 2
      )
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def interleave_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """
  Interleaves tensors a and b along the last dimension using stack and reshape.
  Input shape: [B, S, N, D/2]
  Output shape: [B, S, N, D]
  """
  stacked = torch.stack([a, b], dim=-1)  # Shape: [B, S, N, D/2, 2]
  return stacked.reshape(stacked.shape[:-2] + (-1,))  # Shape: [B, S, N, D]


def apply_rotary_emb(
  q: torch.Tensor,
  k: torch.Tensor,
  freqs_cos_sin: tuple[torch.Tensor, torch.Tensor],
  unsqueeze_dim: int = 2,  # Based on q/k shape [B, S, N, D] and cos/sin shape [B, S, D]
) -> tuple[torch.Tensor, torch.Tensor]:
  # Currently PyTorch/XLA doesn't support torch.polar:
  # https://github.com/pytorch/xla/blob/master/codegen/xla_native_functions.yaml
  # We need to reimplement logic in terms of sin/cos math. See
  # https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
  # for background on the logic here.
  # TODO: figure out if we can optimize this or add proper torch.polar support to PyTorch/XLA.
  cos_full, sin_full = freqs_cos_sin
  head_dim = q.size(-1)

  # Unsqueeze cos and sin for broadcasting over the num_heads dimension
  # Input cos_full/sin_full: [B, S, D]
  # After unsqueeze: [B, S, 1, D]
  cos_expanded = cos_full.unsqueeze(unsqueeze_dim)
  sin_expanded = sin_full.unsqueeze(unsqueeze_dim)

  # Extract D/2 unique cosine and sine values
  # Shape: [B, S, 1, D/2]
  cos_freq = cos_expanded[..., : head_dim // 2]
  sin_freq = sin_expanded[..., : head_dim // 2]

  # Split q and k into even and odd indexed components
  # Shape: [B, S, N, D/2]
  q_x = q[..., ::2]  # Even indices
  q_y = q[..., 1::2]  # Odd indices
  k_x = k[..., ::2]
  k_y = k[..., 1::2]

  # Apply rotation
  # Shape: [B, S, N, D/2]
  q_x_out = q_x * cos_freq - q_y * sin_freq
  q_y_out = q_y * cos_freq + q_x * sin_freq
  k_x_out = k_x * cos_freq - k_y * sin_freq
  k_y_out = k_y * cos_freq + k_x * sin_freq

  # Interleave results using interleave_concat
  q_out = interleave_concat(q_x_out, q_y_out)  # Shape: [B, S, N, D]
  k_out = interleave_concat(k_x_out, k_y_out)  # Shape: [B, S, N, D]

  # Ensure output dtype matches input dtype
  return q_out.type_as(q), k_out.type_as(k)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
    batch, num_key_value_heads, n_rep, slen, head_dim
  )
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Llama4TextAttention(nn.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  def __init__(self, config: DictConfig, layer_idx: int | None = None):
    super().__init__()
    self.config = config
    self.attention_block = AttentionModule(config)
    self.layer_idx = layer_idx
    if layer_idx is None:
      logger.warning_once(
        f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        "when creating this class."
      )
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.scaling = config.head_dim**-0.5
    self.attn_scale = config.attn_scale
    self.floor_scale = config.floor_scale
    self.attn_temperature_tuning = config.attn_temperature_tuning
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.use_rope = bool(
      config.no_rope_layers
      and layer_idx < len(config.no_rope_layers)
      and config.no_rope_layers[layer_idx]
    )
    if (config.head_dim * config.num_attention_heads) != config.hidden_size:
      raise ValueError(
        f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {config.hidden_size}"
        f" and `num_attention_heads`: {config.num_attention_heads})."
      )

    self.hidden_size = config.hidden_size
    self.head_dim = self.hidden_size // self.num_attention_heads
    self.max_position_embeddings = config.max_position_embeddings

    self.q_proj = nn.Linear(
      config.hidden_size,
      config.num_attention_heads * self.head_dim,
      bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
      config.hidden_size,
      config.num_key_value_heads * self.head_dim,
      bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
      config.hidden_size,
      config.num_key_value_heads * self.head_dim,
      bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
      config.hidden_size,
      config.hidden_size,
      bias=config.attention_bias,
    )
    if config.use_qk_norm and self.use_rope:
      self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

  @xp.trace_me("Llama4TextAttention")
  def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
  ) -> torch.FloatTensor:
    bsz, q_len, _ = hidden_states.size()

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if self.use_rope:  # the 16E model skips rope for long context on certain layers
      query_states, key_states = apply_rotary_emb(
        query_states, key_states, position_embeddings
      )

    if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
      query_states = self.qk_norm(query_states)
      key_states = self.qk_norm(key_states)

    # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
    if self.attn_temperature_tuning and not self.use_rope:
      seq_len = input_shape[-1]
      cache_position = torch.arange(seq_len, device=query_states.device)
      attn_scales = (
        torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0)
        * self.attn_scale
        + 1.0
      )
      attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand(
        (*input_shape, 1, 1)
      )  # batch size > 1
      query_states = (query_states * attn_scales).to(query_states.dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    attn_output = self.attention_block(
      query_states, key_states, value_states, attention_mask
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output


class Llama4TextDecoderLayer(nn.Module):
  def __init__(self, config, layer_idx):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = Llama4TextAttention(config, layer_idx)
    self.use_chunked_attention = config.attention_chunk_size is not None and bool(
      config.no_rope_layers
      and layer_idx < len(config.no_rope_layers)
      and config.no_rope_layers[layer_idx]
    )
    self.is_moe_layer = layer_idx in config.moe_layers
    if self.is_moe_layer:  # the 128E model interleaves dense / sparse
      self.feed_forward = Llama4TextMoe(config)
    else:
      self.feed_forward = Llama4TextMLP(
        config, intermediate_size=config.intermediate_size_mlp
      )

    self.input_layernorm = Llama4TextRMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = Llama4TextRMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )

    self.layer_idx = layer_idx

  @xp.trace_me("Llama4TextDecoderLayer")
  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    | None = None,  # necessary, but kept here for BC
  ) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
        position_embeddings (*optional*):
            Positional embeddings table.
    """
    # This gives the `hidden_states` tensor a name so that we can layer specify
    # to offload this tensor to host RAM to save memory. This is not a standard
    # torch API because there is no such feature in PyTorch. Instead, the name
    # becomes node metadata during FX graph capture.
    hidden_states = offloading.offload_name(hidden_states, "decoder_input")

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = self.self_attn(
      hidden_states=hidden_states,
      position_embeddings=position_embeddings,
      attention_mask=attention_mask,
      position_ids=position_ids,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.feed_forward(hidden_states)
    if self.is_moe_layer:
      hidden_states, router_logits = hidden_states
    else:
      pass
      # router_logits = None
    hidden_states = residual + hidden_states.view(residual.shape)

    return hidden_states


class Llama4TextModel(nn.Module):
  """
  Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Llama4DecoderLayer`]

  Args:
      config: DictConfig which is a text config for Llama4 model.
  """

  def __init__(self, config: DictConfig):
    super().__init__()
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
    # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
    self.layers = HomogeneousSequential(
      *[
        Llama4TextDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
      ]
    )
    self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    rope_scaling = config.get("rope_scaling", None)
    if rope_scaling is not None:
      rope_scaling = RopeScaling(**rope_scaling)
    self.rotary_emb = Llama4TextRotaryEmbedding(
      head_dim=config.head_dim,
      rope_theta=config.rope_theta,
      scaling=rope_scaling,
    )

  @xp.trace_me("Llama4Model")
  def forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.FloatTensor | None = None,
  ) -> torch.Tensor:
    # convert input ids to embeddings
    inputs_embeds = self.embed_tokens(input_ids)

    seq_length = inputs_embeds.size(1)

    # TODO(https://github.com/pytorch/xla/issues/8783): Pass position_ids as `long()`
    # when `scan` can take non-differentiable inputs.
    position_ids = (
      torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).float()
    )

    # Create a causal attention mask
    causal_mask = torch.triu(
      torch.full((seq_length, seq_length), float("-inf"), device=inputs_embeds.device),
      diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension

    if attention_mask is not None:
      causal_mask = causal_mask * attention_mask[:, None, None, :]

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    freq_cis = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    hidden_states = self.layers(
      inputs_embeds,
      attention_mask=causal_mask,
      position_ids=position_ids,
      position_embeddings=freq_cis,
    )

    hidden_states = self.norm(hidden_states)
    return hidden_states


class Llama4TextForCausalLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = Llama4TextModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.apply(self._init_weights)

  def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()

  @xp.trace_me("Llama4TextForCausalLM")
  def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.FloatTensor | None = None,
  ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
    hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    if labels is None:
      return logits, None
    loss = cross_entropy_loss(logits, labels=labels, vocab_size=self.config.vocab_size)
    return logits, loss
