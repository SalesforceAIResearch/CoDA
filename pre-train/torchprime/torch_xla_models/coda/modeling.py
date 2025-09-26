"""
PyTorch Qwen3 model.
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
Added diffusion training support
"""
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import RopeScaling, default_rope_frequencies
IS_TPU = not torch.cuda.is_available()
if IS_TPU:
  from torchprime.torch_xla_models import offloading
from torchprime.torch_xla_models.coda.attention import AttentionModule

logger = logging.get_logger(__name__)

class Qwen3RMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-6):
    """
    Qwen3RMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen3MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]

  def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
  """Applies Rotary Position Embedding to the query and key tensors.

  Args:
    q (`torch.Tensor`): The query tensor.
    k (`torch.Tensor`): The key tensor.
    cos (`torch.Tensor`): The cosine part of the rotary embedding.
    sin (`torch.Tensor`): The sine part of the rotary embedding.
    position_ids (`torch.Tensor`, *optional*):
      Deprecated and unused.
    unsqueeze_dim (`int`, *optional*, defaults to 1):
      The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
      sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
      that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
      k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
      cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
      the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
  Returns:
    `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
  module: nn.Module,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
  scaling: float,
  dropout: float = 0.0,
  **kwargs,
):
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)

  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
  attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()

  return attn_output, attn_weights

class Qwen3Attention(nn.Module):
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
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = getattr(config, "attention_dropout", 0.0)
    # weiran: diffullama
    self.is_causal = False
    
    self.q_proj = nn.Linear(
      self.hidden_size, 
      self.num_heads * self.head_dim, 
      bias=getattr(config, "attention_bias", False)
    )
    self.k_proj = nn.Linear(
      self.hidden_size, 
      self.num_key_value_heads * self.head_dim, 
      bias=getattr(config, "attention_bias", False)
    )
    self.v_proj = nn.Linear(
      self.hidden_size, 
      self.num_key_value_heads * self.head_dim, 
      bias=getattr(config, "attention_bias", False)
    )
    self.o_proj = nn.Linear(
      self.num_heads * self.head_dim, self.hidden_size, bias=getattr(config, "attention_bias", False)
    )
    self.q_norm = Qwen3RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
    self.k_norm = Qwen3RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
    
    # Handle sliding window - check if layer_types exists and if this layer should use sliding attention
    if not config.use_sliding_window:
      self.sliding_window = None
    else:
      raise NotImplementedError("Sliding window is not implemented for Qwen3")

  def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
  ) -> torch.FloatTensor:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Apply q_norm and k_norm to the head dimension
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    # Apply normalization
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    # Transpose to get the right shape for attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output = self.attention_block(
      query_states, key_states, value_states, attention_mask, segment_ids=segment_ids
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output


class Qwen3RotaryEmbedding(nn.Module):
  inv_freq: nn.Buffer

  def __init__(
    self,
    head_dim,
    rope_theta,
    scaling: RopeScaling | None = None,
  ):
    super().__init__()
    if scaling is None:
      inv_freq = default_rope_frequencies(head_dim, theta=rope_theta)
    else:
      raise NotImplementedError("Scaling is not implemented for Qwen3")
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  @torch.no_grad()
  def forward(self, x, position_ids):
    # x: [bs, num_attention_heads, seq_len, head_size]
    inv_freq_expanded = (
      self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = x.device.type
    device_type = (
      device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
        1, 2
      )
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3DecoderLayer(nn.Module):
  def __init__(self, config: DictConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.layer_idx = layer_idx

    self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

    self.mlp = Qwen3MLP(config)
    self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
    self.post_attention_layernorm = Qwen3RMSNorm(
      config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor]| None = None,  # necessary, but kept here for BC
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
  ) -> torch.Tensor:
    """
    Args:
      hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
      attention_mask (`torch.FloatTensor`, *optional*):
        attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
        query_sequence_length, key_sequence_length)` if default attention is used.
    """    
    # This gives the `hidden_states` tensor a name so that we can layer specify
    # to offload this tensor to host RAM to save memory. This is not a standard
    # torch API because there is no such feature in PyTorch. Instead, the name
    # becomes node metadata during FX graph capture.
    if IS_TPU:
      hidden_states = offloading.offload_name(hidden_states, "decoder_input")

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      position_embeddings=position_embeddings,
      segment_ids=segment_ids,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
      
    return hidden_states


class Qwen3Model(nn.Module):
  """
  Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

  Args:
    config: DictConfig
  """

  def __init__(self, config: DictConfig):
    super().__init__()
    self.vocab_size = config.vocab_size
    if "pad_token_id" not in config:
      self.padding_idx = None
    else:
      self.padding_idx = config.pad_token_id
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
    # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
    # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
    self.layers = HomogeneousSequential(
      *[
        Qwen3DecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
      ]
    )
    self.norm = Qwen3RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

    rope_scaling = config.get("rope_scaling", None)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.rope_theta = getattr(config, "rope_theta", 10000.0)
    if rope_scaling is not None:
      rope_scaling = RopeScaling(**rope_scaling)
    self.rotary_emb = Qwen3RotaryEmbedding(
      head_dim=head_dim, rope_theta=self.rope_theta, scaling=rope_scaling
    )

  def forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.FloatTensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
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
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    hidden_states = self.layers(
      hidden_states,
      attention_mask=causal_mask,
      position_ids=position_ids,
      position_embeddings=position_embeddings,
      segment_ids=segment_ids,
    )

    hidden_states = self.norm(hidden_states)
    
    return hidden_states


class Qwen3ForCausalLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = Qwen3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.mask_token_id = config.mask_token_id
    self.eos_token_id = config.eos_token_id

    # Initialize weights and apply final processing
    self.apply(self._init_weights)

  def _init_weights(self, module):
    std = getattr(self.config, "initializer_range", 0.02)
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()

  def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.FloatTensor | None = None,
    src_mask: torch.BoolTensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
    training_mode: str = "pretrain",
    **kwargs,
  ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
    if not self.training:
      # haolin: during inference the masking is done when preprocessing the input, we don't need src_mask and noising
      model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
      hidden_states = model_output
      logits = self.lm_head(hidden_states) # NOTE: we shift logits in generate()
      # logits = logits.float()[..., :-1, :].contiguous() # NOTE: we shift logits in inference_utils at inference time
      return logits, None

    if training_mode == "sft" and src_mask is None:
      raise ValueError("SFT mode requires a non-null src_mask")


    # weiran: diffullama
    sampling_eps = 1e-3
    mask_token_id = self.mask_token_id
    loss_func = nn.CrossEntropyLoss(reduction="none")
    batch_size, seq_len = input_ids.shape     # input_ids: [batch_size, seq_len]
    masking_schedule = kwargs.get("masking_schedule", None)
    
    # Create maskable_mask based on training mode and src_mask
    # For SFT: src_mask is provided, maskable_mask = ~src_mask
    # For pretrain: src_mask is None, maskable_mask = all True
    if src_mask is not None: # SFT
      maskable_mask = ~src_mask
    else: # pretrain or midtrain
      maskable_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
      # Use masking_config if provided, otherwise fall back to config values
      if masking_schedule is not None:
        prefix_probability = masking_schedule.get("prefix_probability", 0)
        truncate_probability = masking_schedule.get("truncate_probability", 0)
      else:
        prefix_probability = getattr(self.config, "prefix_probability", 0)
        truncate_probability = getattr(self.config, "truncate_probability", 0)
      # Generate random decisions for all batch items
      apply_prefix = torch.rand(batch_size, device=input_ids.device) < prefix_probability
      # Only apply truncation to rows that are NOT prefixed
      apply_truncate = torch.rand(batch_size, device=input_ids.device) < truncate_probability
      apply_truncate = apply_truncate & ~apply_prefix

      if prefix_probability > 0:
        maskable_mask = prefix_input_ids(input_ids, maskable_mask, apply_prefix)
      if truncate_probability > 0:
        input_ids = truncate_input_ids(input_ids, apply_truncate, self.config.pad_token_id)
        maskable_mask = maskable_mask & (input_ids != self.config.pad_token_id) # NOTE: necessary?

    # add noise to input_ids
    sigma = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps
    dsigma = torch.reciprocal(sigma)

    # Sample mask block size
    # Use mask_block_sizes from masking_probs if provided, otherwise fall back to config
    if masking_schedule is not None and "mask_block_sizes" in masking_schedule:
      mask_block_sizes = masking_schedule["mask_block_sizes"]
    else:
      mask_block_sizes = getattr(self.config, "mask_block_sizes", None)
    # Use masking_config if provided, otherwise fall back to config values
    if masking_schedule is not None:
      block_masking_probability = masking_schedule.get("block_masking_probability", 0)
    else:
      block_masking_probability = getattr(self.config, "block_masking_probability", 0)
    if block_masking_probability > 0 and mask_block_sizes is not None and len(mask_block_sizes) > 0:
      mask_block_size = mask_block_sizes[torch.randint(0, len(mask_block_sizes), (1,)).item()]
    else:
      mask_block_size = 1

    noisy_input_ids = transition(
      input_ids,
      sigma[:, None],
      maskable_mask=maskable_mask,
      mask_token_id=mask_token_id,
      mask_block_size=mask_block_size
    )
    loss_mask = noisy_input_ids == mask_token_id

    hidden_states = self.model(input_ids=noisy_input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
    # hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # logits: [bs, seq_len, vocab_size]
    # Shifted logits and labels
    # logits: [bs, seq_len-1, vocab_size]
    logits = logits[..., :-1, :].contiguous()
    # weiran: if the shifted token is not masked in the original input, the loss is 0
    # loss_mask: [bs, seq_len-1]
    loss_mask = loss_mask[..., 1:].contiguous()
    target_ids = input_ids[..., 1:].contiguous()
    # loss: [bs, seq_len-1]
    loss = loss_func(
      logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
    ).reshape(target_ids.shape[0],-1)
    loss = loss.masked_fill(~loss_mask, 0)
    # weiran: divide by the number of tokens in the sequence instead of the number of masked tokens
    # justification is dsigma already accounts for the number of masked tokens
    # this is a hack to get something like per token loss
    # https://github.com/ML-GSAI/SMDM/blob/main/pretrain/train_mdm_rl.py#L281-L283
    loss = (dsigma[:, None] * loss).sum() / (input_ids.shape[0] * input_ids.shape[1])
    return logits, loss


def transition(
  x_0, sigma, maskable_mask, mask_token_id, mask_block_size: int = 1
):
  """Apply masking to input tokens. If mask_block_size > 1, use block masking for all rows."""

  if mask_block_size == 1:
    # Original behavior
    # weiran: diffullama
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < sigma) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t

  # Block masking for entire batch
  return block_masking(x_0, sigma, maskable_mask, mask_token_id, mask_block_size)


def block_masking(x_0, sigma, maskable_mask, mask_token_id, mask_block_size):
    """
    XLA-compatible block masking applied uniformly to all rows in the batch.
    Uses efficient tensor operations to avoid dynamic loops.
    """
    batch_size, seq_len = x_0.shape

    if seq_len < mask_block_size:
        return x_0

    # Calculate number of possible block positions
    num_windows = seq_len - mask_block_size + 1
    
    # Create all possible block positions: [num_windows, mask_block_size]
    window_starts = torch.arange(num_windows, device=x_0.device)
    block_offsets = torch.arange(mask_block_size, device=x_0.device)
    all_positions = window_starts.unsqueeze(1) + block_offsets.unsqueeze(0)
    
    # Check which blocks are fully maskable: [batch_size, num_windows]
    maskable_blocks = maskable_mask.unsqueeze(1).expand(-1, num_windows, -1).gather(
        2, all_positions.unsqueeze(0).expand(batch_size, -1, -1)
    )
    fully_maskable = maskable_blocks.all(dim=2)

    # Determine which blocks should be masked: (batch_size, num_windows)
    effective_sigma = 1 - (1-sigma)**(1/mask_block_size) # NOTE: since we mask with blocks, we need to scale sigma by block size
    should_mask = (torch.rand(batch_size, num_windows, device=x_0.device) < effective_sigma) & fully_maskable

    # Create final mask using simple broadcasting (fully XLA-compatible)
    # For each position in the sequence, check if it's part of any masked block
    position_indices = torch.arange(seq_len, device=x_0.device)  # [seq_len]
    
    # Check for each position if it falls within any masked block
    # position_indices: [seq_len] -> [1, 1, seq_len]
    # all_positions: [num_windows, mask_block_size] -> [1, num_windows, mask_block_size]  
    # should_mask: [batch_size, num_windows] -> [batch_size, num_windows, 1]
    
    position_indices = position_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
    all_positions = all_positions.unsqueeze(0)  # [1, num_windows, mask_block_size]
    should_mask = should_mask.unsqueeze(2)  # [batch_size, num_windows, 1]
    
    # Check if each position matches any of the positions in masked blocks
    # [1, 1, seq_len] == [1, num_windows, mask_block_size] -> [1, num_windows, seq_len]
    position_matches = (position_indices == all_positions.unsqueeze(3)).any(dim=2)  # [1, num_windows, seq_len]
    
    # Apply should_mask to get final positions to mask
    # [batch_size, num_windows, 1] & [1, num_windows, seq_len] -> [batch_size, num_windows, seq_len]
    should_mask_positions = should_mask & position_matches
    
    # Reduce over windows: if any window masks this position, mask it
    final_mask = should_mask_positions.any(dim=1)  # [batch_size, seq_len]

    # Apply the mask
    result = torch.where(final_mask, mask_token_id, x_0)

    return result


def prefix_input_ids(input_ids, maskable_mask, apply_prefix):
  """Apply prefix to input_ids based on configured probability. Return a masksable mask such that the prefix is not masked."""
  batch_size, seq_len = input_ids.shape
  # Generate random prefix lengths for all batch items
  prefix_lengths = torch.randint(1, seq_len, (batch_size,), device=input_ids.device)
  # Create position indices: [1, seq_len]
  position_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) # [1, seq_len]
  # Create prefix mask: True where position < prefix_length
  prefix_mask = position_indices < prefix_lengths.unsqueeze(1)  # [batch_size, seq_len]
  # Apply prefix masking: set to False where we should apply prefix masking
  maskable_mask = maskable_mask & ~(apply_prefix.unsqueeze(1) & prefix_mask)
  return maskable_mask


def truncate_input_ids(input_ids, apply_truncate, pad_token_id):
  """Truncate input_ids at random position and fill with pad token. Return the input_ids with suffix truncated and filled with pad token."""
  batch_size, seq_len = input_ids.shape
  # Generate random truncation positions for all batch items
  truncate_positions = torch.randint(1, seq_len, (batch_size,), device=input_ids.device)
  # Create position indices: [1, seq_len]
  position_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) # [1, seq_len]
  # Create truncate mask: True where position >= truncate_position
  truncate_mask = position_indices >= truncate_positions.unsqueeze(1)  # [batch_size, seq_len]
  # Apply truncation: fill with pad token where we should truncate
  input_ids = torch.where(apply_truncate.unsqueeze(1) & truncate_mask, pad_token_id, input_ids)
  return input_ids



