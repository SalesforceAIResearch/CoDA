import math
from typing import Any

import torch
from torch import nn

# Detect if we're on TPU (no CUDA available) vs GPU
IS_TPU = not torch.cuda.is_available()

if IS_TPU:
  # TPU environment - use original torch_xla imports
  import torch_xla.debug.profiler as xp
  import torch_xla.distributed.spmd as xs
  from torch_xla.experimental.custom_kernel import FlashAttention, flash_attention
  from torch_xla.experimental.splash_attention import (
    SplashAttentionConfig,
    splash_attention,
  )
else:
  # GPU environment - use PyTorch's native SDPA flash attention
  from torch.nn.functional import scaled_dot_product_attention
  from torch.nn.attention import SDPBackend, sdpa_kernel


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


class AttentionModule(nn.Module):
  def __init__(self, config, kernel_config: dict[str, Any] | None = None):
    super().__init__()
    self.config = config
    self.kernel_config = kernel_config

  def _forward_tpu(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
  ):
    """Original TPU/XLA implementation"""

    if self.config.attention_kernel != "splash_attention":
      num_key_value_groups = (
        self.config.num_attention_heads // self.config.num_key_value_heads
      )
      key_states = repeat_kv(key_states, num_key_value_groups)
      value_states = repeat_kv(value_states, num_key_value_groups)

    bsz, num_heads, q_len, head_dim = query_states.size()
    # TODO: q, k dim unintentionally changed after the apply_rotary_pos_emb. Use
    # v's dim temporarily to bypass shape assertion failure. Remove the
    # following line after resolving
    # https://github.com/AI-Hypercomputer/torchprime/issues/195.
    head_dim = value_states.shape[-1]

    kv_seq_len = key_states.shape[-2]

    if segment_ids is not None:
      segment_ids = segment_ids.int() # NOTE: haolin: bypass the scan limitation with integer tensors

    # Non FA path doesn't deal with 2D sharding.
    self.partition_spec = None
    segment_ids_partition_spec = None
    if xs.get_global_mesh() is not None:
      self.partition_spec = (("data", "fsdp"), "tensor", None, None)
      segment_ids_partition_spec = (("data", "fsdp"), None)

    match self.config.attention_kernel:
      case "splash_attention":
        raise NotImplementedError("Splash Attention is not supported yet")
        # Integrated with PyTorch/XLA Pallas Splash Attention:
        assert xs.get_global_mesh() is not None, (
          "Global mesh is required for Splash Attention"
        )
        sa_config = SplashAttentionConfig(
          mesh=str(xs.get_global_mesh()),
          qkv_partition_spec=self.partition_spec,
          segment_ids_partition_spec=segment_ids_partition_spec,
        )
        if self.kernel_config is not None:
          for key, value in self.kernel_config.items():
            if hasattr(sa_config, key):
              setattr(sa_config, key, value)
        query_states /= math.sqrt(head_dim)
        attn_output = splash_attention(
          query_states, key_states, value_states, sa_config.to_json()
        )
      case "flash_attention":
        # Integrated with PyTorch/XLA Pallas Flash Attention:
        default_block_sizes = {
          "block_q": 2048,
          "block_k_major": 512,
          "block_k": 512,
          "block_b": 2,
          "block_q_major_dkv": 2048,
          "block_k_major_dkv": 512,
          "block_q_dkv": 2048,
          "block_k_dkv": 512,
          "block_q_dq": 2048,
          "block_k_dq": 256,
          "block_k_major_dq": 512,
        }
        if self.kernel_config is not None:
          default_block_sizes.update(self.kernel_config)
        FlashAttention.DEFAULT_BLOCK_SIZES = default_block_sizes

        query_states /= math.sqrt(head_dim)
        attn_output = flash_attention(
          query_states,
          key_states,
          value_states,
          causal=False, # weiran: causal=False for bi-directional attention
          q_segment_ids=segment_ids,
          kv_segment_ids=segment_ids,
          partition_spec=self.partition_spec,
        )
      case "default" | None:
        # Default attention implementation (no flash attention)
        attn_weights = torch.matmul(
          query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
          raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
          )
        
        # Create document mask from segment_ids if provided
        if segment_ids is not None:
          # segment_ids shape: [batch_size, seq_len]
          # Create document mask where tokens can only attend within the same segment
          # Expand segment_ids for query and key positions
          query_segment_ids = segment_ids.unsqueeze(2)  # [batch_size, seq_len, 1]
          key_segment_ids = segment_ids.unsqueeze(1)    # [batch_size, 1, seq_len]
          
          # Create mask: True where segments match, False otherwise
          document_mask = (query_segment_ids == key_segment_ids)  # [batch_size, seq_len, seq_len]
          
          # Convert to attention mask format (0 for allowed, -inf for masked)
          document_mask = document_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
          document_mask = document_mask.to(attn_weights.dtype)
          document_mask = (1.0 - document_mask) * torch.finfo(attn_weights.dtype).min
          # Apply document mask
          attn_weights = attn_weights + document_mask

        # haolin: disable causal mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #   causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #   attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
          attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
          attn_weights, p=self.config.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
      case _:
        raise NotImplementedError(f"Attention kernel {self.config.attention_kernel} is not supported yet")

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
        f" {attn_output.size()}"
      )
    return attn_output

  def _forward_gpu(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
  ):
    """GPU-optimized PyTorch implementation"""
    if self.config.attention_kernel != "splash_attention":
      num_key_value_groups = (
        self.config.num_attention_heads // self.config.num_key_value_heads
      )
      key_states = repeat_kv(key_states, num_key_value_groups)
      value_states = repeat_kv(value_states, num_key_value_groups)

    bsz, num_heads, q_len, head_dim = query_states.size()
    # TODO: q, k dim unintentionally changed after the apply_rotary_pos_emb. Use
    # v's dim temporarily to bypass shape assertion failure. Remove the
    # following line after resolving
    # https://github.com/AI-Hypercomputer/torchprime/issues/195.
    head_dim = value_states.shape[-1]

    kv_seq_len = key_states.shape[-2]

    # Use SDPA with appropriate backend
    
    match self.config.attention_kernel:
      case "splash_attention":
        raise NotImplementedError("Splash Attention is not supported in GPU environment")
        
      case "flash_attention":
        # Try to use flash attention backend, fallback to default if not available
        try:
          with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = scaled_dot_product_attention(
              query_states,
              key_states,
              value_states,
              dropout_p=self.config.attention_dropout if self.training else 0.0,
              is_causal=False,  # weiran: causal=False for bi-directional attention
            )
        except (RuntimeError, NotImplementedError):
          # Flash attention not available, use default backend
          with sdpa_kernel(SDPBackend.MATH):
            attn_output = scaled_dot_product_attention(
              query_states,
              key_states,
              value_states,
              dropout_p=self.config.attention_dropout if self.training else 0.0,
              is_causal=False,  # weiran: causal=False for bi-directional attention
            )

      case _:
        # Default implementation - use math backend for compatibility
        with sdpa_kernel(SDPBackend.MATH):
          attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,  # weiran: causal=False for bi-directional attention
          )

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
        f" {attn_output.size()}"
      )
    return attn_output

  def forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,  # Should be int32 for TPU compatibility
  ):
    if IS_TPU:
      return self._forward_tpu(query_states, key_states, value_states, attention_mask, segment_ids=segment_ids)
    else:
      return self._forward_gpu(query_states, key_states, value_states, attention_mask, segment_ids=segment_ids)
