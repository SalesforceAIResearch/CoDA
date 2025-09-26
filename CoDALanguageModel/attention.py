import math
from typing import Any

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from .model_config import CoDAConfig



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
    def __init__(self, config: CoDAConfig, kernel_config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.kernel_config = kernel_config
        self.partition_spec = None

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """GPU-optimized PyTorch implementation"""

        if self.config.attention_kernel != "splash_attention":
            num_key_value_groups = (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.size()
        head_dim = value_states.shape[-1]
        kv_seq_len = key_states.shape[-2]

        # Use SDPA with appropriate backend
        match self.config.attention_kernel:
            case "splash_attention":
                raise NotImplementedError(
                    "Splash Attention is not supported in GPU environment"
                )

            case "flash_attention":
                # Try to use flash attention backend, fallback to default if not available
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn_output = scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        dropout_p=(
                            self.config.attention_dropout if self.training else 0.0
                        ),
                        is_causal=False,  # weiran: causal=False for bi-directional attention
                    )
            case _:
                # Default implementation - use math backend for compatibility
                with sdpa_kernel(SDPBackend.MATH):
                    attn_output = scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        dropout_p=(
                            self.config.attention_dropout if self.training else 0.0
                        ),
                        is_causal=False,  # weiran: causal=False for bi-directional attention
                    )

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )
        return attn_output
