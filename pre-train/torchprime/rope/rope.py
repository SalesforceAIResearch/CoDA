"""
Rotary Positional Embeddings (RoPE) implementation.
Reference: https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/docs/10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md
"""

import math
from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class RopeScaling:
  """
  RoPE scaling parameters. The defaults are what was selected in Llama 3.1.
  """

  factor: float = 8.0
  low_freq_factor: float = 1.0
  high_freq_factor: float = 4.0
  original_context_len: int = 8192


def default_rope_frequencies(
  head_dim: int,
  theta: float = 10000.0,
) -> torch.Tensor:
  """
  Computes the original RoPE frequencies in e.g. Llama 2.
  Args:
      head_dim: the size of a single attention head.
      theta: a hyperparameter controlling how fast the embeddings rotate.
  Returns:
      The frequencies for the RoPE embeddings.
  """
  return 1.0 / (
    theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
  )


def llama3_rope_frequencies(
  head_dim: int,
  theta: float = 10000.0,
  scaling: RopeScaling | None = None,
) -> torch.Tensor:
  """
  Computes Llama 3 and 3.1 RoPE frequencies. In Llama 3.1, RoPE frequencies
  may be scaled and interpolated as we move beyond the original context length.
  """
  freqs = default_rope_frequencies(head_dim=head_dim, theta=theta)
  if scaling is None:
    return freqs

  low_freq_wavelen = scaling.original_context_len / scaling.low_freq_factor
  high_freq_wavelen = scaling.original_context_len / scaling.high_freq_factor

  assert low_freq_wavelen >= high_freq_wavelen, (
    f"low_freq_wavelen {low_freq_wavelen} must be greater or equal to "
    f"high_freq_wavelen {high_freq_wavelen}"
  )

  wavelen = 2 * math.pi / freqs
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  freqs = torch.where(wavelen > low_freq_wavelen, freqs / scaling.factor, freqs)
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (scaling.original_context_len / wavelen - scaling.low_freq_factor) / (
    scaling.high_freq_factor - scaling.low_freq_factor
  )
  smoothed_freqs = (1 - smooth_factor) * freqs / scaling.factor + smooth_factor * freqs
  is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
  freqs = torch.where(is_medium_freq, smoothed_freqs, freqs)

  return freqs
