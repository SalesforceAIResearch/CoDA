import math

import pytest
import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from torchprime.rope import rope

LLAMA3_SCALING = rope.RopeScaling(
  factor=8,
  low_freq_factor=1,
  high_freq_factor=4,
  original_context_len=8192,
)


@pytest.mark.parametrize(
  "hidden_size, num_attention_heads, theta",
  [(4096, 32, 500000.0), (16384, 128, 500000.0), (65536, 128, 500000.0)],
)
class TestRope:
  def test_default_rope(self, hidden_size, num_attention_heads, theta):
    head_dim = hidden_size // num_attention_heads
    ours = rope.default_rope_frequencies(head_dim=head_dim, theta=theta)

    hf_rope_fn = ROPE_INIT_FUNCTIONS["default"]
    hf, scale = hf_rope_fn(
      PretrainedConfig.from_dict(
        {
          "hidden_size": hidden_size,
          "num_attention_heads": num_attention_heads,
          "rope_theta": theta,
        }
      )
    )

    assert scale == 1
    torch.testing.assert_close(ours, hf)

  def test_llama3_rope_against_hf(self, hidden_size, num_attention_heads, theta):
    head_dim = hidden_size // num_attention_heads
    ours = rope.llama3_rope_frequencies(
      head_dim=head_dim,
      theta=theta,
      scaling=LLAMA3_SCALING,
    )

    hf_rope_fn = ROPE_INIT_FUNCTIONS["llama3"]
    hf, scale = hf_rope_fn(
      PretrainedConfig.from_dict(
        {
          "hidden_size": hidden_size,
          "num_attention_heads": num_attention_heads,
          "rope_theta": theta,
          "rope_scaling": {
            "factor": 8,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 8192,
          },
        }
      ),
      device="cpu",
    )

    assert scale == 1
    torch.testing.assert_close(ours, hf)

  def test_llama3_rope_against_reference(self, hidden_size, num_attention_heads, theta):
    head_dim = hidden_size // num_attention_heads
    ours = rope.llama3_rope_frequencies(
      head_dim=head_dim,
      theta=theta,
      scaling=LLAMA3_SCALING,
    )
    reference = _llama3_reference_apply_scaling(
      rope.default_rope_frequencies(head_dim=head_dim, theta=theta)
    )
    torch.testing.assert_close(ours, reference)


def _llama3_reference_apply_scaling(freqs: torch.Tensor):
  """
  Reference from https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/train_llama3.py#L80
  """
  # Values obtained from grid search
  scale_factor = 8
  low_freq_factor = 1
  high_freq_factor = 4
  old_context_len = 8192  # original llama3 length

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor
  new_freqs = []
  for freq in freqs:
    wavelen = 2 * math.pi / freq
    if wavelen < high_freq_wavelen:
      new_freqs.append(freq)
    elif wavelen > low_freq_wavelen:
      new_freqs.append(freq / scale_factor)
    else:
      assert low_freq_wavelen != high_freq_wavelen
      smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
      )
      new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
  return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
