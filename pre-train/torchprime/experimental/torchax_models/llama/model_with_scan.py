# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py

import math

import jax
import torch
import torch.nn.functional as F
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from torch import nn
from torchax import interop

from .model import ModelArgs

with_sharding_constraint = interop.torch_view(jax.lax.with_sharding_constraint)


# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.

transformer_configs = {
  "8B": {
    "dim": 4096,
    "ffn_dim_multiplier": 1.3,
    "multiple_of": 1024,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 32,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256,
  },
  "70B": {
    "dim": 8192,
    "ffn_dim_multiplier": 1.3,
    "multiple_of": 4096,
    "n_heads": 64,
    "n_kv_heads": 8,
    "n_layers": 80,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256,
  },
  "405B": {
    "dim": 16384,
    "ffn_dim_multiplier": 1.2,
    "multiple_of": 4096,
    "n_heads": 128,
    "n_kv_heads": 16,
    "n_layers": 126,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256,
  },
}


class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


def apply_scaling(freqs: torch.Tensor):
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


def precompute_freqs_cis(
  dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  if use_scaled:
    freqs = apply_scaling(freqs)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
  xq: torch.Tensor,
  xk: torch.Tensor,
  freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.to(torch.bfloat16), xk_out.to(torch.bfloat16)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
  bs, slen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
    x[:, :, :, None, :]
    .expand(bs, slen, n_kv_heads, n_rep, head_dim)
    .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
  )


class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.n_local_heads = args.n_heads
    self.n_local_kv_heads = self.n_kv_heads
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = args.dim // args.n_heads

    self.wq = nn.Linear(
      args.dim,
      args.n_heads * self.head_dim,
      bias=False,
    )
    self.wk = nn.Linear(
      args.dim,
      self.n_kv_heads * self.head_dim,
      bias=False,
    )
    self.wv = nn.Linear(
      args.dim,
      self.n_kv_heads * self.head_dim,
      bias=False,
    )
    self.wo = nn.Linear(
      args.n_heads * self.head_dim,
      args.dim,
      bias=False,
    )

  def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor | None,
  ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    xq = with_sharding_constraint(xq, P("fsdp", None, "tp", None))
    xk = with_sharding_constraint(xk, P("fsdp", None, "tp", None))
    xv = with_sharding_constraint(xv, P("fsdp", None, "tp", None))

    # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xq = interop.call_jax(checkpoint_name, xq, "query_proj")
    xk = interop.call_jax(checkpoint_name, xk, "key_proj")
    xv = interop.call_jax(checkpoint_name, xv, "value_proj")

    keys = xk
    values = xv

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(
      keys, self.n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
    values = repeat_kv(
      values, self.n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

    output = torch.nn.functional.scaled_dot_product_attention(
      xq, keys, values, is_causal=(mask is not None)
    )

    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    out = self.wo(output)
    out = with_sharding_constraint(out, P("fsdp", None, None))
    out = interop.call_jax(checkpoint_name, out, "out_proj")
    return out


class FeedForward(nn.Module):
  def __init__(
    self,
    dim: int,
    hidden_dim: int,
    multiple_of: int,
    ffn_dim_multiplier: float | None,
  ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(
      dim,
      hidden_dim,
      bias=False,
    )
    self.w2 = nn.Linear(
      hidden_dim,
      dim,
      bias=False,
    )
    self.w3 = nn.Linear(
      dim,
      hidden_dim,
      bias=False,
    )

  def forward(self, x):
    wi1 = self.w1(x)
    wi3 = self.w3(x)
    wi1 = interop.call_jax(checkpoint_name, wi1, "mlpwi")
    wi3 = interop.call_jax(checkpoint_name, wi3, "mlpwi")
    output = self.w2(F.silu(wi1) * wi3)
    output = interop.call_jax(checkpoint_name, output, "mlpwo")
    return output


class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim=args.dim,
      hidden_dim=4 * args.dim,
      multiple_of=args.multiple_of,
      ffn_dim_multiplier=args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor | None,
  ):
    x = with_sharding_constraint(x, P("fsdp", None, "tp"))
    x = interop.call_jax(checkpoint_name, x, "decoder_layer_input")
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out


class ScanLayer(nn.Module):
  # requirement: submodule's return value must be the same type/shape as input

  def __init__(self, submodule: nn.Module, num_layers: int):
    super().__init__()
    self.m = submodule
    self.num_layers = num_layers
    one_block_statedict = self.m.state_dict()
    self.layer_weights_keys = list(one_block_statedict.keys())
    stacked_weights = self._stack_layer_weights(one_block_statedict, num_layers)
    # register those as parameters on this module

    self.params = nn.ParameterDict(
      {self._param_name_new(k): nn.Parameter(v) for k, v in stacked_weights.items()}
    )

  def _stack_layer_weights(self, orig_state_dict, num_layers):
    # Create weights such that, for every [n, m] weights
    # becomes [k, n, m] where k is number of layer
    # i.e. stacking layer weights together
    temp = {}
    for k, v in orig_state_dict.items():
      newv = torch.stack([v for _ in range(num_layers)])
      temp[k] = newv
    return temp

  def _param_name_new(self, old):
    return "___".join(old.split("."))

  def _param_name_old(self, new):
    return ".".join(new.split("___"))

  def forward(self, *args, **kwargs):
    assert not kwargs
    weights = {k: self.params[self._param_name_new(k)] for k in self.layer_weights_keys}
    scan = interop.torch_view(jax.lax.scan)
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
      names_which_can_be_saved=[],
      names_which_can_be_offloaded=[
        "decoder_layer_input",
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj",
      ],
      offload_src="device",
      offload_dst="pinned_host",
    )

    def eval_one_layer(args, weight):
      # unpack args
      h, *rest = args
      newh = torch.func.functional_call(self.m, weight, args)
      # next layer's input; and residual to be added to list
      return (newh, *rest), torch.ones(1)

    _eval_one_layer = interop.call_jax(
      jax.checkpoint,
      eval_one_layer,
      policy=policy,
    )
    h, _ = scan(
      _eval_one_layer,
      args,
      weights,
    )
    return h[0]


class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    self.tok_embeddings = nn.Embedding(
      params.vocab_size,
      params.dim,
    )

    # self.layers = torch.nn.ModuleList()
    # for layer_id in range(params.n_layers):
    #     self.layers.append(TransformerBlock(layer_id, params))
    self.layers = ScanLayer(TransformerBlock(0, params), params.n_layers)

    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(
      params.dim,
      params.vocab_size,
      bias=False,
    )

  def forward(self, tokens: torch.Tensor, start_pos: int, freqs_cis, mask):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    h = self.layers(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h)
    return output
