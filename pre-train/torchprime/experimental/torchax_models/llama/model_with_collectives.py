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
from dataclasses import dataclass

import jax
import torch
import torch.nn.functional as F
from jax.ad_checkpoint import checkpoint_name
from torch import nn
from torchax import interop

all_gather = interop.torch_view(jax.lax.all_gather)
all_reduce_sum = interop.torch_view(jax.lax.psum)


@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: int | None = None
  vocab_size: int = -1
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier: float | None = None
  norm_eps: float = 1e-5
  rope_theta: float = 500000
  use_scaled_rope: bool = False

  max_batch_size: int = 32
  max_seq_len: int = 8192

  # vision model params
  vision_chunk_size: int = -1  # image resolution for image models
  vision_max_num_chunks: int = 4
  vision_num_cross_attention_layers: int = -1

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if hasattr(self, k):
        setattr(self, k, v)

    if self.n_kv_heads is None:
      self.n_kv_heads = self.n_heads
    assert self.n_kv_heads <= self.n_heads
    assert self.n_heads % self.n_kv_heads == 0
    assert self.dim % self.n_heads == 0


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
    self.n_local_heads = args.n_heads // args.tp_size
    self.n_local_kv_heads = self.n_kv_heads // args.tp_size
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

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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
    out = all_reduce_sum(out, axis_name="tp")
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
    output = all_reduce_sum(output, axis_name="tp")
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
    x = interop.call_jax(checkpoint_name, x, "decoder_layer_input")
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out


#   "layers.*.attention.wo.weight" : ('fsdp', 'tp'), #  torch.int8 (4096, 4096)
#   "layers.*.attention.wq.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
#   "layers.*.attention.wk.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
#   "layers.*.attention.wv.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
#   "layers.*.feed_forward.w1.weight" : ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
#   "layers.*.feed_forward.w2.weight" : ('fsdp', 'tp'), #  torch.float32 (4096, 11008)
#   "layers.*.feed_forward.w3.weight": ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
#   "layers.*.attention_norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
def _fsdp_axis(name):
  if any(key in name for key in ("wq", "wk", "wv", "w1", "w3")):
    return 1
  else:
    return 0


class ScanLayer(nn.Module):
  # requirement: submodule's return value must be the same type/shape as input

  def __init__(self, submodule: nn.Module, num_layers: int, unroll_layers: int):
    super().__init__()
    self.m = submodule
    self.num_layers = num_layers
    self.unroll_layers = unroll_layers
    one_block_statedict = self.m.state_dict()
    self.layer_weights_keys = list(one_block_statedict.keys())
    stacked_weights = self._stack_layer_weights(one_block_statedict, num_layers)
    # register those as parameters on this module

    self.params = nn.ParameterDict(
      {self._param_name_new(k): nn.Parameter(v) for k, v in stacked_weights.items()}
    )

    # self._eval_one_layer = eval_one_layer

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
    rest = args[1:]

    def eval_one_layer(h, weight):
      # unpack args
      def gather_and_call(h, new_weights):
        gathered = {}
        for k, val in new_weights.items():
          gathered[k] = all_gather(
            val, axis=_fsdp_axis(k), axis_name="fsdp", tiled=True
          )
        newh = torch.func.functional_call(self.m, gathered, (h, *rest))
        return newh

      for i in range(self.unroll_layers):
        new_weights = {}
        for k, v in weight.items():
          new_weights[k] = v[i]
        h = gather_and_call(h, new_weights)
      # next layer's input; and residual to be added to list
      return h, torch.ones(1)

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
    _eval_one_layer = interop.call_jax(
      jax.checkpoint,
      eval_one_layer,
      policy=policy,
    )
    reshaped_weights = {}
    for k, v in weights.items():
      shape = v.shape
      reshaped_weights[k] = v.reshape(
        shape[0] // self.unroll_layers, self.unroll_layers, *shape[1:]
      )
    h, _ = scan(
      _eval_one_layer,
      args[0],
      reshaped_weights,
    )
    return h


class Transformer(nn.Module):
  def __init__(self, params: ModelArgs, unroll_layers):
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
    self.layers = ScanLayer(
      TransformerBlock(0, params),
      params.n_layers,
      unroll_layers=unroll_layers,
    )

    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(
      params.dim,
      params.vocab_size,
      bias=False,
    )

  def forward(self, tokens: torch.Tensor, start_pos: int, freqs_cis, mask):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    h = all_gather(h, axis_name="tp", tiled=True, axis=2)
    h = self.layers(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h)
    output = all_gather(output, axis_name="tp", tiled=True, axis=2)
    return output
