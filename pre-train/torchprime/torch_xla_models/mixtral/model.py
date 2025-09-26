# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Mixtral model."""

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd as xs
from omegaconf import DictConfig
from torch import nn
from torch.nn import init
from torch_xla.distributed.spmd.xla_sharding import MarkShardingFunction

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.torch_xla_models.attention import AttentionModule
from torchprime.torch_xla_models.loss import cross_entropy_loss
from torchprime.torch_xla_models.topology import get_num_slices


# TODO (https://github.com/AI-Hypercomputer/torchprime/pull/60): Refactor and
# move layers to a separate folder and add unit tests
class MixtralRMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-6):
    """
    MixtralRMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  @xp.trace_me("MixtralRMSNorm")
  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class MixtralRotaryEmbedding(nn.Module):
  def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
    super().__init__()

    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (
      self.base
      ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Build here to make `torch.jit.trace` work.
    self._set_cos_sin_cache(
      seq_len=max_position_embeddings,
      device=self.inv_freq.device,
      dtype=torch.get_default_dtype(),
    )

  def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
      self.inv_freq
    )

    freqs = torch.outer(t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

  @xp.trace_me("MixtralRotaryEmbedding")
  def forward(self, x, seq_len=None):
    # x: [bs, num_attention_heads, seq_len, head_size]
    if seq_len > self.max_seq_len_cached:
      self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

    return (
      self.cos_cached[:seq_len].to(dtype=x.dtype),
      self.sin_cached[:seq_len].to(dtype=x.dtype),
    )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
  """Applies Rotary Position Embedding to the query and key tensors.

  Args:
      q (`torch.Tensor`): The query tensor.
      k (`torch.Tensor`): The key tensor.
      cos (`torch.Tensor`): The cosine part of the rotary embedding.
      sin (`torch.Tensor`): The sine part of the rotary embedding.
      position_ids (`torch.Tensor`):
          The position indices of the tokens corresponding to the query and key tensors. For example, this can be
          used to pass offsetted position ids when working with a KV-cache.
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
  cos = cos[position_ids.long()].unsqueeze(unsqueeze_dim)
  sin = sin[position_ids.long()].unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Mixtral
class MixtralAttention(nn.Module):
  def __init__(self, config: DictConfig, layer_idx: int | None = None):
    super().__init__()
    self.config = config
    self.attention_block = AttentionModule(config)
    self.layer_idx = layer_idx

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.attention_dropout = config.attention_dropout

    if (self.head_dim * self.num_heads) != self.hidden_size:
      raise ValueError(
        f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        f" and `num_heads`: {self.num_heads})."
      )
    self.q_proj = nn.Linear(
      self.hidden_size, self.num_heads * self.head_dim, bias=False
    )
    self.k_proj = nn.Linear(
      self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
    )
    self.v_proj = nn.Linear(
      self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
    )
    self.o_proj = nn.Linear(
      self.num_heads * self.head_dim, self.hidden_size, bias=False
    )

    self.rotary_emb = MixtralRotaryEmbedding(
      self.head_dim,
      max_position_embeddings=self.max_position_embeddings,
      base=self.rope_theta,
    )

  def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return (
      tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
      .transpose(1, 2)
      .contiguous()
    )

  @xp.trace_me("MixtralAttention")
  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
      bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
      bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
      bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
      query_states, key_states, cos, sin, position_ids
    )

    attn_output = self.attention_block(
      query_states, key_states, value_states, attention_mask
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output


class MixtralBlockSparseTop2MLP(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size

    self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
    self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
    self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    self.act_fn = F.silu

  @xp.trace_me("MixtralBlockSparseTop2MLP")
  def forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


class MixtralExpertCapacityTop2MLP(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size
    self.num_experts = config.num_local_experts

    self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
    self.w2 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
    self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))

    self.act_fn = F.silu

    init.kaiming_uniform_(self.w1, a=math.sqrt(5))
    init.kaiming_uniform_(self.w2, a=math.sqrt(5))
    init.kaiming_uniform_(self.w3, a=math.sqrt(5))

  @xp.trace_me("MixtralExpertCapacityTop2MLP")
  def forward(self, dispatch_input):
    mesh = xs.get_global_mesh()
    assert mesh is not None
    layer_w1 = torch.einsum("ebcm,emh->ebch", dispatch_input, self.w1)
    layer_w1 = MarkShardingFunction.apply(
      layer_w1, mesh, ("expert", ("data", "fsdp"), None, None)
    )
    layer_w3 = torch.einsum("ebcm,emh->ebch", dispatch_input, self.w3)
    layer_w3 = MarkShardingFunction.apply(
      layer_w3, mesh, ("expert", ("data", "fsdp"), None, None)
    )
    layer_multiply = self.act_fn(layer_w1) * layer_w3
    intermediate_layer = torch.einsum("ebch,ehm->ebcm", layer_multiply, self.w2)
    intermediate_layer = MarkShardingFunction.apply(
      intermediate_layer, mesh, ("expert", ("data", "fsdp"), None, None)
    )
    return intermediate_layer


class Gmm(torch.autograd.Function):
  @staticmethod
  def _eager_gmm(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
  ) -> torch.Tensor:
    """
    For testing purpose.
    """
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = lhs[start : start + size, :] @ rhs[i, :, :]
      out.append(result)
      start += group_sizes[i]
    return torch.cat(out)

  @staticmethod
  def _eager_gmm_backward(grad_output, lhs, rhs, group_sizes):
    """
    For testing purpose.
    """
    grad_lhs = []
    grad_rhs = []
    start = 0
    for i, size in enumerate(group_sizes):
      grad_lhs.append(
        grad_output[start : start + size, :] @ rhs[i, :, :].transpose(-1, -2)
      )
      grad_rhs.append(
        lhs[start : start + size, :].t() @ grad_output[start : start + size, :]
      )
      start += size
    return torch.cat(grad_lhs), torch.stack(grad_rhs)

  @staticmethod
  @xp.trace_me("gmm_forward")
  def forward(
    ctx,
    hidden_states: torch.Tensor,
    top_ks: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
  ) -> torch.Tensor:
    """
    Integrated with PyTorch/XLA Pallas gmm:

    lhs: [m, hidden_size]
    top_ks: [m, k]
    w1: [num_experts, hidden_size, ffn_dim]
    w2: [num_experts, ffn_dim, hidden_size]
    w3: [num_experts, hidden_size, ffn_dim]
    """
    from torch_xla.experimental.custom_kernel import _histogram, gmm

    device = hidden_states.device
    if device == torch.device("cpu"):
      gmm = Gmm._eager_gmm
    # m is global shape
    m, k, n, num_experts, ffn_dim = (
      hidden_states.shape[0],
      top_ks.shape[1],
      hidden_states.shape[-1],
      w1.shape[0],
      w1.shape[-1],
    )

    # Create a new node to keep the original sharding spec.
    zero = torch.zeros((1,), device=device, dtype=hidden_states.dtype)
    full_w1 = w1 + zero
    full_w2 = w2 + zero
    full_w3 = w3 + zero

    # Enter manual sharding zone
    if xs.get_global_mesh() is not None:
      hidden_states = xs.enable_manual_sharding(
        hidden_states, (("data", "fsdp"), None)
      ).global_tensor
      top_ks = xs.enable_manual_sharding(top_ks, (("data", "fsdp"), None)).global_tensor
      w1 = xs.enable_manual_sharding(full_w1, (None, None, "tensor")).global_tensor
      w2 = xs.enable_manual_sharding(full_w2, (None, "tensor", None)).global_tensor
      w3 = xs.enable_manual_sharding(full_w3, (None, None, "tensor")).global_tensor

    # We want to create one big batch of tokens that has all top-k choices in it.
    # Our tokens will thus be duplicated k-times in the batch. To do this we,
    # first flatten the expert choices list and argsort it. This gives us an array
    # of length B * K. We then create a tiled arange of size B * K and index
    # into the expert choices list. This will give us the set of indices we need
    # to gather from the xs to create this big batch.
    top_flat = top_ks.flatten()
    hidden_states_order = top_flat.argsort()
    hidden_states_reverse_order = hidden_states_order.argsort()
    # Always replicated, so okay to skip manual sharding.
    hidden_states_indices = torch.arange(
      hidden_states.shape[0], device=device
    ).repeat_interleave(k)[hidden_states_order]
    hidden_states_sorted = hidden_states[hidden_states_indices]

    group_sizes = _histogram(top_flat.to(torch.int32), 0, num_experts - 1)
    gmm1 = gmm(hidden_states_sorted, w1, group_sizes, tiling=(512, 1024, 1024))
    gmm3 = gmm(hidden_states_sorted, w3, group_sizes, tiling=(512, 1024, 1024))
    silu = F.silu(gmm1)
    sgmm = silu * gmm3
    gmm2 = gmm(sgmm, w2, group_sizes, tiling=(512, 1024, 1024))
    current_hidden_states = gmm2[hidden_states_reverse_order].reshape(-1, k, n)

    # Exit manual sharding zone
    if xs.get_global_mesh() is not None:
      # For 2D sharding, we need to manually reduce-scatter the final results
      mesh = xs.get_global_mesh()
      if mesh.shape()["tensor"] > 1:
        # Assume tensor axis is the last dim. Otherwise, we will need some complicated transforms.
        assert mesh.get_axis_name_idx("tensor") == len(mesh.mesh_shape) - 1
        device_ids = mesh.get_logical_mesh()
        device_ids = device_ids.reshape(-1, device_ids.shape[-1])
        ctx.device_ids = device_ids

        # Only reduce-scatter along tensor axis.
        current_hidden_states = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM,
          current_hidden_states,
          1.0,
          -1,
          device_ids.shape[-1],
          device_ids.tolist(),
        )

      current_hidden_states = xs.disable_manual_sharding(
        current_hidden_states, (("data", "fsdp"), None, "tensor"), (m, k, n)
      ).global_tensor
      # Checkpoints for backward
      hidden_states_sorted = xs.disable_manual_sharding(
        hidden_states_sorted, (("data", "fsdp"), None), (m * k, n)
      ).global_tensor
      gmm1 = xs.disable_manual_sharding(
        gmm1, (("data", "fsdp"), "tensor"), (m * k, ffn_dim)
      ).global_tensor
      gmm3 = xs.disable_manual_sharding(
        gmm3, (("data", "fsdp"), "tensor"), (m * k, ffn_dim)
      ).global_tensor
      silu = xs.disable_manual_sharding(
        silu, (("data", "fsdp"), "tensor"), (m * k, ffn_dim)
      ).global_tensor
      sgmm = xs.disable_manual_sharding(
        sgmm, (("data", "fsdp"), "tensor"), (m * k, ffn_dim)
      ).global_tensor

    # Save for backward
    ctx.save_for_backward(
      hidden_states_sorted,
      full_w1,
      full_w2,
      full_w3,
      gmm1,
      gmm3,
      silu,
      sgmm,
      hidden_states_order,
      hidden_states_reverse_order,
      group_sizes,
    )
    ctx.k = k

    return current_hidden_states

  @staticmethod
  @xp.trace_me("gmm_backward")
  def backward(ctx, grad_output):
    from torch_xla.experimental.custom_kernel import gmm_backward

    device = grad_output.device
    if device == torch.device("cpu"):
      gmm_backward = Gmm._eager_gmm_backward

    (
      hidden_states_sorted,
      full_w1,
      full_w2,
      full_w3,
      gmm1,
      gmm3,
      silu,
      sgmm,
      hidden_states_order,
      hidden_states_reverse_order,
      group_sizes,
    ) = ctx.saved_tensors
    m, k, n = grad_output.shape[0], ctx.k, hidden_states_sorted.shape[-1]

    # Create a new node to keep the original sharding spec.
    zero = torch.zeros((1,), device=device, dtype=hidden_states_sorted.dtype)
    hidden_states_sorted = hidden_states_sorted + zero
    gmm1 = gmm1 + zero
    gmm3 = gmm3 + zero
    silu = silu + zero
    sgmm = sgmm + zero

    # Enter manual sharding zone
    if xs.get_global_mesh() is not None:
      hidden_states_sorted = xs.enable_manual_sharding(
        hidden_states_sorted, (("data", "fsdp"), None)
      ).global_tensor
      w1 = xs.enable_manual_sharding(full_w1, (None, None, "tensor")).global_tensor
      w2 = xs.enable_manual_sharding(full_w2, (None, "tensor", None)).global_tensor
      w3 = xs.enable_manual_sharding(full_w3, (None, None, "tensor")).global_tensor
      temp_sharding_spec = (("data", "fsdp"), "tensor")
      gmm1 = xs.enable_manual_sharding(gmm1, temp_sharding_spec).global_tensor
      gmm3 = xs.enable_manual_sharding(gmm3, temp_sharding_spec).global_tensor
      silu = xs.enable_manual_sharding(silu, temp_sharding_spec).global_tensor
      sgmm = xs.enable_manual_sharding(sgmm, temp_sharding_spec).global_tensor
      grad_output = xs.enable_manual_sharding(
        grad_output, (("data", "fsdp"), None, None)
      ).global_tensor

    grad_output_sorted = grad_output.reshape(-1, n)[hidden_states_order]
    grad_output, grad_w2 = gmm_backward(
      grad_output_sorted, sgmm, w2, group_sizes, tiling=(512, 1024, 1024)
    )

    grad_gmm1 = gmm3 * grad_output
    grad_gmm1 = torch.ops.aten.silu_backward(grad_gmm1, gmm1)

    grad_gmm1, grad_w1 = gmm_backward(
      grad_gmm1, hidden_states_sorted, w1, group_sizes, tiling=(512, 1024, 1024)
    )

    grad_gmm3 = silu * grad_output
    grad_gmm3, grad_w3 = gmm_backward(
      grad_gmm3, hidden_states_sorted, w3, group_sizes, tiling=(512, 1024, 1024)
    )

    grad_output = grad_gmm1 + grad_gmm3

    grad_output = grad_output[hidden_states_reverse_order]
    grad_output = grad_output.reshape(-1, k, grad_output.shape[-1]).sum(dim=1)
    # Exit manual sharding zone
    if xs.get_global_mesh() is not None:
      if not hasattr(ctx, "device_ids"):
        # Here we do a manual reduce scatter as SPMD will not be able to infer this after the manual sharding zone.
        mesh = xs.get_global_mesh()
        assert mesh is not None
        num_slices = get_num_slices()
        num_devices_per_slice = len(mesh.device_ids) // num_slices
        groups = [
          list(range(i * num_devices_per_slice, (i + 1) * num_devices_per_slice))
          for i in range(num_slices)
        ]
        world_size = len(groups[0])
        grad_w1 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w1, 1 / world_size, -1, world_size, groups
        )
        grad_w2 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w2, 1 / world_size, -2, world_size, groups
        )
        grad_w3 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w3, 1 / world_size, -1, world_size, groups
        )

        grad_output = xs.disable_manual_sharding(
          grad_output, (("data", "fsdp"), None), (m, n)
        ).global_tensor
        # TODO: make the 0s more programmatic.
        # grad_w* sharding isn't affected by multipod.
        grad_w1 = xs.disable_manual_sharding(
          grad_w1, (None, None, "fsdp"), w1.shape
        ).global_tensor
        grad_w2 = xs.disable_manual_sharding(
          grad_w2, (None, "fsdp", None), w2.shape
        ).global_tensor
        grad_w3 = xs.disable_manual_sharding(
          grad_w3, (None, None, "fsdp"), w3.shape
        ).global_tensor
      else:  # 2d sharding
        device_ids = ctx.device_ids

        # Only reduce-scatter along tensor axis.
        grad_output = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_output, 1.0, -1, device_ids.shape[-1], device_ids.tolist()
        )

        # Only reduce-scatter along fsdp axis.
        # TODO: support multi-slice.
        device_ids = device_ids.T
        world_size = device_ids.shape[-1]
        grad_w1 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w1, 1 / world_size, -2, world_size, device_ids.tolist()
        )
        grad_w2 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w2, 1 / world_size, -1, world_size, device_ids.tolist()
        )
        grad_w3 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(
          xm.REDUCE_SUM, grad_w3, 1 / world_size, -2, world_size, device_ids.tolist()
        )

        grad_output = xs.disable_manual_sharding(
          grad_output, ("fsdp", "tensor"), (m, n)
        ).global_tensor
        grad_w1 = xs.disable_manual_sharding(
          grad_w1, (None, "fsdp", "tensor"), full_w1.shape
        ).global_tensor
        grad_w2 = xs.disable_manual_sharding(
          grad_w2, (None, "tensor", "fsdp"), full_w2.shape
        ).global_tensor
        grad_w3 = xs.disable_manual_sharding(
          grad_w3, (None, "fsdp", "tensor"), full_w3.shape
        ).global_tensor
    return grad_output, None, grad_w1, grad_w2, grad_w3


class MixtralGmmTop2MLP(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size
    self.num_experts = config.num_local_experts

    self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
    self.w2 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
    self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))

    self.reset_parameters()

  # The followings are copied from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L49
  def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.w1, a=math.sqrt(5))
    init.kaiming_uniform_(self.w2, a=math.sqrt(5))
    init.kaiming_uniform_(self.w3, a=math.sqrt(5))

  @xp.trace_me("MixtralGmmTop2MLP")
  def forward(self, hidden_states, top_ks):
    return Gmm.apply(hidden_states, top_ks, self.w1, self.w2, self.w3)


class MixtralMoeBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.hidden_dim = config.hidden_size
    self.ffn_dim = config.intermediate_size
    self.num_experts = config.num_local_experts
    self.top_k = config.num_experts_per_tok

    # Possible options are gmm, gmm_stack, dropping and static.
    # Huggingface native only implements static.
    self.moe_implementation = config.moe_implementation

    # gating
    self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    # initialize experts based on moe implementation
    match self.moe_implementation:
      case "gmm":
        self.experts = MixtralGmmTop2MLP(config)
      case "dropping":
        self.experts = MixtralExpertCapacityTop2MLP(config)
        # Only used for dropping implementation
        self.capacity_factor = config.capacity_factor
      case _:
        # gmm_stack and static initialize weights the same way.
        self.experts = nn.ModuleList(
          [MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

  @xp.trace_me("generate_masks")
  def generate_masks(self, top_k_indices, softmax_probs, mesh):
    """Generate masks to dispatch tokens to experts and combine moe activations afterwards.

    Only used for dropping implementation.
    """
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    tokens_per_batch = seq_len * self.top_k
    expert_capacity_per_batch = int(
      (tokens_per_batch / self.num_experts) * self.capacity_factor
    )

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok, num_experts)
    expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).to(torch.int32)
    expert_mask_fused = expert_mask.view(
      batch_size, seq_len * self.top_k, self.num_experts
    )  # (batch, s * top_k, e)
    expert_mask_fused = MarkShardingFunction.apply(
      expert_mask_fused, mesh, (("data", "fsdp", "expert"), None, None)
    )

    expert_token_count_fused = torch.cumsum(
      expert_mask_fused, dim=1
    )  # (b, s * top_k , e)
    expert_token_count = expert_token_count_fused.view(
      batch_size, seq_len, self.top_k, self.num_experts
    )  # (b, s, k, e)
    expert_token_count = MarkShardingFunction.apply(
      expert_token_count, mesh, (("data", "fsdp", "expert"), None, None, None)
    )

    trunc_expert_mask = expert_mask * (
      expert_token_count <= expert_capacity_per_batch
    ).to(torch.int32)  # (b, s, k, e)
    combined_expert_mask = trunc_expert_mask.sum(dim=2)  # (b, s, e)

    # reshape & update weights
    softmax_probs = softmax_probs * combined_expert_mask  # (b, s, e)

    # calculate token position in expert capacity dimension
    expert_token_position_fused = (
      expert_mask_fused * expert_token_count_fused
    )  # (b, s, k, e)
    expert_token_position = expert_token_position_fused.view(
      batch_size, seq_len, self.top_k, self.num_experts
    )  # (b, s, k, e)
    combined_expert_token_position = (
      expert_token_position.sum(dim=2) * combined_expert_mask
    )  # (b, s, e)

    expert_token_position_in_capacity = F.one_hot(
      combined_expert_token_position, num_classes=expert_capacity_per_batch + 1
    ).to(torch.int32)  # (b, s, e, c)

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs.unsqueeze(-1) * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.bool()  # (b, s, e, c)

    return dispatch_mask, combine_mask

  @xp.trace_me("load_balance_loss_func")
  def load_balance_loss(self, top_k_indices, logits):
    """Additional loss to ensure tokens are equally distributed between experts.

    Only used when dropping implementation is used.
    Reference Switch Transformer https://arxiv.org/pdf/2101.03961
    """
    expert_mask = torch.nn.functional.one_hot(
      top_k_indices, num_classes=self.num_experts
    ).to(torch.int32)
    summed_expert_mask = torch.sum(expert_mask, dim=2)
    # Get fraction of tokens dispatched to each expert
    density = torch.mean(summed_expert_mask.float(), dim=1)  # Convert to float for mean
    # get fraction of probability allocated to each expert
    density_prob = torch.mean(logits, dim=1)
    loss = torch.mean(density * density_prob) * (self.num_experts**2)
    return loss

  @xp.trace_me("MixtralMoeBlock")
  def forward(
    self, hidden_states: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    expert_weights = F.softmax(router_logits, dim=1, dtype=hidden_states.dtype)
    routing_weights, selected_experts = torch.topk(expert_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)
    loss = 0.0
    match self.moe_implementation:
      case "static":
        final_hidden_states = torch.zeros(
          (batch_size * sequence_length, hidden_dim),
          dtype=hidden_states.dtype,
          device=hidden_states.device,
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
          expert_layer = self.experts[expert_idx]
          routing_weights_idx = routing_weights.masked_fill(
            selected_experts != expert_idx, 0.0
          ).sum(dim=-1, keepdim=True)
          current_hidden_states = (
            expert_layer(hidden_states) * routing_weights_idx
          )  # We can't mask the input as there is non-linearities in the expert layer.
          final_hidden_states += current_hidden_states.to(hidden_states.dtype)
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
      case "dropping":
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        mesh = xs.get_global_mesh()
        assert mesh is not None
        selected_experts = selected_experts.view(
          batch_size, sequence_length, self.top_k
        )
        expert_weights = expert_weights.view(
          batch_size, sequence_length, self.num_experts
        )
        dispatch_mask, combine_mask = self.generate_masks(
          selected_experts, expert_weights, mesh
        )
        mask_axes = (("data", "fsdp", "expert"), None, None, None)
        dispatch_mask = MarkShardingFunction.apply(dispatch_mask, mesh, mask_axes)
        combine_mask = MarkShardingFunction.apply(combine_mask, mesh, mask_axes)
        loss = self.load_balance_loss(selected_experts, expert_weights)
        hidden_states = MarkShardingFunction.apply(
          hidden_states, mesh, (("data", "fsdp", "expert"), None, None)
        )
        with xp.Trace("bsm,bsec->ebcm"):
          dispatch = torch.einsum("bsm,bsec->ebcm", hidden_states, dispatch_mask)
        dispatch = MarkShardingFunction.apply(
          dispatch, mesh, ("expert", ("data", "fsdp"), None, None)
        )
        expert_layer = self.experts(dispatch)
        with xp.Trace("ebcm,bsec -> bsm"):
          final_hidden_states = torch.einsum(
            "ebcm,bsec -> bsm", expert_layer, combine_mask
          )
        final_hidden_states = MarkShardingFunction.apply(
          final_hidden_states, mesh, (("data", "fsdp", "expert"), None, None)
        )
      case "gmm_stack":
        w1 = torch.stack([expert.w1.weight.t() for expert in self.experts])
        w2 = torch.stack([expert.w2.weight.t() for expert in self.experts])
        w3 = torch.stack([expert.w3.weight.t() for expert in self.experts])
        final_hidden_states = Gmm.apply(hidden_states, selected_experts, w1, w2, w3)
        final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(
          dim=1
        )
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
      case "gmm":
        final_hidden_states = self.experts(hidden_states, selected_experts)
        final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(
          dim=1
        )
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
      case _:
        raise NotImplementedError(
          f"Unsupported moe implementation {self.moe_implementation}"
        )
    return final_hidden_states, router_logits, torch.tensor(loss)


class MixtralDecoderLayer(nn.Module):
  def __init__(self, config: DictConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size

    self.self_attn = MixtralAttention(config, layer_idx)

    self.block_sparse_moe = MixtralMoeBlock(config)
    self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = MixtralRMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )

  @xp.trace_me("MixtralDecoderLayer")
  def forward(
    self,
    hidden_states: torch.Tensor,
    cumulative_loss: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, router_logits, loss = self.block_sparse_moe(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states, cumulative_loss + loss)
    return outputs


# Copied from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->MIXTRAL,Mistral->Mixtral
class MixtralModel(nn.Module):
  """
  Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

  Args:
      config: DictConfig
  """

  def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
    # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
    self.layers = HomogeneousSequential(
      *[
        MixtralDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
      ]
    )
    self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

  @xp.trace_me("MixtralModel")
  def forward(
    self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None
  ) -> tuple:
    batch_size, seq_length = input_ids.shape

    device = input_ids.device
    # TODO(https://github.com/pytorch/xla/issues/8783): Pass position_ids as `long()`
    # when `scan` can take non-differentiable inputs.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).float()
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    inputs_embeds = self.embed_tokens(input_ids)

    causal_mask = torch.triu(
      torch.full((seq_length, seq_length), float("-inf"), device=inputs_embeds.device),
      diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension

    if attention_mask is not None:
      causal_mask = causal_mask * attention_mask[:, None, None, :]

    hidden_states = inputs_embeds

    load_balance_loss = torch.tensor(0.0, device=device)
    hidden_states, load_balance_loss = self.layers(
      hidden_states,
      load_balance_loss,
      attention_mask=causal_mask,
      position_ids=position_ids,
    )

    load_balance_loss = load_balance_loss / len(self.layers)

    hidden_states = self.norm(hidden_states)
    return (hidden_states, load_balance_loss)


class MixtralForCausalLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = MixtralModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.router_aux_loss_coef
    self.num_experts = config.num_local_experts
    self.num_experts_per_tok = config.num_experts_per_tok
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

  @xp.trace_me("MixtralForCausalLM")
  def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
  ) -> tuple:
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    if labels is None:
      return logits, None
    loss = (
      cross_entropy_loss(logits, labels=labels, vocab_size=self.config.vocab_size)
      + self.router_aux_loss_coef * outputs[-1]
    )
    return logits, loss
