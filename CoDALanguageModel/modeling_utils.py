from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn

class HomogeneousSequential(nn.Sequential):
    """
    HomogenousSequential is a sequential container that requires all child modules
    to be of the same type and have matching input/output shapes. In turn, it may be
    compiled with the `scan` higher order operator to save compile time.
    """

    repeated_layer: type
    """The type of the layer being looped over."""

    def __init__(self, *args: nn.Module) -> None:
        super().__init__(*args)
        types = set(type(module) for module in args)
        assert len(types) == 1, f"All modules must be of the same type. Got {types}"
        self.repeated_layer = types.pop()

    def forward(self, *input, **broadcasted_inputs):
        """
        Much like `torch.nn.Sequential`, this takes `input` and forwards it to the
        first module it contains. It then "chains" outputs to inputs sequentially for
        each subsequent module, finally returning the output of the last module.
        Different from `torch.nn.Sequential`, you may specify `broadcasted_inputs` via
        keyword arguments. The same keyword arguments will be passed to every layer
        without changes (i.e. "broadcasted").
        """
        for module in self:
            input = module(*splat(input), **broadcasted_inputs)
        return input


def splat(input):
    if not isinstance(input, list | tuple):
        input = (input,)
    return input


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



def transition(x_0, sigma, maskable_mask, mask_token_id, mask_block_size: int = 1):
    """Apply masking to input tokens. If mask_block_size > 1, use block masking for all rows."""

    if mask_block_size == 1:
        # Original behavior
        # weiran: diffullama
        move_indices = (
            torch.rand(*x_0.shape, device=x_0.device) < sigma
        ) & maskable_mask
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
    maskable_blocks = (
        maskable_mask.unsqueeze(1)
        .expand(-1, num_windows, -1)
        .gather(2, all_positions.unsqueeze(0).expand(batch_size, -1, -1))
    )
    fully_maskable = maskable_blocks.all(dim=2)

    # Determine which blocks should be masked: (batch_size, num_windows)
    effective_sigma = 1 - (1 - sigma) ** (
        1 / mask_block_size
    )  # NOTE: since we mask with blocks, we need to scale sigma by block size
    should_mask = (
        torch.rand(batch_size, num_windows, device=x_0.device) < effective_sigma
    ) & fully_maskable

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
    position_matches = (position_indices == all_positions.unsqueeze(3)).any(
        dim=2
    )  # [1, num_windows, seq_len]

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
    position_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(
        0
    )  # [1, seq_len]
    # Create prefix mask: True where position < prefix_length
    prefix_mask = position_indices < prefix_lengths.unsqueeze(
        1
    )  # [batch_size, seq_len]
    # Apply prefix masking: set to False where we should apply prefix masking
    maskable_mask = maskable_mask & ~(apply_prefix.unsqueeze(1) & prefix_mask)
    return maskable_mask


def truncate_input_ids(input_ids, apply_truncate, pad_token_id):
    """Truncate input_ids at random position and fill with pad token. Return the input_ids with suffix truncated and filled with pad token."""
    batch_size, seq_len = input_ids.shape
    # Generate random truncation positions for all batch items
    truncate_positions = torch.randint(
        1, seq_len, (batch_size,), device=input_ids.device
    )
    # Create position indices: [1, seq_len]
    position_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(
        0
    )  # [1, seq_len]
    # Create truncate mask: True where position >= truncate_position
    truncate_mask = position_indices >= truncate_positions.unsqueeze(
        1
    )  # [batch_size, seq_len]
    # Apply truncation: fill with pad token where we should truncate
    input_ids = torch.where(
        apply_truncate.unsqueeze(1) & truncate_mask, pad_token_id, input_ids
    )
    return input_ids
