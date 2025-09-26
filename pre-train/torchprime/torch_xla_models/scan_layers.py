import torch
import torch.nn as nn
from functorch.compile import default_partition
from torch_xla.experimental.scan_layers import scan_layers

from torchprime.layers.sequential import HomogeneousSequential, PyTree, splat


class HomogeneousSequentialScan(HomogeneousSequential):
  def __init__(self, *args, partition_fn=default_partition):
    super().__init__(*args)
    self.partition_fn = partition_fn

  def forward(self, *input, **broadcasted_inputs: PyTree):
    # `self.children()` returns an iterator over the immediate submodules, i.e.
    # the layers we want to scan over. In the `BroadcastArguments` we extend each
    # layer's return value to also output the broadcasted inputs
    # (position IDs in case of LLMs, etc). This plumbs those values across scan
    # iterations so the same values are available to all layers.
    layers = [BroadcastArguments(m) for m in self.children()]
    if len(input) == 1:
      # Handle single argument case: we don't need to call the module with a tuple.
      input = input[0]
    out, _broadcasted_inputs_back = scan_layers(
      layers, (input, broadcasted_inputs), partition_fn=self.partition_fn
    )
    return out


class BroadcastArguments(torch.nn.Module):
  def __init__(self, mod: nn.Module):
    super().__init__()
    self.mod = mod

  def forward(self, orig_input, broadcasted_inputs):
    out = self.mod(*splat(orig_input), **broadcasted_inputs)
    return (out, broadcasted_inputs)


def compile_one_stack(
  mod: HomogeneousSequential, partition_fn=default_partition
) -> HomogeneousSequential:
  # Replace base class with our optimized subclass.
  if isinstance(mod, HomogeneousSequentialScan):
    raise NotImplementedError("Cannot compile HomogeneousSequential twice")
  new_mod = HomogeneousSequentialScan(*mod.children(), partition_fn=partition_fn)
  return new_mod


def compile(
  mod: nn.Module, sequential_to_scan: str, partition_fn=default_partition
) -> nn.Module:
  seq = mod.get_submodule(sequential_to_scan)
  if not isinstance(seq, HomogeneousSequential):
    raise ValueError(f"compile only supports HomogeneousSequential, got {type(seq)}")
  # Replace the submodule
  mod.set_submodule(
    sequential_to_scan, compile_one_stack(seq, partition_fn=partition_fn)
  )
  return mod
