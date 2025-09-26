from typing import Any

import torch.nn as nn

PyTree = Any


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

  def forward(self, *input, **broadcasted_inputs: PyTree):
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
