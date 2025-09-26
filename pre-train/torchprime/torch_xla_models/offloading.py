from collections.abc import Sequence
from typing import Any

import torch
import torch.fx as fx
from functorch.compile import aot_function, make_boxed_func  # type:ignore
from torch.utils._pytree import tree_iter
from torch.utils.checkpoint import CheckpointPolicy
from torch_xla.experimental.stablehlo_custom_call import place_to_device, place_to_host

from .remat_all import remat_all_partition_fn


@torch.library.custom_op("xla::offload_name", mutates_args=())
def offload_name(t: torch.Tensor, name: str) -> torch.Tensor:
  """Given an input tensor, returns a named tensor for offloading selection.

  `offload_name` is an identity function that associates the input
  tensor with `name`. It is primarily useful in conjunction with
  `remat_all_and_offload_these_inputs`, which will rematerialize
  intermediate activations and also offload inputs with the specified
  names to host memory, moving them back during the backward pass.
  """
  if t is None:
    return None
  return t.clone()


@offload_name.register_fake
def _offload_name_fake(t: torch.Tensor, name: str) -> torch.Tensor:
  if t is None:
    return None
  return torch.empty_like(t)


@offload_name.register_autograd
def _offload_name_backward(ctx, grad):
  return grad, None


def remat_all_and_offload_these_inputs(
  joint_module: fx.GraphModule,
  _joint_inputs,
  *,
  num_fwd_outputs,
  names_to_offload: Sequence[str],
  static_lifetime_input_indices=None,
):
  """Partition the graph to rematerialize forward activations and offload
  forward inputs to host.

  `remat_all_and_offload_these_inputs` will rematerialize (recompute) all
  intermediate activations in `joint_module`, and offload inputs with the
  specified names to host memory, moving them back during the backward pass.
  It transforms the joint graph into separate forward and backward graphs.
  """
  input_device = next(iter(tree_iter(_joint_inputs))).device
  names_to_offload_set = set(names_to_offload)

  # Modify the module such that all `offload_name` tensors whose name match
  # `names_to_offload_set` must be saved during the forward pass. Then these
  # nodes will show up in the output of the `fwd` graph as additional
  # residuals. Later, we'll walk over the graph output to identify the nodes.
  for node in joint_module.graph.nodes:
    if (
      tensor_name := _get_tensor_name_if_node_is_offload_name(node)
    ) and tensor_name in names_to_offload_set:
      # This trick is taken from https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/utils/checkpoint.py#L1290
      node.meta["recompute"] = CheckpointPolicy.MUST_SAVE

  fwd, bwd = remat_all_partition_fn(
    joint_module,
    _joint_inputs,
    num_fwd_outputs=num_fwd_outputs,
    static_lifetime_input_indices=static_lifetime_input_indices,
  )
  with torch.device(input_device):
    fw_example_args = _make_arguments(fwd)
    bw_example_args = _make_arguments(bwd)

  fw_name_in_output_indices = _get_offload_name_to_fw_output_indices(fwd)
  bw_name_in_input_names = _get_offload_name_to_bw_input_names(
    bwd, num_fwd_outputs, fw_name_in_output_indices
  )

  def _debug(msg):
    return f"""
In the forward graph:
{fwd.print_readable()}

In the backward graph:
{bwd.print_readable()}

{msg}
  """

  for name in names_to_offload_set:
    if name not in fw_name_in_output_indices:
      raise ValueError(
        _debug(
          f"Did not find {name} in fw_name_in_output_indices: {fw_name_in_output_indices}."
        )
      )
    if name not in bw_name_in_input_names:
      raise ValueError(
        _debug(
          f"Did not find {name} in bw_name_in_input_names: {bw_name_in_input_names}."
        )
      )

  with torch.no_grad():

    def forward(**kwargs):
      out = fwd(**kwargs)
      indices_to_offload = set(
        [fw_name_in_output_indices[name] for name in names_to_offload_set]
      )
      return tuple(
        place_to_host(v) if i in indices_to_offload else v for i, v in enumerate(out)
      )

    def backward(**kwargs):
      arguments_to_move_back = set(
        [bw_name_in_input_names[name] for name in names_to_offload_set]
      )
      kwargs = {
        k: place_to_device(v) if k in arguments_to_move_back else v
        for k, v in kwargs.items()
      }
      return bwd(**kwargs)

    # Use AOTAutograd to retrace forward and backward, thus incorporating
    # the offloading ops.
    graph = [None]

    def get_graph(g, _):
      graph[0] = g
      return make_boxed_func(g)

    _ = aot_function(forward, fw_compiler=get_graph)(**fw_example_args)
    aot_forward = graph[0]

    _ = aot_function(backward, fw_compiler=get_graph)(**bw_example_args)
    aot_backward = graph[0]

    return aot_forward, aot_backward


def _make_arguments(gm: fx.GraphModule):
  """
  Given a graph module, `make_arguments` returns a dictionary of example inputs
  that can be used as keyword arguments to call the graph module.
  """
  example_args = {}
  for node in gm.graph.nodes:
    if node.op != "placeholder":
      continue
    if "tensor_meta" in node.meta:
      tensor_meta = node.meta["tensor_meta"]
      tensor = torch.zeros(
        tensor_meta.shape,
        dtype=tensor_meta.dtype,
        requires_grad=tensor_meta.requires_grad,
      )
      example_args[node.name] = tensor
  return example_args


def _get_offload_name_nodes(gm: torch.fx.GraphModule):
  """Build a dict from `offload_name` function call nodes to their names."""
  named_nodes: dict[Any, str] = {}

  for node in gm.graph.nodes:
    if tensor_name := _get_tensor_name_if_node_is_offload_name(node):
      named_nodes[node] = tensor_name

  return named_nodes


def _get_offload_name_to_fw_output_indices(gm: torch.fx.GraphModule):
  """Given a forward graph `gm`, build a dict from tensor names to their
  position in the forward graph outputs."""

  named_nodes = _get_offload_name_nodes(gm)
  res: dict[str, int] = {}

  for node in gm.graph.nodes:
    if node.op == "output":
      assert len(node.args) <= 1
      if len(node.args) == 0:
        continue
      for i, arg in enumerate(next(iter(node.args))):  # type: ignore
        if arg in named_nodes:
          res[named_nodes[arg]] = i

  return res


def _get_offload_name_to_bw_input_names(
  gm: torch.fx.GraphModule,
  num_fwd_outputs: int,
  offload_name_to_output_indices: dict[str, int],
):
  """Given a backward graph `gm`, build a dict from tensor names to their
  corresponding keyword argument names in the backward graph inputs."""

  res = {}
  placeholder_idx = 0
  bw_input_idx_to_name = {}
  for k, v in offload_name_to_output_indices.items():
    bw_input_idx_to_name[v - num_fwd_outputs] = k

  for node in gm.graph.nodes:
    if node.op == "placeholder":
      if placeholder_idx in bw_input_idx_to_name:
        res[bw_input_idx_to_name[placeholder_idx]] = node.target
      placeholder_idx += 1

  return res


def _get_tensor_name_if_node_is_offload_name(node: torch.fx.Node) -> str | None:
  """If the node is a call to the `offload_name` function, return the `name` string argument
  that was used to call the function. Otherwise, return None.
  """
  if (
    node.op == "call_function"
    and hasattr(node.target, "name")
    and node.target.name() == offload_name._qualname  # type: ignore
  ):
    assert isinstance(node.args[1], str)
    return node.args[1]

  return None
