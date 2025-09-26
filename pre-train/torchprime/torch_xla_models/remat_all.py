import torch._functorch.config
import torch.fx
from functorch.compile import min_cut_rematerialization_partition
from torch.utils.checkpoint import CheckpointPolicy


def remat_all_partition_fn(
  joint_module: torch.fx.GraphModule,
  _joint_inputs,
  *,
  num_fwd_outputs,
  static_lifetime_input_indices=None,
):
  """
  remat_all_partition_fn is a graph partition function that closely matches the
  default behavior of `torch.utils.checkpoint`, which is to discard all intermediate
  activations and recompute all of them during the backward pass.
  """
  # Mark anything that does not have a policy as MUST_RECOMPUTE
  for node in joint_module.graph.nodes:
    if _is_call(node) and "recompute" not in node.meta:
      # This trick is taken from https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/utils/checkpoint.py#L1290
      node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

      # min_cut_rematerialization_partition checks the graph ID to handle multiple
      # graphs at once. We only have one graph so this can simply be 0.
      node.meta["ac_graph_id"] = 0

  return min_cut_rematerialization_partition(
    joint_module,
    _joint_inputs,
    num_fwd_outputs=num_fwd_outputs,
    static_lifetime_input_indices=static_lifetime_input_indices,
  )


def _is_call(node: torch.fx.Node):
  # See documentation here: https://pytorch.org/docs/stable/fx.html
  match node.op:
    case "call_function" | "call_method" | "call_module":
      return True
    case _:
      return False
