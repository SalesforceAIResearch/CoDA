import re
import logging
from collections.abc import Callable
from copy import copy
from typing import Optional

import torch.nn

logger = logging.getLogger(__name__)

ShardWeightFn = Callable[[torch.Tensor, str], torch.Tensor]
"""
ShardWeightFn optionally transforms a weight tensor based on its name.

Args:

  weight (torch.Tensor): The weight tensor to be transformed.

  name (str): The name of the weight tensor as it appears in the state dict.

Returns:

  torch.Tensor: The transformed weight tensor.

"""


ShardActivationFn = Callable[[torch.nn.Module, str], torch.nn.Module]
"""
ShardActivationFn optionally transforms a module based on its name.

Args:

  module (torch.nn.Module): The module to be transformed.

  name (str): The name of the module as it appears in the state dict.

Returns:

  torch.nn.Module: The transformed module, or the original module if
    no transformation is needed.

"""


def shard_model(
  model: torch.nn.Module,
  shard_weight: ShardWeightFn,
  shard_activation: ShardActivationFn,
) -> torch.nn.Module:
  """
  Transforms `model` by applying `shard_weight` to each weight tensor and applying
  `shard_activation` to each module. Returns the transformed module.
  """
  state_dict = {}
  for name, param in model.state_dict().items():
    state_dict[name] = shard_weight(param, name)
  model.load_state_dict(state_dict, assign=True)
  return wrap_module(model, shard_activation)


ModuleTransform = Callable[[torch.nn.Module, str], torch.nn.Module]
"""
A transform takes a module and its fully qualified name (as a dot-joined string)
and returns the (possibly) transformed module.
"""


def wrap_module(
  mod: torch.nn.Module, transform: ModuleTransform, prefix: tuple[str, ...] = tuple()
) -> torch.nn.Module:
  """
  Recursively transforms the modules by calling `transform` on them.

  You may use this to apply sharding, checkpointing, optimization barriers, etc.

  Start from the leaf modules and work our way up, to handle cases where one
  module is the child of another. The child modules will be transformed first,
  and then the parent module will be transformed, possibly with transformed
  children.
  """
  new_children = {}
  for name, child in mod.named_children():
    new_children[name] = wrap_module(child, transform, prefix + (name,))
  for name, new_child in new_children.items():
    mod.set_submodule(name, new_child)
  return transform(mod, ".".join(prefix))


TAIL_INDEX_REGEX = re.compile(r"\[\d+\]$")


def shard_model_from_config(
  model: torch.nn.Module,
  config: dict,
  shard_output: Callable[[torch.Tensor, tuple[str, ...]], torch.Tensor],
  shard_param: Callable[[torch.Tensor, tuple[str, ...]], torch.Tensor] | None = None,
) -> torch.nn.Module:
  """
  Given a config of pattern to partition spec, shard the model accordingly.

  Example:

    ```python
    config = {
      # Shard the embedding projection
      'model.embed_tokens.weight': ['fsdp', None],
      # Shard the self-attention query projection
      'model.layers.*.self_attn.q_proj.weight': ['fsdp', None],
      # Shard the decoder layer outputs
      'model.layers.*': ['fsdp', None, None],
      # An empty string matches the output of the entire module.
      '': ['fsdp', None, None],

      # Advanced usage: an index at the end means indexing into a particular
      # output of a module, then sharding that. This is useful if your module
      # output is a list or tensor and only one of the element should be sharded.
      'model.layers.*[0]': ['fsdp', None, None],
    }

    model = shard_model_from_config(model, config, xs.mark_sharding)
    ```

  A pattern may have an asterisk, which matches all immediate children whose
  name is an integer. This is useful for sharding all layers in a model.

  If a pattern matches a model parameter, then the parameter will be sharded.
  If a pattern matches a module, then the output of the module will be sharded.
  """

  if shard_param is None:
    shard_param = shard_output

  config, shard_output_fns = _process_tail_index_syntax(config, shard_output)
  seen_params = set()
  seen_modules = set()

  def shard_weight(param, name):
    name = _process_sharding_name(name)
    spec = config.get(name)
    if spec is not None:
      seen_params.add(name)
      return shard_param(param, tuple(spec))
    return param

  def shard_activation(mod, name):
    name = _process_sharding_name(name)
    spec = config.get(name)
    if spec is not None:
      seen_modules.add(name)
      return ShardedModule(mod, shard_output_fns.get(name, shard_output), tuple(spec))
    return mod

  model = shard_model(model, shard_weight, shard_activation)

  want_names = set(config.keys())
  seen_names = seen_params.union(seen_modules)
  diff = "\n".join(want_names - seen_names)
  assert (
    seen_names == want_names
  ), f"""Requested to shard these names: {want_names}, but only sharded these: {seen_names}.

These names were not found in the model:
{diff}
"""

  return model


def shard_torchax_model_from_config(
  model: torch.nn.Module,
  config: dict,
  mesh: "jax.sharding.Mesh",  # type: ignore  # noqa: F821
):
  """
  Given a config of pattern to partition spec, shard a torchax model accordingly.

  See `shard_model_from_config` for more details on the config.
  """
  import jax
  from jax.sharding import NamedSharding, PartitionSpec
  from torchax.interop import torch_view

  jax_mark_sharding = torch_view(jax.lax.with_sharding_constraint)

  def shard_param(tensor, spec: tuple[str, ...]):
    sharding = NamedSharding(mesh, PartitionSpec(*spec))
    # Note that when sharding the parameters, we need to use eager calls to
    # move tensors to the device. jax_mark_sharding only works under jit,
    # and models are usually constructed eagerly in torchax.
    return torch_view(
      jax.make_array_from_callback(
        tensor.shape, sharding, lambda slice_index: tensor[slice_index]
      )
    )

  def shard_output(tensor, spec: tuple[str, ...]):
    sharding = NamedSharding(mesh, PartitionSpec(*spec))
    return jax_mark_sharding(tensor, sharding)

  return shard_model_from_config(model, config, shard_output, shard_param)


def shard_torch_xla_model_from_config(
  model: torch.nn.Module,
  config: dict,
  mesh: Optional["torch_xla.distributed.spmd.Mesh"] = None,  # type: ignore  # noqa: F821
):
  """
  Given a config of pattern to partition spec, shard a torch_xla model accordingly.

  See `shard_model_from_config` for more details on the config.

  If `mesh` is not given, there must be a registered global mesh.
  """
  import torch_xla.distributed.spmd as xs

  def shard_activation(tensor, spec: tuple[str, ...]):
    the_mesh = mesh if mesh is not None else xs.get_global_mesh()
    assert the_mesh is not None, "No mesh found"
    return xs.mark_sharding_with_gradients(tensor, the_mesh, spec)

  # TODO(https://github.com/pytorch/xla/issues/8809): If we shard parameters with
  # `MarkShardingFunction.apply`, that causes Mixtral to OOM. Gradient HLO arrays end up
  # living much longer than needed.
  def shard_param(tensor, spec: tuple[str, ...]):
    the_mesh = mesh if mesh is not None else xs.get_global_mesh()
    assert the_mesh is not None, "No mesh found"
    return xs.mark_sharding(tensor, the_mesh, spec).global_tensor

  return shard_model_from_config(
    model,
    config,
    shard_activation,
    shard_param,
  )


def _process_tail_index_syntax(
  config: dict,
  shard_output: Callable[[torch.Tensor, tuple[str, ...]], torch.Tensor],
):
  """
  Replace names like `model.layers.*[0]` with `model.layers.*` and add a
  custom sharding function that just shards the requested element.
  """

  shard_output_fns = {}
  config = copy(config)
  for name in list(config.keys()):
    matches = TAIL_INDEX_REGEX.search(name)
    if matches is None:
      continue
    new_name = name[: matches.start()]
    assert new_name not in config, f"{name} conflicts with {new_name}"
    config[new_name] = config[name]
    del config[name]
    index = int(matches.group(0)[1:-1])

    # This lambda indexes into the module output and shards the element we want.
    def make_shard_output_fn(index):
      def shard_output_fn(outputs, spec):
        sharded = shard_output(outputs[index], spec)
        out_list = list(outputs[:index]) + [sharded] + list(outputs[index + 1 :])
        if isinstance(outputs, list):
          return out_list
        else:
          assert isinstance(outputs, tuple), (
            f"outputs must be a list or tuple, got {type(outputs)}"
          )
          return tuple(out_list)

      return shard_output_fn

    shard_output_fns[new_name] = make_shard_output_fn(index)

  return config, shard_output_fns


def _process_sharding_name(name):
  """Replace integers in param name with *."""

  def is_integer(t):
    try:
      int(t)
      return True
    # pylint: disable-next=all
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)


class ShardedModule(torch.nn.Module):
  """
  Wraps an existing module and marks its output as sharded.
  """

  def __init__(self, mod, mark_sharding, spec):
    super().__init__()
    self._orig_mod = mod
    self.mark_sharding = mark_sharding
    self.spec = spec

  def forward(self, *args, **kwargs):
    return self.mark_sharding(self._orig_mod(*args, **kwargs), self.spec)
