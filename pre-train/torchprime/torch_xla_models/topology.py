"""
`topology` obtains information on the accelerator topology in the current environment.
"""

import math
from functools import lru_cache

import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from omegaconf import DictConfig

from torchprime.mesh.custom_mesh import maybe_get_custom_mesh


@lru_cache
def is_multi_slice() -> bool:
  """
  Check if we are running in multi-slice/pod environment.
  """
  num_slices = get_num_slices()
  return num_slices > 1


@lru_cache
def get_num_slices():
  """
  Get the number of slices (ICI connected groups of TPUs) in the environment.
  """
  device_attributes = xr.global_runtime_device_attributes()
  slice_indices = []
  for d in device_attributes:
    index = d.get("slice_index", 0)
    assert isinstance(index, int), (
      f"Expected slice_index to be an int, got {type(index)} {index}"
    )
    slice_indices.append(index)
  num_slices = max(slice_indices) + 1
  return num_slices


def is_1d_sharding(mesh_values: tuple[int, ...]) -> bool:
  """
  Check if the devices are distributed along exactly one dimension of the virtual device mesh.
  """
  non_trivial = [v for v in mesh_values if v > 1]
  if len(non_trivial) == 0:
    # Special case if the mesh looks like `[1, 1, 1, 1]`.
    return True
  return len(non_trivial) == 1


def get_mesh(config: DictConfig, num_devices: int | None = None) -> xs.Mesh:
  """
  Get an optimized virtual device mesh based on the requested ICI and DCN mesh dimensions.

  If `num_devices` is not specified, it will be queried from the XLA runtime.
  """

  if num_devices is None:
    num_devices = xr.global_runtime_device_count()
  assert num_devices == math.prod(list(config.ici_mesh.values())) * math.prod(
    list(config.dcn_mesh.values())
  ), (
    f"Mesh is not using all the available devices. The environment has {num_devices} devices. \
    Mesh requested: ICI={config.ici_mesh}, DCN={config.dcn_mesh}"
  )

  ici_mesh_shape = (
    config.ici_mesh.data,
    config.ici_mesh.fsdp,
    config.ici_mesh.tensor,
    config.ici_mesh.expert,
  )
  dcn_mesh_shape = (
    config.dcn_mesh.data,
    config.dcn_mesh.fsdp,
    config.dcn_mesh.tensor,
    config.dcn_mesh.expert,
  )

  # If there is a faster custom mesh available, use that instead.
  devices = maybe_get_custom_mesh(
    ici_mesh_shape=ici_mesh_shape,
    dcn_mesh_shape=dcn_mesh_shape,
    num_devices=num_devices,
    num_slices=get_num_slices(),
  )
  if devices is not None:
    mesh_shape = tuple(np.multiply(ici_mesh_shape, dcn_mesh_shape).tolist())
    return xs.Mesh(devices, mesh_shape, ("data", "fsdp", "tensor", "expert"))

  # TODO(https://github.com/pytorch/xla/issues/8683): When nightly torch_xla no longer crashes
  # during training, we will be able to remove this special case and always use `HybridMesh` in
  # both single and multi slice.
  if is_multi_slice():
    mesh = xs.HybridMesh(
      ici_mesh_shape=ici_mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      axis_names=("data", "fsdp", "tensor", "expert"),
    )
  else:
    for k, v in config.dcn_mesh.items():
      assert v == 1, (
        f"DCN mesh dim `{k}` must be 1 in single slice environments, got {v}"
      )
    mesh_shape = (
      config.ici_mesh.data,
      config.ici_mesh.fsdp,
      config.ici_mesh.tensor,
      config.ici_mesh.expert,
    )
    mesh = xs.Mesh(
      list(range(num_devices)), mesh_shape, ("data", "fsdp", "tensor", "expert")
    )
  return mesh
