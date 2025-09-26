import pytest
from omegaconf import OmegaConf

from torchprime.torch_xla_models import topology


@pytest.fixture(scope="session", autouse=True)
def enable_spmd():
  import torch_xla.runtime as xr

  xr.use_spmd()


def test_basic():
  import torch_xla.runtime as xr

  # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
  # work on PyTorch/XLA. We must run this on the TPU for now.
  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")

  multi_slice = topology.is_multi_slice()
  num_slices = topology.get_num_slices()
  if multi_slice:
    assert num_slices > 1
  else:
    assert num_slices == 1


def test_is_1d_sharding():
  assert topology.is_1d_sharding((1,))
  assert topology.is_1d_sharding((1, 1, 1, 1))
  assert topology.is_1d_sharding((1, 1, 2, 1))
  assert topology.is_1d_sharding((2,))
  assert topology.is_1d_sharding((2, 1))
  assert not topology.is_1d_sharding((2, 2))
  assert not topology.is_1d_sharding((1, 2, 2, 1))


def test_get_mesh():
  import torch_xla.runtime as xr

  # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
  # work on PyTorch/XLA. We must run this on the TPU for now.
  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")

  if topology.is_multi_slice():
    pytest.skip("This test only works on single slice")

  # Test a custom mesh
  config = OmegaConf.create(
    {
      "ici_mesh": {
        "data": 1,
        "fsdp": 64,
        "tensor": 4,
        "expert": 1,
      },
      "dcn_mesh": {
        "data": 1,
        "fsdp": 1,
        "tensor": 1,
        "expert": 1,
      },
    }
  )
  mesh = topology.get_mesh(config, num_devices=256)
  from torchprime.tests.test_custom_mesh import get_64x4_reference_device_ids_1pod

  assert (
    mesh.get_logical_mesh() == get_64x4_reference_device_ids_1pod().reshape(1, 64, 4, 1)
  ).all()

  # Test a simple FSDP mesh
  config = OmegaConf.create(
    {
      "ici_mesh": {
        "data": 1,
        "fsdp": xr.global_runtime_device_count(),
        "tensor": 1,
        "expert": 1,
      },
      "dcn_mesh": {
        "data": 1,
        "fsdp": 1,
        "tensor": 1,
        "expert": 1,
      },
    }
  )
  mesh = topology.get_mesh(config)
  assert mesh.get_logical_mesh().shape == (1, xr.global_runtime_device_count(), 1, 1)
