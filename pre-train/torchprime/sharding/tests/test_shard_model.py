import pytest
import torch
import torch.nn as nn

from torchprime.sharding.shard_model import (
  ShardedModule,
  shard_model,
  shard_model_from_config,
  shard_torch_xla_model_from_config,
  shard_torchax_model_from_config,
)


class SimpleLinear(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(128, 64)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(64, 128)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return z


class MockShardedTensor(torch.Tensor):
  """
  This class simulates a sharded tensor.
  """

  def __init__(self, orig):
    super().__init__()
    self.orig = orig


class MockShardedModule(nn.Module):
  """
  This class simulates an activation (output) sharded module.
  """

  def __init__(self, orig):
    super().__init__()
    self.orig = orig

  def forward(self, x):
    return self.orig(x)


def test_traverse_weights():
  model = SimpleLinear()
  visited = set()

  def shard_weight(weight, name):
    visited.add(name)
    return MockShardedTensor(weight)

  model = shard_model(model, shard_weight, lambda x, _: x)
  assert visited == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}

  # Check that all weights are sharded.
  sharded = set()
  for name, param in model.named_parameters():
    assert isinstance(param.data, MockShardedTensor)
    sharded.add(name)
  assert sharded == visited


def test_traverse_modules():
  model = SimpleLinear()
  visited = set()

  def shard_activation(module, name):
    visited.add(name)
    return MockShardedModule(module)

  model = shard_model(model, lambda x, _: x, shard_activation)
  # Note that the empty string refers to the whole model.
  assert visited == {"", "fc1", "relu", "fc2"}

  # Check that all modules are sharded.
  sharded = set()
  for name, mod in model.named_modules():
    if isinstance(mod, MockShardedModule):
      sharded.add(name)
  assert len(sharded) == len(visited)


def test_traverse_modules_nested():
  model = nn.Sequential(SimpleLinear(), SimpleLinear())
  visited = set()

  def shard_activation(module, name):
    visited.add(name)
    return MockShardedModule(module)

  model = shard_model(model, lambda x, _: x, shard_activation)
  # Note that the empty string refers to the whole model.
  assert visited == {
    "",
    "0",
    "1",
    "0.fc1",
    "0.relu",
    "0.fc2",
    "1.fc1",
    "1.relu",
    "1.fc2",
  }

  # Check that all modules are sharded.
  sharded = set()
  for name, mod in model.named_modules():
    if isinstance(mod, MockShardedModule):
      sharded.add(name)
  assert len(sharded) == len(visited)


def test_shard_model_from_config_mock():
  model = nn.Sequential(SimpleLinear(), SimpleLinear())
  config = {
    "*.fc1": ["fsdp", None],
    "*.relu": ["fsdp", None],
    "*.fc2": ["fsdp", None],
  }

  num_shard_output_calls = 0

  def shard_output(output, spec):
    nonlocal num_shard_output_calls
    assert spec == ("fsdp", None)
    num_shard_output_calls += 1
    return output

  model = shard_model_from_config(model, config, shard_output, lambda x, _: x)

  # Verify that output mark sharding is called for the right number of times.
  # There should be 6 sharding calls, because there are two `SimpleLinear`,
  # and we annotated 3 modules in each.
  inputs = torch.randn((32, 128))
  output = model(inputs)
  assert output.shape == (32, 128)
  assert num_shard_output_calls == 6


def test_shard_model_from_config_multi_output_mock():
  class Foo(nn.Module):
    def __init__(self) -> None:
      super().__init__()

    def forward(self, x):
      return torch.tensor(100), x

  class Bar(nn.Module):
    def __init__(self) -> None:
      super().__init__()

    def forward(self, x):
      return x, torch.tensor(100)

  class MyMod(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.foo = Foo()
      self.bar = Bar()

    def forward(self, x):
      a, b = self.foo(x)
      c, d = self.bar((a, b))
      return c, d

  model = MyMod()
  config = {
    "foo[0]": ["fsdp", None],
    "bar[1]": ["fsdp", None],
  }

  num_shard_output_calls = 0

  def shard_output(output, spec):
    nonlocal num_shard_output_calls
    assert spec == ("fsdp", None)
    torch.testing.assert_close(output, torch.tensor(100))
    num_shard_output_calls += 1
    return output

  model = shard_model_from_config(model, config, shard_output, lambda x, _: x)

  # Verify that output mark sharding is called for the right number of times.
  # There should be 2 sharding calls for `foo` and `bar` in total.
  x = torch.tensor(42)
  c, d = model(x)
  torch.testing.assert_close(d, torch.tensor(100))
  a, b = c
  torch.testing.assert_close(a, torch.tensor(100))
  torch.testing.assert_close(b, x)
  assert num_shard_output_calls == 2


def test_shard_model_from_config_torchax():
  # Create 4 CPU devices for SPMD
  with temporary_env({"XLA_FLAGS": "--xla_force_host_platform_device_count=4"}):
    import jax
    import torchax
    import torchax.interop
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from torchax.interop import JittableModule, jax_view

    with torchax.default_env():
      model = SimpleLinear().to("jax")

    config = {
      "fc1": ["fsdp", None],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": ["fsdp", None],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    }

    # Define mesh for test
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("fsdp",))

    model = shard_torchax_model_from_config(model, config, mesh)

    # In order to shard activations, corresponding modules are
    # wrapped with ShardedModule.
    assert isinstance(model.fc1, ShardedModule)
    assert isinstance(model.fc2, ShardedModule)

    # Check the sharding of weights.
    state_dict = model.state_dict()
    expected_sharding = {
      "fc1._orig_mod.weight": (None, "fsdp"),
      "fc1._orig_mod.bias": (None,),
      "fc2._orig_mod.weight": (None, "fsdp"),
      "fc2._orig_mod.bias": (None,),
    }
    seen_count = 0
    for name, param in state_dict.items():
      param = jax_view(param.data)
      expectation = expected_sharding.get(name)
      if expectation is None:
        continue
      assert param.sharding.spec == expectation
      seen_count += 1
    assert seen_count == len(expected_sharding)

    # Run the model and check the sharding of outputs.
    jit_model = JittableModule(model)
    with torchax.default_env():
      inputs = torch.randn((32, 128), device="jax")
      output = jit_model(inputs)

    assert isinstance(output, torch.Tensor)
    assert jax_view(output).sharding.spec == ("fsdp",)


def test_shard_model_from_config_torch_xla():
  import numpy as np
  import torch_xla
  import torch_xla.runtime as xr
  from torch_xla.distributed.spmd import Mesh

  # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
  # work on PyTorch/XLA. We must run this on the TPU for now.
  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")

  xr.use_spmd()

  model = SimpleLinear().to(torch_xla.device())

  config = {
    "fc1": ["fsdp", None],
    "fc1.weight": [None, "fsdp"],
    "fc1.bias": [None],
    "fc2": ["fsdp", None],
    "fc2.weight": [None, "fsdp"],
    "fc2.bias": [None],
  }

  # Define mesh for test
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (num_devices,)
  assert num_devices > 1, "The TPU VM should have more than 1 device for SPMD testing"
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ("fsdp",))

  model = shard_torch_xla_model_from_config(model, config, mesh)
  torch_xla.sync()

  # In order to shard activations, corresponding modules are
  # wrapped with ShardedModule.
  assert isinstance(model.fc1, ShardedModule)
  assert isinstance(model.fc2, ShardedModule)

  # Check the sharding of weights.
  state_dict = model.state_dict()
  none_fsdp_sharded = (
    f"{{devices=[1,{num_devices}]{','.join(str(v) for v in range(num_devices))}}}"
  )
  none_sharded = "{replicated}"
  expected_sharding = {
    "fc1._orig_mod.weight": none_fsdp_sharded,
    "fc1._orig_mod.bias": none_sharded,
    "fc2._orig_mod.weight": none_fsdp_sharded,
    "fc2._orig_mod.bias": none_sharded,
  }
  seen_count = 0
  for name, param in state_dict.items():
    expectation = expected_sharding.get(name)
    if expectation is None:
      continue
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(param)
    assert sharding_spec == expectation
    seen_count += 1
  assert seen_count == len(expected_sharding)

  # Run the model and check the sharding of outputs.
  inputs = torch.randn((32, 128), device=torch_xla.device())
  torch_xla.sync()
  output = model(inputs)
  torch_xla.sync()
  assert isinstance(output, torch.Tensor)
  fsdp_none_sharded = (
    f"{{devices=[{num_devices},1]{','.join(str(v) for v in range(num_devices))}}}"
  )
  sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(output)
  assert sharding_spec == fsdp_none_sharded


def temporary_env(env_dict):
  import os
  from contextlib import contextmanager

  @contextmanager
  def _temporary_env(env_dict):
    old_env = {}
    for key, value in env_dict.items():
      old_env[key] = os.environ.get(key)
      os.environ[key] = value
    try:
      yield
    finally:
      for key, value in old_env.items():
        if value is None:
          del os.environ[key]
        else:
          os.environ[key] = value

  return _temporary_env(env_dict)
