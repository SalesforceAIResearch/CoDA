from functools import partial

import pytest
import torch
import torch_xla
from functorch.compile import aot_function  # type:ignore

from torchprime.torch_xla_models.offloading import (
  offload_name,
  remat_all_and_offload_these_inputs,
)

from .test_remat_all import _count_function_calls, _make_get_graph_compiler


def test_offload_simple():
  def fn(x, y):
    x = offload_name(x, "x")
    a = x @ y
    b = torch.sin(a)
    c = torch.exp(b)
    return c

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  x = torch.randn((1, 10)).to(torch_xla.device()).detach().requires_grad_(True)
  y = torch.randn((10, 1)).to(torch_xla.device()).detach().requires_grad_(True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=partial(remat_all_and_offload_these_inputs, names_to_offload=["x"]),
  )
  out = fn_compiled(x, y)
  out.backward()
  hlo: str = torch_xla._XLAC._get_xla_tensors_hlo([x.grad])

  # Test that the graph that computes `x.grad` involves two device placement ops,
  # one to move `x` to host, another to move `x` to device.
  #
  # For reference, an "offload to host" HLO call looks like this:
  #
  #   %custom-call.4 = f32[1,10] custom-call(f32[1,10]{1,0} %p1.3),
  #       custom_call_target="annotate_device_placement",
  #       custom_call_has_side_effect=true,
  #       api_version=API_VERSION_UNSPECIFIED,
  #       frontend_attributes={_xla_buffer_placement="pinned_host"}
  #
  # And an "move back to device" HLO call looks like this:
  #
  #   %custom-call.5 = f32[1,10] custom-call(f32[1,10] %custom-call.4),
  #       custom_call_target="annotate_device_placement",
  #       custom_call_has_side_effect=true,
  #       api_version=API_VERSION_UNSPECIFIED,
  #       frontend_attributes={_xla_buffer_placement="device"}
  #
  assert hlo.count("annotate_device_placement") == 2
  assert hlo.count('xla_buffer_placement="pinned_host"') == 1
  assert hlo.count('xla_buffer_placement="device"') == 1

  # Test that the forward graph contains an offloading op.
  fw_graph = get_fwd()
  assert _count_function_calls(fw_graph, "place_to_host") == 1
  assert _count_function_calls(fw_graph, "place_to_device") == 0

  # Test that the backward graph contains an offloading-back op.
  bw_graph = get_bwd()
  assert _count_function_calls(bw_graph, "place_to_host") == 0
  assert _count_function_calls(bw_graph, "place_to_device") == 1


def test_offload_multiple():
  def fn(x, y):
    x = offload_name(x, "x")
    y = offload_name(y, "y")
    a = x @ y
    b = torch.sin(a)
    c = torch.exp(b)
    return c

  fw_compiler, _ = _make_get_graph_compiler()
  bw_compiler, _ = _make_get_graph_compiler()

  x = torch.randn((1, 10)).to(torch_xla.device()).detach().requires_grad_(True)
  y = torch.randn((10, 1)).to(torch_xla.device()).detach().requires_grad_(True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=partial(
      remat_all_and_offload_these_inputs, names_to_offload=["x", "y"]
    ),
  )
  out = fn_compiled(x, y)
  out.backward()
  hlo: str = torch_xla._XLAC._get_xla_tensors_hlo([x.grad, y.grad])

  # Test that the graph involves four device placement ops, two for each grad.
  assert hlo.count("annotate_device_placement") == 4
  assert hlo.count('xla_buffer_placement="pinned_host"') == 2
  assert hlo.count('xla_buffer_placement="device"') == 2


def test_offload_middle_tensor():
  def fn(x, y):
    x = torch.sin(x)
    y = torch.sin(y)
    x = offload_name(x, "x")
    y = offload_name(y, "y")
    x = torch.exp(x)
    y = torch.exp(y)
    return x @ y

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  x = torch.randn((1, 10)).to(torch_xla.device()).detach().requires_grad_(True)
  y = torch.randn((10, 1)).to(torch_xla.device()).detach().requires_grad_(True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=partial(
      remat_all_and_offload_these_inputs, names_to_offload=["x", "y"]
    ),
  )
  out = fn_compiled(x, y)
  out.backward()
  hlo: str = torch_xla._XLAC._get_xla_tensors_hlo([x.grad, y.grad])

  # Test that the graph involves four device placement ops, two for each grad.
  assert hlo.count("annotate_device_placement") == 4
  assert hlo.count('xla_buffer_placement="pinned_host"') == 2
  assert hlo.count('xla_buffer_placement="device"') == 2

  # Check the FX graph.
  fw_graph = get_fwd()
  bw_graph = get_bwd()

  # Forward graph does two sin and two exp
  assert _count_function_calls(fw_graph, "sin") == 2
  assert _count_function_calls(fw_graph, "exp") == 2

  # Backward graph does two cos (backward of sin) and two exp (backward of exp).
  # It won't do two sin because the input into the exp is offloaded. Therefore,
  # there is no need to recompute the input to the exp.
  assert _count_function_calls(bw_graph, "sin") == 0
  assert _count_function_calls(bw_graph, "cos") == 2
  assert _count_function_calls(bw_graph, "exp") == 2


def test_offload_wrong_name():
  def fn(x, y):
    x = offload_name(x, "x")
    a = x @ y
    b = torch.sin(a)
    c = torch.exp(b)
    return c

  fw_compiler, _ = _make_get_graph_compiler()
  bw_compiler, _ = _make_get_graph_compiler()

  x = torch.randn((1, 10)).to(torch_xla.device()).detach().requires_grad_(True)
  y = torch.randn((10, 1)).to(torch_xla.device()).detach().requires_grad_(True)

  with pytest.raises(ValueError, match="Did not find"):
    fn_compiled = aot_function(
      fn,
      fw_compiler=fw_compiler,
      bw_compiler=bw_compiler,
      partition_fn=partial(
        remat_all_and_offload_these_inputs, names_to_offload=["asdfasfasdsadf"]
      ),
    )
    out = fn_compiled(x, y)
    out.backward()
