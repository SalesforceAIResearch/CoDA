import time

import jax
import torch
import torchax as tx
import torchax.interop
from torch.utils import _pytree as pytree
from torch_xla.core import xla_model as xm


def run_model_torchax(model, batch_size, number_of_runs, eager):
  torchax.enable_globally()
  model = model.to("jax")

  wait_val = tx.interop.torch_view(jax.block_until_ready)

  if not eager:
    model = tx.interop.JittableModule(model)

  times = []
  for _i in range(number_of_runs):
    inputs = model.get_sample_inputs(batch_size)
    args, kwargs = pytree.tree_map_only(torch.Tensor, lambda t: t.to("jax"), inputs)
    args, kwargs = wait_val((args, kwargs))
    start = time.perf_counter()
    res = model.forward(*args, **kwargs)
    wait_val(res)
    end = time.perf_counter()
    times.append(end - start)
  return times


def run_model_xla(model, batch_size, number_of_runs):
  import torch_xla

  model = model.to("xla")

  times = []
  for _i in range(number_of_runs):
    inputs = model.get_sample_inputs(batch_size)
    args, kwargs = pytree.tree_map_only(torch.Tensor, lambda t: t.to("xla"), inputs)
    torch_xla.sync(wait=True)
    start = time.perf_counter()
    res = model.forward(*args, **kwargs)
    xm.unlazy([res])
    torch_xla.sync(wait=True)
    end = time.perf_counter()
    times.append(end - start)
  return times
