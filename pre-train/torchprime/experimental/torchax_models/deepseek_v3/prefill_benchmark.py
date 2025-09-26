import functools
import json
import time

import jax
import jax.numpy as jnp
import model as ds_model
import torch
import torchax
import torchax.interop
import torchax.ops.mappings as tx_mappings
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from model import ModelArgs, Transformer
from torchax import interop
from torchax.interop import JittableModule


def _process_sharding_name(name):
  """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

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


def _get_sharding_sepc(sharding_map, name):
  sharding_spec = sharding_map.get(name)
  if sharding_spec is not None:
    return sharding_spec
  sharding_spec = sharding_map.get(_process_sharding_name(name))
  return sharding_spec


def make_weight_shard(weight_meta, slice_index):
  weight_shard_meta = weight_meta[slice_index]
  with torchax.default_env():
    return interop.jax_view(
      torch.randn(weight_shard_meta.shape, dtype=weight_shard_meta.dtype)
    )


def make_cache_shard(weight_meta, slice_index):
  weight_shard_meta = weight_meta[slice_index]
  return jnp.zeros(
    weight_shard_meta.shape, dtype=tx_mappings.t2j_dtype(weight_shard_meta.dtype)
  )


def create_sharded_weights(model, mesh, sharding_map, env):
  res = {}
  for name, weight_meta in model.state_dict().items():
    sharding_spec = _get_sharding_sepc(sharding_map, name)
    if sharding_spec is None:
      print("Skipping weight:", name)
      continue
    sharding = NamedSharding(mesh, P(*sharding_spec))
    res[name] = env.j2t_iso(
      jax.make_array_from_callback(
        weight_meta.shape, sharding, functools.partial(make_weight_shard, weight_meta)
      )
    )
  return res


def create_sharded_kv_cache(cache_dict, mesh, env):
  res = {}
  # shard at num device
  sharding = NamedSharding(mesh, P(None, None, name0, None))
  for name, weight_meta in cache_dict.items():
    if name.endswith("_cache"):
      res[name] = env.j2t_iso(
        jax.make_array_from_callback(
          weight_meta.shape, sharding, functools.partial(make_cache_shard, weight_meta)
        )
      )
  return res


name0 = "tp0"
# name1 = "tp1"
sharding_map_1d_tp = {
  "embed.weight": (name0, None),
  "layers.*.attn.wq.weight": (None, name0),
  "layers.*.attn.wq.bias": (name0,),
  "layers.*.attn.wkv_a.weight": (None, name0),
  "layers.*.attn.kv_norm.weight": (name0,),
  "layers.*.attn.wkv_b.weight": (name0, None),
  "layers.*.attn.wkv_b.bias": (name0,),
  "layers.*.attn.wo.weight": (name0, None),
  "layers.*.attn.wo.bias": (name0, None),
  "layers.0.ffn.w1.weight": (name0, None),
  "layers.0.ffn.w1.bias": (name0,),
  "layers.0.ffn.w2.weight": (None, name0),
  "layers.0.ffn.w2.bias": (name0,),
  "layers.0.ffn.w3.weight": (name0, None),
  "layers.0.ffn.w3.bias": (name0,),
  "layers.*.ffn.cond_ffn.w1": (None, name0, None),
  "layers.*.ffn.cond_ffn.w2": (None, None, name0),
  "layers.*.ffn.cond_ffn.w3": (None, name0, None),
  "layers.*.ffn.gate.weight": (None, name0),
  "layers.*.ffn.gate.bias": (name0,),
  "layers.*.ffn.shared_experts.w1.weight": (name0, None),
  "layers.*.ffn.shared_experts.w1.bias": (name0,),
  "layers.*.ffn.shared_experts.w2.weight": (None, name0),
  "layers.*.ffn.shared_experts.w2.bias": (name0,),
  "layers.*.ffn.shared_experts.w3.weight": (name0, None),
  "layers.*.ffn.shared_experts.w3.bias": (name0,),
  "layers.*.attn_norm.weight": (name0,),
  "layers.*.ffn_norm.weight": (name0,),
  "norm.weight": (name0,),
  "head.weight": (name0, None),
  "head.bias": (name0,),
  "freqs_cis": (),
}


def _replicate(x, env, mesh):
  with jax.default_device(jax.devices("cpu")[0]):
    xj = env.to_xla(x).jax()
  xj = env.j2t_iso(
    jax.make_array_from_callback(xj.shape, NamedSharding(mesh, P()), lambda a: xj)
  )
  return xj


def main(config=None, seqlen=2048, batch_size=1):
  config_dict = None
  if config is not None:
    with open(config) as f:
      config_dict = json.load(f)

  print("======= multi_device =======")
  torch.set_default_dtype(torch.bfloat16)
  env = torchax.default_env()
  config_dict = config_dict or {}

  env.config.use_torch_native_for_cpu_tensor = False

  torch.manual_seed(42)
  torchax.enable_performance_mode()
  torchax.enable_globally()
  args = ModelArgs(**config_dict)

  dev_array = create_device_mesh((len(jax.devices()),), allow_split_physical_axes=True)
  mesh = Mesh(dev_array, (name0,))

  torch.set_default_device("meta")
  with env, torch.device("meta"):
    model = Transformer(args)

  jitted = JittableModule(model)
  freqs_cis = ds_model.precompute_freqs_cis(args)
  freqs_cis = _replicate(freqs_cis, env, mesh)
  jitted.buffers["freqs_cis"] = freqs_cis

  print(model)
  caches_dict = create_sharded_kv_cache(jitted.buffers, mesh, env)
  sharded_weights = create_sharded_weights(model, mesh, sharding_map_1d_tp, env)

  jitted.params = sharded_weights
  jitted.buffers.update(caches_dict)

  with mesh:
    x = torch.randint(0, args.vocab_size, (1, seqlen))
    x = _replicate(x, env, mesh)
    input_pos = torch.arange(seqlen, device="jax")
    for i in range(5):
      step_start = time.perf_counter()
      logits = jitted(x, input_pos)
      jax.block_until_ready(torchax.tensor.t2j(logits))
      step_end = time.perf_counter()
      print(
        i,
        "step latency: ",
        step_end - step_start,
      )


if __name__ == "__main__":
  import fire

  fire.Fire(main)
