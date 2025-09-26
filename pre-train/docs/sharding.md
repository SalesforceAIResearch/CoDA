# How to configure model sharding

This guide demonstrates how to shard models in torchprime. Sharding is the task
of taking a large model and distributing the weights, activations, and compute
across multiple devices. It is also known as multi-dimensional parallelism.

Since torchprime uses the SPMD paradigm, consider familiarizing yourself with the
[PyTorch/XLA SPMD user guide][spmd-guide] first.

## Single device model + sharding configs

Going from single device training to distributed training usually doesn't require
changes to model code. For example, if you take a look at the [Llama][llama]
model, it doesn't call any sharding/parallelism APIs in the code. This makes for
a familiar experience for eager mode GPU users and is generally good software
engineering practice.

Instead, torchprime shards the model by modifying its parameters and layers
according to configurations specified at run time. The logic is implemented in
[`shard_model.py`][shard-model] and invoked from the [trainer][trainer]. Here is
an example sharding configuration that implements the [FSDP (aka ZeRO-3)][fsdp]
strategy for Llama dense models:

<!-- GitHub markdown embed -->
https://github.com/AI-Hypercomputer/torchprime/blob/b123c0cc157c28f32a0f6588f19e2d352d2a3617/torchprime/torch_xla_models/configs/model/sharding/llama-fsdp.yaml#L1-L17


> 📝 NOTE: Compared to the FSDP wrapper in PyTorch upstream that uses eager
> collective operations, torchprime stages out a computation graph corresponding
> to the training step where specific nodes in the graph are annotated with
> sharding constraints. The XLA compiler then propagates those constraints to all
> nodes in the graph and inserts the appropriate collective operations
> automatically. In contrast to eager PyTorch, the XLA compiler decides the best
> weight prefetching schedules.

## How to shard weights

To shard a particular weight in the model, simply spell out its name as it
appears in the [state dict][state-dict], followed by the
[partition spec][partition-spec]:

```yaml
# Shard the first dimension of the embedding layer along the `fsdp` mesh axis.
model.embed_tokens.weight: [fsdp, null]
```

The name can also contain wildcards, which matches any integer. This is useful
for uniformly sharding all layers in a `nn.ModuleList`:

```yaml
# Shard all query projection weights with the `[fsdp, null]` partition spec.
model.layers.*.self_attn.q_proj.weight: [fsdp, null]
```

`null` is interpreted to be `None` in Python and means to replicate that
tensor dimension across all devices.

Internally, this is implemented as a call to
`xs.mark_sharding(weight, mesh, partition_spec)`. See
[`shard_torch_xla_model_from_config`][shard_torch_xla_model_from_config].

## How to shard activations

Besides sharding weights, you can also shard module outputs, also called
_activations_. To shard the output of a particular module, simply spell out its
name as it appears in the module tree. For example, the Llama dense model class
`LlamaForCausalLM` has two submodules: `model` and `lm_head`. The `model`
submodule has a nested submodule called `layers`, containing the sequence of
decoder layers. You can shard their outputs this way:

```yaml
# Shard the batch dimension of the language modeling head output along `fsdp` mesh axis.
lm_head: [fsdp, null, null]

# Shard the batch dimension of each decoder layer outputs along `fsdp` mesh axis.
model.layers.*: [fsdp, null, null]
```

Internally, this is implemented as a call to
`xs.mark_sharding_with_gradients(output, mesh, partition_spec)`. See
[`shard_torch_xla_model_from_config`][shard_torch_xla_model_from_config].

In FSDP (ZeRO-3), you must shard the batch dimension of all module outputs
uniformly. However, thanks to the SPMD sharding propagation pass in the XLA
compiler, you don't have to exhaustively list out every single module in the
configuration file.

The detailed propagation rules are spelled out in the [GSPMD][GSPMD] paper.
A rule of thumb is that if an operation preserves an input dimension in its
output (e.g. the batch dim in the case of batched matmul), then the sharding of
the output dimension will inherit the sharding of the corresponding input
dimension.

### Indexing syntax

Sometimes the output of a module is a `list` or `tuple`. In order to identify
which tensor element to shard, add a `[i]` suffix to the name where `i` is an
integer that indexes into the module output. For example, you'll find the
following configuration in the FSDP sharding config for Mixtral because the
Mixtral decoder layer returns both the embedding and also a load balancing loss.

```yaml
# Shard the first output of the decoder layer
model.layers.*[0]: [fsdp, null, null]
```

## Conclusion

Thanks to the SPMD feature in the PyTorch/XLA framework, you can flexibly
implement multi-dimensional parallelism purely from configuration without
modifying the modeling code. The examples in this guide demonstrates 1D FSDP
but you may read on to [`llama-fsdp-tp.yaml`][llama-fsdp-tp] for combined
FSDP + Tensor Parallelism (TP) 2D sharding. You may override the sharding from
the command line using Hydra config syntax or create new configuration files
as needed.

<!-- xrefs -->

[spmd-guide]: https://pytorch.org/xla/master/perf/spmd_basic.html
[llama]: ../torchprime/torch_xla_models/llama/model.py
[llama-fsdp]: ../torchprime/torch_xla_models/configs/model/sharding/llama-fsdp.yaml
[llama-fsdp-tp]: ../torchprime/torch_xla_models/configs/model/sharding/llama-fsdp-tp.yaml
[shard-model]: ../torchprime/sharding/shard_model.py
[trainer]: ../torchprime/torch_xla_models/train.py
[fsdp]: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
[state-dict]: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
[partition-spec]: https://pytorch.org/xla/master/perf/spmd_basic.html#partition-spec
[shard_torch_xla_model_from_config]: https://github.com/AI-Hypercomputer/torchprime/tree/master/torchprime/sharding/shard_model.py#L201
[GSPMD]: https://arxiv.org/pdf/2105.04663
