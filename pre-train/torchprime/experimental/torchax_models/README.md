# torchax models

## Llama

The `llama` directory demonstrates training of llama model. The model
implementation is forked at the official reference implementation at
https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
this file is kept as `model_original.py` for reference.

Then, 3 versions of the model is provided: 

* `model.py` is a modification to `model_original.py` by replacing fairscale layers
with regular `torch.nn` layers.

* `model_with_scan.py` is a modification on `model.py` that uses `scan` to iterate
  the layers of model

* `model_with_collectives.py` is a further modification on `model_with_scan.py` by
  using manual collectives (`all_gather`, `all_reduce`) along with `shard_map` to simulate
  how PyTorch native on multi-GPUs would have done. Instead of relying on GSPMD.

Choosing which version of the model to run can be controlled by passing
`--model_impl` flag to the `run.py` script.

If passing `--model_impl=titan` It will import the llama model from torchtitan 
and use that instead, if so, please install torchtitan in the current environment,
following instructions here: https://github.com/pytorch/torchtitan


## Install dependencies

First one need a valid installation of `torchax`, torch, and jax

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/pytorch/xla.git
cd xla/torchax
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax tensorflow tensorboard-plugin-profile
pip install -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Running locally on a TPU VM

Setup environment as per [README][README-examples].

```sh
python run.py model_impl=<orig|scan|scan_manual>
```

### Llama 3.1 8B on v6e-8

Recipe for global batch size 8, sequence length 8192.
Expected step duration: 1.7s. MFU: 30%.

```sh
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"

python run.py model_impl=scan tp=1 global_batch_size=8 seqlen=8192
```

## Running on a XPK cluster

First follow the [distributed training][distributed-training] guide to setup the
cluster information.

Run `tp run <local command>` to run the training command on the XPK cluster.

### Llama 3.1 405B on 2 pods of v6e-256

Recipe for global batch size 256, sequence length 8192.
Expected step duration: 46.12s. MFU: 28.7%.

```sh
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"

tp run torchprime/experimental/torchax_models/run.py \
    global_batch_size=256 \
    model_type=405B \
    seqlen=8192 \
    use_custom_offload=True \
    use_custom_mesh=True \
    model_impl=scan \
    tp=4 \
    unroll_layers=1
```

[README-examples]: ../../README.md#examples
[distributed-training]: ../../README.md#distributed-training
