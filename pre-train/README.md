<div align="center">

# torchprime

#### High-performance training for PyTorch on Cloud TPU

</div>
<br /><br />

`torchprime` is a reference implementation for training PyTorch models on TPU. It
is designed to showcase best practices for large-scale, high-performance model
training using `torch_xla` ([project][torch_xla]), with
minimal changes to model code. It aims to demystify training on XLA-based
accelerators, providing clear patterns and best practices to help the PyTorch
community unlock top performance and efficiency on Google Cloud TPUs.

`torchprime` is under active development, and we're eager for feedback and input
from the PyTorch community.

## Setup and Use Guide 
The environment setup is alike what's described in sections below except that we do **NOT** have multi VM training and `xpk` support (instead we have a single VM with multihost workers)
So most of our setup or training is based on the following snippet
```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='pwd'
```
which is sent through a local machine.

High level steps:
1. Create a TPU VM
2. Setup environments. You can just run `bash setup_tpu.sh` **with your TPU VM name and zone**
4. Prepare `.env` according to `env.example`
5. Run `bash copy_env.sh` to copy the .env file to TPU
6. Create your own branch `git checkout -b your_branch`
7. Add your recipes under `recipes/your_recipe.sh`
8. Change the tpu/branch/recipe names in `remote_run.sh` and run training! Or use recipe with: `bash remote_run.sh --recipes recipes/your_recipe.sh`


## Environment setup

For development and debugging purposes it is useful to run `torchprime`
locally on a TPU VM. You can create a single-host TPU VM using
this guide: https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm
Or you can create TPU VM from the Google Cloud Console for your cloud project.
Spot quota is available for v5e and v6e chips in multiple regions:
https://cloud.google.com/tpu/docs/regions-zones

Make sure that you are using the correct runtime when creating
your VM: https://cloud.google.com/tpu/docs/runtimes#pytorch_and_jax

For example:

```sh
gcloud compute tpus tpu-vm create <tpu-name> \
  --zone=us-central1-a \
  --accelerator-type=v6e-4 \
  --version=v2-alpha-tpuv6e \
  --spot
```

Once the VM is created you can `ssh` into it:
https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#tpu-connect

```
gcloud compute tpus tpu-vm ssh <tpu-name> --zone=<zone>
```

### Install `torch_xla`

Before installing `torchprime`, you will need to first install
[torch_xla][torch_xla] following its respective project README.
You need to install nightly version of
PyTorch/XLA.

Test that the environment is correctly installed and configured.
Start `python` interpreter and run following:

```python
import torch_xla.core.xla_model as xm
print("XLA devices:", xm.get_xla_supported_devices())
print("Default XLA device type:", xm.xla_device_hw(xm.xla_device()))
```

### Install `torchprime`

Make sure that `pip` and `setuptools` are up-to-date:

```sh
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools==69.5.1
```

```sh
git clone https://github.com/AI-Hypercomputer/torchprime.git
cd torchprime
pip install -e '.[dev]'
```

## Examples

### Local training

Here is a simple example of training on a single TPU VM with 4 TPU chips.
Train Llama 3 8B using `torch_xla`:

```sh
export HF_TOKEN='...your huggingface token...'
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python3 torchprime/torch_xla_models/train.py
```

By default, this will distribute the model over 4 TPU chips attached to the VM
using the [FSDP parallelism strategy][fsdp].

The first two training steps will take a while to compile. After that, the graphs
will hit the compilation cache and you should see something like this:

```
...
[2025-04-29 06:58:36,445][__main__][INFO] - Num replicas: 1
[2025-04-29 06:58:36,447][__main__][INFO] - Starting training
[2025-04-29 06:58:36,448][__main__][INFO] -     Max step: 15
[2025-04-29 06:58:36,448][__main__][INFO] -     Global batch size: 4
[2025-04-29 07:01:16,240][__main__][INFO] - Epoch: 0, step: 0, loss: 12.5574, trace time: 155003.85 ms
[2025-04-29 07:04:24,182][__main__][INFO] - Epoch: 0, step: 10, loss: 9.7555, trace time: 1564.54 ms
...
```

Refer to `README.md` in `torchprime/torch_xla_models` for more details.

### Configuring training

`torchprime` uses [hydra][hydra] to read configurations (e.g. model name, batch
size) from the command line and `.yaml` files.

In the `torch_xla_models` directory, you'll find a `configs/default.yaml`. That
specifies the default configuration for the trainer. You may override configs on
the command line with a `key=value` syntax. For example, the following command
will train Mixtral 8x7B with a global batch size of 256, and set the FSDP SPMD
ICI mesh axis length to 64:

```sh
python3 torchprime/torch_xla_models/train.py \
    model=mixtral-8x7b \
    training_mode=pretrain \
    dataset=wikitext \
    global_batch_size=256 \
    ici_mesh.fsdp=64
```

You may refer to the hydra docs for other ways to specify configs.

### Multi-VM distributed training

`torchprime` uses [xpk][xpk] as the standard path for iterating on distributed
training code that needs to run on multiple VMs.

First teach `torchprime` about the XPK cluster it is using, the artifact storage
location, the Google Cloud project/zone, and the TPU topology. You only need
to do this on first clone or when switching to a different topology or cluster.
Example:

```sh
tp use \
    --cluster <XPK CLUSTER NAME> \
    --project <my-gcp-project> \
    --docker-project <my-docker-project-if-it-is-different-from-gcp-project> \
    --zone us-east5-b \
    --num-slices 1 \
    --tpu-type v6e-256 \
    --artifact-dir <my-gs-bucket-dir>
```

`torchprime` natively supports [multi-slice or multi-pod][multi-slice] training.
`--num-slices` specifies the number of [slices][tpu-slice] used by the workload.
`--tpu-type` specifies the [accelerator type][accelerator-type] in each slice.
To do multi-pod training, simply specify a `--tpu-type` that is as big as a
[pod][tpu-pod].

After configuring the cluster, prepend `tp run` to a particular Python file you
would like to run remotely, including arguments, e.g.

```sh
# Train Llama 3.0 8B on 256 chips
tp run torchprime/torch_xla_models/train.py \
    model=llama-3-8b \
    global_batch_size=256 \
    ici_mesh.fsdp=256
```

`tp run` will broadcast the specified command to all VMs in the XPK cluster,
which is the convention for running SPMD distributed workloads. See `tp run
--help` for more advanced features.

The version of `torch_xla` used by `tp run` is specified in `pyproject.toml`.

#### Env vars passed to the workload

`tp run` will pick up these environment variables locally and proxy them to the
distributed workload, if found:

- `HF_TOKEN`: HuggingFace token
- `XLA_IR_DEBUG`: [torch_xla debugging flag][torch_xla_debug_env]
- `XLA_HLO_DEBUG`: [torch_xla debugging flag][torch_xla_debug_env]
- `LIBTPU_INIT_ARGS`: XLA flags that affect compilation and execution behavior

#### Additional CLI arguments passed to the workload

Besides forwarding your command line arguments, `tp run` will add:

- `profile_dir=[...]`: path to a [profile][torch_xla_profile] directory
  accessible by the workload
- `output_dir=[...]` path to a directory where the workload may store output
  artifacts such as metrics and checkpoints

#### Troubleshooting distributed training setup

See the guide to [troubleshoot distributed training][troubleshoot-distributed]
for troubleshooting tools and an FAQ. 

## Supported Models

torchprime has implementations for the following models:

- [Llama 3.0 8B](torchprime/torch_xla_models/README.md#llama-30-8b-on-v6e-256)
- [Llama 3.1 8B](torchprime/torch_xla_models/README.md#llama-31-8b-on-v6e-256)
- [Llama 3.1 70B](torchprime/torch_xla_models/README.md#llama-31-70b-on-v6e-256)
- [Llama 3.1 405B](torchprime/torch_xla_models/README.md#llama-31-405b-on-v6e-256)
- [Mixtral 8x7B](torchprime/torch_xla_models/README.md#mixtral-8x7b-on-v6e-256)

All implemented models will have a training recipe, and are backed by unit tests.

Interested in another model? File an [issue](https://github.com/AI-Hypercomputer/torchprime/issues).

## Structure

This repo will contain a set of reference models that we have optimized and runs
well on TPU. The best performing scaling configuration (parallelism techniques,
checkpointing, etc.) for a model on various hardwares will be provided for ease
of reproducibility.

`docs` contains guides for optimizing performance and debugging issues.

`torchprime/launcher` contains scripts to train a model on a large TPU cluster.

`torchprime/data` contains dataset and data loading utilities.

`torchprime/torch_xla_models` contains model implementations using `torch_xla`.

`torchprime/experimental` contains experimental model implementations

Finally, each model may also provide a GPU "original" version that illustrates
and attributes where this model code came from, if any. This also helps to
showcase what changes we have done to make it performant on TPU. The original
version is not expected to be run.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

Refer to the [contributor guide](./docs/contributor/README.md) to get started.

## Questions and suggestions

Feel free to ask questions in the [Discussions][discussions] panel, or to look
at previous questions and discussions.

## License

This project is licensed under the New BSD License - see the [LICENSE](LICENSE)
file for details.

For more information on PyTorch/XLA, visit the [official
documentation](https://github.com/pytorch/xla).

[torch_xla]: https://github.com/pytorch/xla
[fsdp]: https://jax-ml.github.io/scaling-book/training/#fully-sharded-data-parallelism-fsdp
[discussions]: https://github.com/AI-Hypercomputer/torchprime/discussions/categories/q-a
[xpk]: https://github.com/AI-Hypercomputer/xpk
[torch_xla_debug_env]:
    https://github.com/pytorch/xla/blob/master/docs/source/learn/troubleshoot.md#environment-variables
[torch_xla_profile]:
    https://github.com/pytorch/xla/blob/master/docs/source/learn/troubleshoot.md#performance-profiling
[hydra]: https://hydra.cc/docs/intro/
[torch_xla_dev_docker]:
    https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md#manually-build-in-docker-container
[tpu-pod]: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod
[tpu-slice]: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices
[accelerator-type]: https://cloud.google.com/tpu/docs/multislice-introduction#concepts
[multi-slice]: https://cloud.google.com/tpu/docs/multislice-introduction
[troubleshoot-distributed]: https://github.com/AI-Hypercomputer/torchprime/docs/troubleshoot-distributed.md