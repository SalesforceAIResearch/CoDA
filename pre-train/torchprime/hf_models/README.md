# Run huggingface transformer models

For contributors to torchprime, `tp run` also supports running the huggingface
trainer, for debugging and comparison. This module implements an adapter over
the huggingface trainer.

To run the huggingface trainer, you can clone
[huggingface/transformers][hf-transformers] under the root directory of
torchprime and name it as `local_transformers`. This allows you to pick any
branch or make code modifications in transformers for experiment:

```sh
git clone https://github.com/huggingface/transformers.git local_transformers
```

If huggingface transformer doesn't exist, torchprime will automatically clone
the repo and build the docker for experiment. To switch to huggingface models,
add flag `--use-hf` to `tp run` command:

```sh
tp run --use-hf torchprime/hf_models/train.py
```

[hf-transformers]: https://github.com/huggingface/transformers
