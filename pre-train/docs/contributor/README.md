# Contributing to torchprime

When developing, use `pip install -e '.[dev]'` to install dev dependencies such
as linter and formatter.

## How to run tests

```sh
pytest
```

## How to run some of the tests, and re-run them whenever you change a file

```sh
tp -i test ... # replace with path to tests/directories
```

## How to format

```sh
ruff format
```

## How to lint

```sh
ruff check [--fix]
```

You can install a Ruff VSCode plugin to check errors and format files from the
editor.

## How to run inside the docker container locally

You can also run locally without XPK with docker. When running inside the docker
container, it will use the same dependencies and build process as used in the
XPK approach, improving the hermeticity and reliability.

```sh
tp docker-run torchprime/torch_xla_models/train.py
```

This will run the torchprime docker image locally. You can also add `--use-hf`
to run HuggingFace model locally.

```sh
tp docker-run --use-hf torchprime/hf_models/train.py
```

## Run distributed training with local torch/torch_xla wheel

torchprime supports running with user specified torch and torch_xla wheels
placed under `local_dist/` directory. The wheel will be automatically installed
in the docker image when use `tp run` command. To use the wheel, add flag
`--use-local-wheel` to `tp run` command:

```sh
tp run --use-local-wheel torchprime/hf_models/train.py
```

The wheels should be built inside a [PyTorch/XLA development docker
image][torch_xla_dev_docker] or the PyTorch/XLA VSCode Dev Container to minimize
compatibility issues.

## Uploading Benchmark Result to Bigquery

You can upload benchmark results to a BigQuery database. This allows you to
track results without manual recording and provides an easier way to view them.

To use this feature, configure your BigQuery table and pass the
`--upload-metrics` flag. If not specified, `bq-project`, `bq-dataset`, and
`bq-table` default to `tpu-pytorch`, `benchmark_dataset_test`, and
`benchmark_experiment` respectively. The default table can be find
[here](http://shortn/_YMeB6vfEXc)

```
tp use \
    --cluster <XPK CLUSTER NAME> \
    --project my-gcp-project \
    --zone us-east5-b \
    --num-slices 1 \
    --tpu-type v6e-256 \
    --artifact-dir gs://bucket/dir
    --bq-project <bq-project>
    --bq-dataset <bq-datase>
    --bq-table <bq-table>
    --upload-metrics
```

When you run a training job, you can add comments to the upload with
`--comments` which will be shown in `logs_comments` columns
```
tp run --comments="Test Comments" \
    torchprime/torch_xla_models/train.py \
    model=llama-3-8b \
    global_batch_size=256 \
    ici_mesh.fsdp=256
```

You can view the database by querying directly in the BigQuery console or by
using Google Sheets.

To view the database in a spreadsheet:

1. Open a new Google Sheet.
2. Go to Data > Data connectors > Connect to BigQuery.
