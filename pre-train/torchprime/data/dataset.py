import os
import json
import os
import random
import logging
from itertools import chain
from typing import Sequence
from glob import glob
from dotenv import load_dotenv
load_dotenv()

import torch_xla.runtime as xr
from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict, IterableDataset, concatenate_datasets
from transformers.tokenization_utils import PreTrainedTokenizerBase

MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)
if MOUNTED_GCS_DIR is None:
  raise ValueError("MOUNTED_GCS_DIR is not set or GCS is not mounted.")

logger = logging.getLogger(__name__)

def is_main_process():
  """Check if this is the main process (rank 0)."""
  return xr.process_index() == 0

def group_texts(examples, block_size):
  """Group texts into blocks of specified size for training."""
  # Concatenate all texts.
  concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
  total_length = (total_length // block_size) * block_size
  # Split by chunks of max_len.
  result = {
    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    for k, t in concatenated_examples.items()
  }
  result["labels"] = result["input_ids"].copy()
  return result


DATASET_TYPES = {
    "fineweb_edu_dedup": (".parquet", "parquet"),  # 512.29 GiB - educational web, largest
    "dclm-baseline-1.0-shuffled": (".json", "json"), # 251.32 GiB - web crawl
    "RedPajama_math": (".json", "json"),           # 168.48 GiB - math dataset
    "stackv2_Python_shuffled": (".json", "json"),  # 186.7 GiB - Python code
    "open-web-math": (".json", "json"),            # 45.39 GiB - math problems
    "Redpajama-Arxiv": (".json", "json"),          # 26.37 GiB - academic papers
    "Falcon-refinedweb": (".json", "json"),        # 130.89 GiB - web text
    "the-stack-v2-train-smol": (".json", "json"),  # 135.9 GiB - code
    "cosmopedia_v2_parquet": (".parquet", "parquet"), # 113.96 GiB - synthetic textbook
    "c4_2023-14": (".json", "json"),               # 77.85 GiB - web crawl
    "RedPajama": (".json", "json"),                # 65.53 GiB - mixed content
    "Wikipedia_en": (".json", "json"),             # 19.53 GiB - encyclopedia
    "python_edu": (".jsonl", "json"),              # 13.88 GiB - code
    "Gutenberg": (".json", "json"),                # 10.68 GiB - literature  
    "DM_Mathematics": (".json", "json"),           # 5.69 GiB - math, smallest
}

def make_gcs_pretokenized_dataset(
  path: str,
  data_files: list[str] = None,
  seed: int = 42,
  checkpoint_dir: str = None,
) -> IterableDataset | Dataset:
  """
  Search for all parquet files in the given path and load them into a dataset.
  Shuffle the files first.
  """
  random.seed(seed)
  if data_files is None:
    if is_main_process():
      logger.info(f"data_files is None, searching for all parquet files in {path}")
    data_files = glob(f"{path}/**/*.parquet", recursive=True)
    random.shuffle(data_files)
  if is_main_process():
    logger.info(f"data_files: {data_files}")
    logger.info(f"number of data_files: {len(data_files)}")
  data = load_dataset(
    "parquet",
    data_files=data_files,
    streaming=True,
    split="train",
  )
  if checkpoint_dir is not None and is_main_process():
    logger.info(f"Saving data_files.json to {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(f"{checkpoint_dir}/data_files.json", "w") as f:
      json.dump(data_files, f, indent=4)

  if isinstance(data, IterableDataset):
    data = data.shuffle(seed=seed, buffer_size=32768)
  else:
    data = data.shuffle(seed=seed, keep_in_memory=True)
  return data

def make_gcs_dataset(
  names: list[str],
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
  seed: int = 42,
  weights: Sequence[float] = None,
) -> IterableDataset:
  if any(name not in DATASET_TYPES for name in names):
    raise ValueError(f"Dataset {names} not found in {DATASET_TYPES}")
  
  def tokenize_fn(examples):
    texts = [example + tokenizer.eos_token for example in examples["text"]]
    return tokenizer(texts)

  data_mixture = []
  for name in names:
    extension, data_type = DATASET_TYPES[name]
    data_files = glob(f"{MOUNTED_GCS_DIR}/data/xgen_cleaned_data/{name}/*{extension}")
    print(f"Loading dataset {name}, data_files example: {data_files[0]}")
    data = load_dataset(
      data_type,
      data_files=data_files,
      streaming=True,
      split="train",
    )
    # Get columns from first example but convert dict_keys to list immediately
    columns = list(list(data.take(1))[0].keys())

    print(f"Shuffling dataset {name}")
    data = data.shuffle(seed=seed, buffer_size=32768)

    print(f"Pretokenizing dataset {name}")
    data = data.map(
      tokenize_fn,
      batched=True,
      remove_columns=columns,
    )

    print(f"Grouping dataset {name}")
    data = data.map(
      lambda examples: group_texts(examples, block_size),
      batched=True,
    )
    data_mixture.append(data)

  if len(data_mixture) == 1:
    return data_mixture[0]
  else:
    return concatenate_datasets(data_mixture)
    # return interleave_datasets(data_mixture, probabilities=weights, seed=seed)


def make_huggingface_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  # Downloading and loading a dataset from the hub.
  data = load_dataset(
    name,
    config_name,
    cache_dir=cache_dir,
  )
  assert isinstance(data, DatasetDict)
  data = data[split]

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  data = data.map(lambda examples: group_texts(examples, block_size), batched=True)
  return data
