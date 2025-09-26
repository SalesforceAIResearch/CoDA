"""WebDataset implementation for efficient TPU training with GCS TAR files."""

import random
import logging
import os
import sys
import json
import io
import gcsfs
import torch
import torch_xla.runtime as xr
import webdataset as wds
import numpy as np

logger = logging.getLogger(__name__)


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0

if not is_main_process():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def list_gcs_shards(gs_prefix: str):
    # If you have an explicit list, just return it.
    # Otherwise use gcsfs to expand a pattern/prefix.
    fs = gcsfs.GCSFileSystem()  # GCE default credentials
    path = gs_prefix.replace("gs://", "") + "/**/*.tar"
    urls = fs.glob(path, recursive=True)
    urls = [f"gs://{u}" if not u.startswith("gs://") else u for u in urls]
    return urls


# Decoder for numpy - needs (key, data) arguments
def numpy_decoder(key, data):
    if isinstance(data, bytes):
        return np.load(io.BytesIO(data))
    return data


def split_by_replica(urls):
    urls = [x for x in urls]
    rank, world_size = xr.process_index(), xr.process_count()
    if world_size > 1:
        return urls[rank::world_size]
    return urls


def split_by_datloader_worker(urls):
    """Split urls per worker
    Selects a subset of urls based on Torch get_worker_info.
    Used as a shard selection function in Dataset.
    replaces wds.split_by_worker"""

    urls = [url for url in urls]

    assert isinstance(urls, list)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        return urls[wid::num_workers]
    else:
        return urls


def webdataset_collate_fn(batch, column_names=None):
    """Collate function for WebDataset samples.

    When using to_tuple(), WebDataset returns tuples of numpy arrays.
    DataLoader passes a list of these tuples to the collate function.
    Each tuple contains multiple numpy arrays (one per column).
    
    Args:
        batch: List of tuples, each containing numpy arrays
        column_names: Optional list of column names. If None, uses defaults.
    """
    if not batch:
        return {}
    
    # Get the number of columns from the first sample
    num_columns = len(batch[0]) if isinstance(batch[0], tuple) else 1
    
    # If only one column (backward compatibility)
    if num_columns == 1:
        arrays = [item[0] if isinstance(item, tuple) else item for item in batch]
        input_ids = torch.from_numpy(np.stack(arrays)).long()
        return {"input_ids": input_ids}
    
    # Multiple columns case
    result = {}    
    for col_idx in range(num_columns):
        # Extract the col_idx-th array from each sample
        arrays = [item[col_idx] for item in batch]
        # Stack into a batch tensor
        column_tensor = torch.from_numpy(np.stack(arrays)).long()
        result[column_names[col_idx]] = column_tensor
    
    return result


def create_webdataset_collate_fn(column_names=None):
    """Create a collate function with specified column names.
    
    Args:
        column_names: List of column names to use in the output dictionary
        
    Returns:
        A collate function that can be used with DataLoader
    """
    def collate_fn(batch):
        return webdataset_collate_fn(batch, column_names)
    return collate_fn


def make_webdataset(
    path: str,
    shard_urls: list[str] = None,
    sample_shuffle=65536,  # sample-level shuffle buffer
    checkpoint_dir: str = None,
    seed: int = 42,
    columns: list[str] = None,  # List of column names to extract (e.g., ["input_ids", "attention_mask"])
):
    """
    Builds a WebDataset pipeline that:
    - shuffles shards and splits them by node and by worker
    - reads samples from tar
    - shuffles samples
    - returns individual samples for DataLoader to batch
    
    Args:
        path: GCS path prefix to search for tar files
        shard_urls: Optional explicit list of shard URLs
        sample_shuffle: Size of sample-level shuffle buffer
        checkpoint_dir: Directory to save data files list
        seed: Random seed for shuffling
        columns: List of column names to extract from tar files. 
                Each column should be stored as "{column_name}.npy" in the tar files.
                If None, defaults to single column mode for backward compatibility.
                Example: ["input_ids", "attention_mask", "labels"]
    
    Returns:
        WebDataset that yields tuples of numpy arrays (one per column)
        
    Usage:
        # Single column (backward compatible)
        dataset = make_webdataset("gs://bucket/data/")
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=webdataset_collate_fn)
        
        # Multiple columns
        columns = ["input_ids", "attention_mask", "labels"]
        dataset = make_webdataset("gs://bucket/data/", columns=columns)
        collate_fn = create_webdataset_collate_fn(columns)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        
    Note:
        Your TAR files should contain files named like:
        - 000000000001.input_ids.npy
        - 000000000001.attention_mask.npy  
        - 000000000001.labels.npy
        Where "000000000001" is the sample key and "input_ids.npy" is the extension.
    """
    # Pipeline definition
    random.seed(seed)
    if shard_urls is None:
        if is_main_process():
            logger.info(f"shard_urls is None, searching for all tar files in {path}")
        shard_urls = list_gcs_shards(path)
        random.shuffle(shard_urls)
    if is_main_process():
        logger.info(f"shard_urls: {shard_urls}")
        logger.info(f"number of shard_urls: {len(shard_urls)}")

    dataset = wds.WebDataset(
        shard_urls,
        nodesplitter=split_by_replica,
        workersplitter=split_by_datloader_worker,
        shardshuffle=False,
        repeat=False,
        handler=wds.handlers.warn_and_continue
    ).shuffle(sample_shuffle)

    if checkpoint_dir is not None and is_main_process():
        logger.info(f"Saving shard_urls.json to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(f"{checkpoint_dir}/data_files.json", "w") as f:
            json.dump(shard_urls, f, indent=4)

    dataset = dataset.decode(numpy_decoder)
    
    # Handle multiple columns or default to single column
    if columns is None:
        # Default behavior: single column (backward compatibility)
        dataset = dataset.to_tuple("npy")
    else:
        # Multiple columns: extract each specified column
        # WebDataset to_tuple works with file extensions/suffixes after the key
        # For files named like "000000000001.input_ids.npy", we need to specify 
        # the extensions that come after the sample key
        column_extensions = [f"{col}.npy" for col in columns]
        dataset = dataset.to_tuple(*column_extensions)
    
    return dataset


# def make_gcs_webdataset(
#     tar_urls: List[str],
#     batch_size: int,
#     num_workers: int = 128,
#     seed: int = 42,
#     shuffle_buffer: int = 32768,
#     prefetch_factor: int = 2,
#     persistent_workers: bool = True,
# ) -> DataLoader:
#     """
#     Create a WebDataset DataLoader for TPU training with proper sharding and shuffling.

#     Args:
#         tar_urls: List of GCS URLs to TAR files (e.g., ["gs://bucket/file1.tar", ...])
#         batch_size: Per-worker batch size (global_batch_size / num_workers)
#         num_workers: Total number of TPU workers (default 128)
#         seed: Random seed for shuffling
#         shuffle_buffer: Number of samples to buffer for shuffling
#         prefetch_factor: Number of batches to prefetch per dataloader worker
#         persistent_workers: Keep dataloader workers alive between epochs

#     Returns:
#         DataLoader configured for efficient TPU training
#     """
#     # Get current world size
#     world_size = xr.process_count()

#     if world_size != num_workers:
#         logger.warning("Actual world size %d differs from expected %d", world_size, num_workers)
#         num_workers = world_size

#     # Ensure deterministic shuffling across workers
#     random.seed(seed)

#     # Shuffle TAR files globally (same order for all workers)
#     shuffled_urls = tar_urls.copy()
#     random.shuffle(shuffled_urls)

#     if is_main_process():
#         logger.info("Total TAR files: %d", len(shuffled_urls))
#         logger.info("First 5 TAR files: %s", shuffled_urls[:5])

#     # Create WebDataset with proper sharding
#     dataset = (
#         wds.WebDataset(
#             shuffled_urls,
#             shardshuffle=True,  # Shuffle shards (TAR files) per worker
#             nodesplitter=wds.split_by_node,  # Split shards across nodes
#             workersplitter=wds.split_by_worker,  # Split within node
#         )
#         .shuffle(shuffle_buffer, initial=shuffle_buffer // 4)  # In-memory shuffle
#         .decode("numpy")  # Decode numpy arrays
#         .to_tuple("input_ids.npy")  # Extract input_ids numpy array
#         .batched(batch_size, partial=False)  # Create batches, drop incomplete
#     )

#     # Convert tuples to proper batch format
#     def collate_fn(batch):
#         """Convert WebDataset batch format to standard PyTorch format."""
#         # batch is a list of tuples, each tuple contains (input_ids numpy array,)
#         # Convert numpy arrays to tensors and stack
#         input_ids = torch.stack([torch.from_numpy(item[0]) for item in batch])
#         return {
#             "input_ids": input_ids,
#         }

#     # Create DataLoader with optimal settings for TPU
#     dataloader = DataLoader(
#         dataset,
#         batch_size=None,  # Batching is handled by WebDataset
#         num_workers=2,  # 2-4 workers per TPU core is usually optimal
#         pin_memory=False,  # Faster CPU-TPU transfer
#         prefetch_factor=prefetch_factor,
#         persistent_workers=persistent_workers,
#         collate_fn=collate_fn,
#     )

#     return dataloader
