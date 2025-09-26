import os
import uuid
from datetime import datetime
from pathlib import Path

import click

from torchprime.metrics.metrics import Metrics
from torchprime.metrics.mfu import get_num_chips_and_tflops_per_chip


def get_metrics(base_artifact_path: str, jobset_name_for_outputs: str) -> dict | None:
  """
  Extracts metrics from train_metrics.json of slice 0 and worker 0
  found in the artifact output directories.

  Args:
      base_artifact_path: The base path where jobset-specific artifact directories are located.
      jobset_name_for_outputs: The name of the jobset, used to find its 'outputs' subdirectory.

  Returns:
      A dictionary containing model_id, metrics_step_time (in seconds),
      and metrics_mfu, or None if the metrics file is not found.
  """
  metric_file_path = Path(
    os.path.join(
      base_artifact_path,
      jobset_name_for_outputs,
      "outputs",
      "0-0",  # Slice 0, Worker 0: Extract metrics from primary host (0-0) as it is consistent across hosts in SPMD.
      "train_metrics.json",
    )
  )
  if not metric_file_path.exists():
    click.echo(f"Metrics file not found at {metric_file_path}", err=True)
    return None

  metrics_data = Metrics.load(metric_file_path)

  return {
    "model_id": metrics_data.model,
    "metrics_step_time": metrics_data.step_execution_time.total_seconds(),
    "metrics_mfu": metrics_data.mfu,
  }


def prepare_benchmark_summary(
  process_returncode: int,
  jobset_name: str,
  tpu_type: str,
  **kwargs,
) -> dict:
  """
  Constructs a summary dictionary of the benchmark run.

  Args:
      process_returncode: The return code of the main training script.
      jobset_name: The name of the jobset.
      tpu_type: The type of TPU used (e.g., "v4-8").
      kwargs: Rest of the schema that is included in the dictionary.

  Returns:
      A dictionary representing the benchmark summary, conforming to the
      workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema structure.

  """
  current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
  unique_id_suffix = str(uuid.uuid4()).split("-")[0]  # Short UUID
  run_id = f"{jobset_name}-{current_time_str}-{unique_id_suffix}"

  hardware_num_chips, tflops_per_chip = get_num_chips_and_tflops_per_chip(tpu_type)

  click.echo(
    f"Tpu type: {tpu_type}, hardware_num_chips: {hardware_num_chips}, tflops_per_chip: {tflops_per_chip}"
  )

  return {
    "run_id": run_id,
    "result_success": (process_returncode == 0),
    "software_id": "torchprime",
    "hardware_id": tpu_type,
    "hardware_num_chips": hardware_num_chips,
    **kwargs,
  }
