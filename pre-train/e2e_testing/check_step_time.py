#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path


def check_step_time(output_dir: str, lower_bound: float, upper_bound: float):
  # List train_metrics.json file in the given directory.
  try:
    result = subprocess.run(
      ["gcloud", "storage", "ls", "-r", f"{output_dir}/train_metrics.json"],
      capture_output=True,
      text=True,
      check=True,
    )
  except subprocess.CalledProcessError as e:
    print(f"Error running gcloud storage ls: {e}", file=sys.stderr)
    return 1

  # Count the number of matching files.
  files = [line for line in result.stdout.splitlines() if line.strip()]
  count = len(files)
  if count != 1:
    print(
      f"Error: Expected one train_metrics.json file in {output_dir}, found {count}.",
      file=sys.stderr,
    )
    return 1

  print(f"Found metrics file at: {files[0]}")

  # Download and parse the metrics file.
  metrics_file = files[0]
  with tempfile.TemporaryDirectory() as temp_dir:
    try:
      subprocess.run(
        ["gcloud", "storage", "cp", metrics_file, temp_dir],
        check=True,
      )
    except subprocess.CalledProcessError as e:
      print(f"Error downloading metrics file: {e}", file=sys.stderr)
      return 1

    metrics_file_path = f"{temp_dir}/train_metrics.json"
    from torchprime.metrics.metrics import Metrics

    metrics = Metrics.load(Path(metrics_file_path))

  # Compare against the required bounds.
  step_execution_time = metrics.step_execution_time
  if step_execution_time is None:
    print("Error: step_execution_time is None in metrics.")
    return 1

  time_seconds = step_execution_time.total_seconds()
  print(f"Step execution time: {time_seconds:.4f} seconds")

  if time_seconds < lower_bound:
    print(
      f"""
Error: step time too fast!

step_execution_time {time_seconds:.4f} seconds is below the lower bound {lower_bound:.4f} seconds.

Refer to https://github.com/AI-Hypercomputer/torchprime/blob/main/e2e_testing/README.md#what-to-do-when-step-time-is-out-of-range
for more information.
""",
      file=sys.stderr,
    )
    return 1

  if time_seconds > upper_bound:
    print(
      f"""
Error: step time too slow!

step_execution_time {time_seconds:.4f} seconds exceeds the upper bound {upper_bound:.4f} seconds.

Refer to https://github.com/AI-Hypercomputer/torchprime/blob/main/e2e_testing/README.md#what-to-do-when-step-time-is-out-of-range
for more information.
""",
      file=sys.stderr,
    )
    return 1

  print("Metrics check passed.")
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: check_step_time.py <output_dir> <lower_bound> <upper_bound>")
    sys.exit(1)
  sys.exit(check_step_time(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])))
