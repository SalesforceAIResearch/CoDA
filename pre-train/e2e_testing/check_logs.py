#!/usr/bin/env python3
import re
import sys

FINISHED_MARKER = r"Finished training run"


def check_logs(file_path):
  try:
    with open(file_path) as f:
      log_data = f.read()
  except Exception as e:
    print(f"Error reading log file {file_path}: {e}")
    return 1

  # Validate that the log contains the expected patterns.
  if not re.search(FINISHED_MARKER, log_data):
    print(f"Error: '{FINISHED_MARKER}' not found in logs")
    return 1
  print(f"Found '{FINISHED_MARKER}' in logs")

  print("Logs check passed.")
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: check_logs.py <log_file>")
    sys.exit(1)
  sys.exit(check_logs(sys.argv[1]))
