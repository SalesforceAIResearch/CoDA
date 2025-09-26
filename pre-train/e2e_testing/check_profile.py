#!/usr/bin/env python3
import subprocess
import sys


def check_profile(profile_dir):
  try:
    # List all .xplane.pb files recursively in the given directory.
    result = subprocess.run(
      ["gcloud", "storage", "ls", "-r", f"{profile_dir}/**/*.xplane.pb"],
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
      f"Error: Expected exactly one .xplane.pb file in {profile_dir}, found {count}.",
      file=sys.stderr,
    )
    print("Files found:", result.stdout)
    return 1

  print(f"Found profile at: {files[0]}")
  print("Profile check passed.")
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: check_profile.py <profile_dir>")
    sys.exit(1)
  sys.exit(check_profile(sys.argv[1]))
