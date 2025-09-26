"""
Parse a profile to determine the median duration of a training step.
"""

import glob
import os
import statistics
import sys

from torchprime.metrics.xplane_pb2 import XSpace  # type: ignore


def step_duration_from_latest_profile(profile_dir: str) -> float:
  profile_dir = os.path.abspath(profile_dir)
  profiles = [
    (f, os.path.getctime(f))
    for f in glob.glob(f"{profile_dir}/**/*.xplane.pb", recursive=True)
  ]
  newest_profile, _time = max(profiles, key=lambda v: v[1])
  return analyze_step_duration(newest_profile)


def analyze_step_duration(file_path: str) -> float:
  xspace = XSpace()

  # Read and parse the xplane proto
  with open(file_path, "rb") as f:
    print(f"Parsing {file_path}", file=sys.stderr)
    xspace.ParseFromString(f.read())

  return analyze_step_duration_from_pb(xspace)


def analyze_step_duration_from_pb(xspace: XSpace) -> float:
  offsets = []
  unique_names = set()

  for plane in xspace.planes:
    # Only consider /device:TPU:0
    if plane.name != "/device:TPU:0":
      continue
    print(f"Plane ID: {plane.id}, Name: {plane.name}", file=sys.stderr)

    for line in plane.lines:
      # Only consider XLA Modules line
      if line.name != "XLA Modules":
        continue
      print(f"  Line ID: {line.id}, Name: {line.name}", file=sys.stderr)

      # Collect offsets and event names
      for event in line.events:
        name = plane.event_metadata[event.metadata_id].name
        offset_ps = event.offset_ps
        unique_names.add(name)
        offsets.append(offset_ps)
        print(
          f"    Event Metadata Name: {name}, "
          f"ID: {event.metadata_id}, Offset: {offset_ps / 1e12:.3f} s, "
          f"Duration: {event.duration_ps / 1e12:.3f} s",
          file=sys.stderr,
        )

  # Make sure we have events at all
  if not offsets:
    raise ValueError("No events found in the given XSpace data.")

  # Confirm we have exactly one unique event name
  if len(unique_names) > 1:
    raise ValueError(f"Ambiguous event names found in XSpace: {unique_names}")

  inferred_event_name = list(unique_names)[0]
  # Sort offsets to compute consecutive differences
  offsets.sort()

  if len(offsets) < 2:
    raise ValueError("Not enough events to compute step durations.")

  # Compute durations based on consecutive offset differences
  durations = []
  for i in range(len(offsets) - 1):
    # Convert picoseconds to seconds
    durations.append((offsets[i + 1] - offsets[i]) / 1e12)

  # If we have no intervals, we can't compute durations
  event_count = len(durations)
  if event_count == 0:
    raise ValueError("Not enough events to compute step durations.")

  print(
    f"Got {event_count} intervals for event '{inferred_event_name}'", file=sys.stderr
  )

  # If fewer than 3 intervals, compute a simple average
  if event_count < 3:
    print(
      "[Warning] Not enough events found to drop outliers.",
      file=sys.stderr,
    )
    return sum(durations) / len(durations)

  # Otherwise, use the median
  average_duration = statistics.median(durations)
  return average_duration


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_proto_file>")
    sys.exit(1)
  proto_file_path = sys.argv[1]
  try:
    median_duration = analyze_step_duration(proto_file_path)
    print(f"Median step duration: {median_duration:.4f}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
