import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from torchprime.launcher import upload_metrics_to_bq

# Workaround for MegaScale crash
#
# TODO(https://github.com/pytorch/xla/issues/8683): Remove the
# `--megascale_grpc_enable_xor_tracer=false` flag when libtpu is updated
xla_flags = os.environ.get("LIBTPU_INIT_ARGS", "")
xla_flags = f"{xla_flags} --megascale_grpc_enable_xor_tracer=false"
os.environ["LIBTPU_INIT_ARGS"] = xla_flags

# Get the artifact dir from env var.
gcs_artifact_dir = os.environ["TORCHPRIME_ARTIFACT_DIR"]
assert gcs_artifact_dir.startswith("gs://"), (
  f"{gcs_artifact_dir} must be in a GCS bucket (start with gs://)"
)
gcs_artifact_dir = gcs_artifact_dir.removeprefix("gs://")
gcs_bucket, *gcs_artifact_subdir = gcs_artifact_dir.split("/")

# Some passes in XLA can only dump to regular folders. Hence,
# mount the GCS bucket at `/tmp/gcs-mount` using gcsfuse.
gcs_mount = "/tmp/gcs-mount"
os.makedirs(gcs_mount, exist_ok=True)
subprocess.run(["gcsfuse", gcs_bucket, gcs_mount], check=True)

# These are set by GKE automatically.
worker_id = os.getenv("TPU_WORKER_ID", "0")
slice_id = os.getenv("MEGASCALE_SLICE_ID", "0")

mounted_artifact_dir = Path(gcs_mount)
for s in gcs_artifact_subdir:
  mounted_artifact_dir = mounted_artifact_dir / s

# Configure XLA graph dump path before doing anything else.
date_string = datetime.now().strftime("%Y%m%d-%H%M")
host_name = f"{slice_id}-{worker_id}"
jobset_name = os.getenv("TORCHPRIME_JOBSET_NAME", date_string)
xla_dump_path = mounted_artifact_dir / jobset_name / "xla_dumps" / host_name
os.environ["XLA_FLAGS"] = " ".join(
  [
    os.getenv("XLA_FLAGS", ""),
    f"--xla_dump_to={xla_dump_path}/",
    "--xla_dump_hlo_as_proto",  # Save HLO protobuf files
    "--xla_dump_hlo_as_text",  # Save HLO text files
  ]
)
print(f"Dumping XLA compiler outputs to {xla_dump_path}", flush=True)

# Determine the profile dir
profile_dir = mounted_artifact_dir / jobset_name / "profile" / host_name
print(f"Profile output directory: {profile_dir}", flush=True)

output_dir = mounted_artifact_dir / jobset_name / "outputs" / host_name
print("Artifact output directory:", output_dir, flush=True)

# Exec into the training script.
args = (
  [sys.executable]
  + sys.argv[1:]
  + [
    f"profile_dir={str(profile_dir)}",
    f"output_dir={str(output_dir)}",
  ]
)
env = os.environ.copy()
process = subprocess.run(args, env=env)

# Upload result to database
upload_metrics = os.getenv("TORCHPRIME_UPLOAD_METRICS")

if upload_metrics.lower() == "true" and slice_id == "0" and worker_id == "0":
  try:
    click.echo(
      f"Primary worker ({host_name}) attempting to upload metrics for job '{jobset_name}'...",
    )
    upload_metrics_to_bq.collect_and_upload_benchmark_summary(
      process_returncode=process.returncode,
      jobset_name=jobset_name,
      mounted_artifact_path_str=str(mounted_artifact_dir),
    )
  except Exception as e:
    click.echo(f"Error uploading results to BigQuery: {e}", err=True)
sys.exit(process.returncode)
