import os

import click
from benchmark_db_writer import bq_writer_utils
from benchmark_db_writer.schema.workload_benchmark_v2 import (
  workload_benchmark_v2_schema,
)

from torchprime.launcher import benchmark_db_util


def _write_summary_to_bq_client(
  summary_data: dict,
  bq_project: str,
  bq_dataset: str,
  bq_table: str,
):
  """
  Uploads the prepared benchmark summary to BigQuery.
  This is the low-level client interaction.
  Args:
      summary_data: A dictionary containing the benchmark summary,
                    typically from prepare_benchmark_summary.
      bq_project: The BigQuery project ID.
      bq_dataset: The BigQuery dataset ID.
      bq_table: The BigQuery table ID.
  """
  click.echo("Attempting to upload benchmark results to BigQuery...")
  client = bq_writer_utils.create_bq_writer_object(
    project=bq_project,
    dataset=bq_dataset,
    table=bq_table,
    dataclass_type=workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
  )
  summary_obj = workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema(**summary_data)
  client.write([summary_obj])
  print(
    f"Benchmark results for run_id '{summary_obj.run_id}' successfully uploaded to BigQuery.",
    flush=True,
  )


def collect_and_upload_benchmark_summary(
  process_returncode: int,
  jobset_name: str,
  mounted_artifact_path_str: str,
):
  """
  Gathers necessary information from environment variables and artifacts,
  prepares the benchmark summary, and uploads it to BigQuery.
  """
  # Gather cluster config from env for DB upload and summary preparation
  _gcs_artifact_dir = os.environ["TORCHPRIME_ARTIFACT_DIR"]
  _cluster = os.environ["TORCHPRIME_CLUSTER"]
  _num_slices = os.environ["TORCHPRIME_NUM_SLICES"]
  _bq_project = os.environ["TORCHPRIME_BQ_PROJECT"]
  _bq_dataset = os.environ["TORCHPRIME_BQ_DATASET"]
  _bq_table = os.environ["TORCHPRIME_BQ_TABLE"]
  _tpu_type = os.environ["TORCHPRIME_TPU_TYPE"]
  _comments = os.environ["TORCHPRIME_COMMENTS"]
  _docker_url = os.environ["TORCHPRIME_DOCKER_URL"]
  _update_person_ldap = os.environ["TORCHPRIME_USER"]
  _configs_xla_flags = os.environ["XLA_FLAGS"]

  metrics = benchmark_db_util.get_metrics(mounted_artifact_path_str, jobset_name)

  summary_dict = benchmark_db_util.prepare_benchmark_summary(
    process_returncode=process_returncode,
    tpu_type=_tpu_type,
    jobset_name=jobset_name,
    update_person_ldap=_update_person_ldap,
    cluster_name=_cluster,
    hardware_num_slices=_num_slices,
    configs_xla_flags=_configs_xla_flags,
    logs_comments=_comments,
    logs_other=_docker_url,
    gcs_metrics_bucket=os.path.join(_gcs_artifact_dir, jobset_name),
    **metrics,
    # TODO Add more relevant schemas. Check workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema for existing schema
  )

  _write_summary_to_bq_client(
    summary_data=summary_dict,
    bq_project=_bq_project,
    bq_dataset=_bq_dataset,
    bq_table=_bq_table,
  )
