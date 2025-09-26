"""
Doctor checks for essential programs needed to launch distributed training.
"""

import getpass
import grp
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypeVar

import click

ConfigT = TypeVar("ConfigT", bound="Config")  # noqa: F821


class CheckFailedError(Exception):
  pass


def check_docker():
  """Check that docker is installed."""
  try:
    subprocess.run(["docker", "help"], check=True, capture_output=True)
  except FileNotFoundError:
    raise CheckFailedError("docker not found. Please install docker first.") from None


def check_gcr_io():
  """Check that docker config contains gcr.io credential helper."""
  try:
    docker_config = json.loads(
      Path(os.path.expanduser("~/.docker/config.json")).read_text()
    )
    cred_helpers = docker_config["credHelpers"]
    _gcr_io = cred_helpers["gcr.io"]
  except (FileNotFoundError, KeyError, json.JSONDecodeError):
    user = getpass.getuser()
    groups_for_user = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
    setup_cmd = "gcloud auth configure-docker"
    if "docker" not in groups_for_user:
      setup_cmd = f"sudo {setup_cmd}"
    raise CheckFailedError(
      f"""
Did not find a handler for `gcr.io` in docker credential helpers.

TorchPrime uploads docker containers to the `gcr.io` docker registry, which
requires valid credentials.

To setup the credentials, please run:

  {setup_cmd}

""".lstrip()
    ) from None


def check_docker_access():
  """Check that the gcloud account can access the gcr.io artifact registry."""
  try:
    subprocess.run(
      ["gcloud", "artifacts", "repositories", "describe", "gcr.io", "--location=us"],
      check=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    account = subprocess.run(
      ["gcloud", "config", "get-value", "account"], capture_output=True, text=True
    ).stdout.strip()
    raise CheckFailedError(
      f"""The current gcloud account `{account}` cannot access the gcr.io registry.
The account may not have the required permissions. If it's a service account, the
VM may not have the correct scopes.

The easiest way to resolve this is to login with your own account:

  gcloud auth login

"""
    ) from e


def check_gcloud_auth_login():
  """Check that gcloud is logged in."""
  try:
    subprocess.run(
      ["gcloud", "auth", "print-access-token"], check=True, capture_output=True
    )
  except subprocess.CalledProcessError as e:
    raise CheckFailedError(
      f"gcloud auth print-access-token failed: {e.stderr.decode()}"
    ) from e


def check_kubectl():
  """Check that kubectl is installed."""
  try:
    subprocess.run(["kubectl", "help"], check=True, capture_output=True)
  except FileNotFoundError:
    raise CheckFailedError(
      f"""kubectl not found.

{get_kubectl_install_instructions()}"""
    ) from None


def check_gke_gcloud_auth_plugin():
  """Check that gke-gcloud-auth-plugin is installed."""
  if is_gcloud_plugin_installed("gke-gcloud-auth-plugin"):
    return
  raise CheckFailedError(
    f"""The `gke-gcloud-auth-plugin` gcloud component is not installed

{get_gke_gcloud_auth_plugin_instructions()}"""
  )


def check_gke_cluster_exist(config: ConfigT | None = None):
  """Check that the GKE cluster exists."""
  try:
    result = subprocess.run(
      [
        "gcloud",
        "container",
        "clusters",
        "describe",
        f"{config.cluster}",
        f"--project={config.project}",
        f"--zone={config.zone}",
      ],
      check=True,
      capture_output=True,
      text=True,
    )
  except subprocess.CalledProcessError as e:
    print(
      f"Error running gcloud command: {e.stderr} Please check the cluster name and project"
    )
    # get existing clusters in the project
    try:
      result = subprocess.run(
        ["gcloud", "container", "clusters", "list", f"--project={config.project}"],
        check=True,
        capture_output=True,
        text=True,
      )
      print(f"Available clusters in the project: \n{result.stdout}")
    except subprocess.CalledProcessError as e:
      print(f"Error running gcloud command: {e.stderr} Unable to get existing clusters")
    raise CheckFailedError(
      f"""The GKE cluster `{config.cluster}` does not exist in project `{config.project}` in zone `{config.zone}`"""
    ) from e


def check_all(config: ConfigT | None = None):
  click.echo("Checking environment...")
  check_list = [
    check_docker,
    check_gcloud_auth_login,
    check_gcr_io,
    check_docker_access,
    check_kubectl,
    check_gke_gcloud_auth_plugin,
  ]
  if config:
    check_list.append(check_gke_cluster_exist)
  for check in check_list:
    assert check.__doc__ is not None
    click.echo(check.__doc__ + "..", nl=False)
    try:
      try:
        check(config)
      except TypeError:
        check()
    except CheckFailedError as e:
      click.echo()
      click.echo()
      click.echo(f"❌ Error during {check.__name__} ❌")
      click.echo(e)
      sys.exit(-1)
    click.echo(" ✅")
  click.echo(
    "🎉 All checks passed. You should be ready to launch distributed training. 🎉"
  )


def get_kubectl_install_instructions():
  # If gcloud is installed via `apt`, then we should do the same for `kubectl`.
  if is_package_installed("google-cloud-cli"):
    return """
Since `gcloud` is installed with `apt`, please install `kubectl` with:

  sudo apt install kubectl

""".lstrip()

  # Otherwise, point users to the GKE docs.
  return "Please visit \
https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl"


def get_gke_gcloud_auth_plugin_instructions():
  # If gcloud is installed via `apt`, then we should do the same for this plugin.
  if is_package_installed("google-cloud-cli"):
    return """
Since `gcloud` is installed with `apt`, please install `gke-gcloud-auth-plugin` with:

  sudo apt install google-cloud-sdk-gke-gcloud-auth-plugin

""".lstrip()

  # Otherwise, point users to the docs.
  return "Please visit \
https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin"


def is_package_installed(package_name):
  try:
    # Run the dpkg-query command to check for the package
    subprocess.run(
      ["dpkg-query", "-W", "-f='${Status}'", package_name],
      check=True,
      capture_output=True,
    )
    return True
  except subprocess.CalledProcessError:
    return False


def is_gcloud_plugin_installed(plugin_name):
  try:
    # Run `gcloud components list` to get installed components
    result = subprocess.run(
      ["gcloud", "components", "list", "--format=json", f"--filter={plugin_name}"],
      check=True,
      capture_output=True,
      text=True,
    )
    # Parse the output and look for the plugin
    components = json.loads(result.stdout)
    for component in components:
      if component.get("id") == plugin_name:
        state = component.get("state")
        if state == "Installed" or (
          isinstance(state, dict) and state.get("name") == "Installed"
        ):
          return True
    return False
  except subprocess.CalledProcessError as e:
    print(f"Error running gcloud command: {e.stderr}")
    return False
  except json.JSONDecodeError:
    print("Error parsing JSON output from gcloud.")
    return False


if __name__ == "__main__":
  check_all()
