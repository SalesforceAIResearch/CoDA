#!/usr/bin/env python3

import datetime
import getpass
import os
import random
import re
import string
import subprocess
from pathlib import Path

import tomli

from torchprime.launcher.util import run_docker


def buildpush(
  torchprime_project_id="",
  torchprime_docker_url: str | None = None,
  base_docker_url: str | None = None,
  push_docker=True,
  placeholder_url=None,
  *,
  build_arg: list[str] | None = None,
) -> str:
  # Determine the path of this script and its directory
  script_path = os.path.realpath(__file__)
  script_dir = Path(os.path.dirname(script_path))
  context_dir = script_dir.parent.parent.relative_to(os.getcwd())
  docker_file = (script_dir / "Dockerfile").relative_to(os.getcwd())
  user = getpass.getuser()

  # Generate date/time string and 4 random lowercase letters
  datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  random_chars = "".join(random.choices(string.ascii_lowercase, k=4))

  # Determine Docker tag
  default_tag = f"{datetime_str}-{random_chars}"
  docker_tag = default_tag

  # Determine Docker URL
  if placeholder_url:
    docker_url = placeholder_url
  else:
    default_url = f"gcr.io/{torchprime_project_id}/torchprime-{user}:{docker_tag}"
    docker_url = torchprime_docker_url if torchprime_docker_url else default_url
    # docker_url doesn't accept `//` and uppercase.
    docker_url = re.sub(r"/+", "/", docker_url).lower()

  print()
  if push_docker:
    print(f"Will build a docker image and upload to: {docker_url}")
  else:
    print(f"Create docker image: {docker_url} and tag locally")
  print()

  build_cmd = "build"
  if build_arg:
    for _arg in build_arg:
      build_cmd += f" --build-arg {_arg}"
  build_cmd += (
    f" --network=host --progress=auto -t {docker_tag} {context_dir} -f {docker_file}"
  )

  # Provide the base image
  pyproject_file = context_dir / "pyproject.toml"
  torch_xla_version = tomli.loads(pyproject_file.read_text())["tool"]["torchprime"][
    "torch_xla_version"
  ]
  if base_docker_url:
    # Use the provided base image
    base_image = base_docker_url
  else:
    # Use torch_xla Python 3.10 as the base image
    base_image = f"us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_{torch_xla_version}"
  build_cmd += f" --build-arg BASE_IMAGE={base_image}"

  # Build, tag, and push Docker image
  try:
    run_docker(build_cmd)
    run_docker(
      f"tag {docker_tag} {docker_url}",
    )
    if push_docker:
      run_docker(f"push {docker_url}")
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")
    exit(e.returncode)

  return docker_url


if __name__ == "__main__":
  # Read environment variables or use defaults
  torchprime_project_id = os.getenv("TORCHPRIME_PROJECT_ID", "tpu-pytorch")
  torchprime_docker_url = os.getenv("TORCHPRIME_DOCKER_URL", None)
  push_docker_str = os.getenv("TORCHPRIME_PUSH_DOCKER", "true")
  push_docker = push_docker_str.lower() in ("true", "1", "yes", "y")
  buildpush(
    torchprime_project_id=torchprime_project_id,
    torchprime_docker_url=torchprime_docker_url,
    push_docker=push_docker,
  )
