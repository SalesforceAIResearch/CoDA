import functools
import getpass
import grp
import subprocess

import click


@functools.lru_cache
def is_sudoless_docker() -> bool:
  """Check if the current user can run Docker commands without sudo.

  This is done by checking if the user is in the 'docker' group.
  """
  user = getpass.getuser()
  groups_for_user = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
  is_sudoless_docker = "docker" in groups_for_user
  if is_sudoless_docker:
    click.echo(
      f"User {user} is in the 'docker' group. You can run Docker commands without sudo."
    )
  else:
    click.echo(
      f"User {user} is NOT in the 'docker' group. "
      "sudo is needed to run Docker commands."
    )
  return is_sudoless_docker


def run_docker(command: str | list[str]):
  if isinstance(command, str):
    command = command.split()
  command = ["docker"] + command
  if not is_sudoless_docker():
    command = ["sudo"] + command
  click.echo(" ".join(command))
  subprocess.run(
    command,
    shell=False,
    check=True,
  )
