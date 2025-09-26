import os
import subprocess

import hydra
from omegaconf import DictConfig


def set_env_variables(config: DictConfig) -> None:
  """Set environment variables from config env section"""
  if hasattr(config, "env"):
    for env_var in config.env:
      for key, value in env_var.items():
        os.environ[str(key)] = str(value)
  # set profile_dir as env
  if config.profile_dir:
    os.environ["PROFILE_LOGDIR"] = str(config.profile_dir)


def build_command(config: DictConfig) -> list:
  # Initialize base command
  cmd = ["python3", str(config.train_script.path)]

  # Replace env variables with actual values
  args = {}
  for k, v in config.train_script.args.items():
    args[k] = v

  for k, v in args.items():
    if v is None:
      # We may delete an argument by setting it to `null` on the CLI.
      continue
    if isinstance(v, bool):
      if v:
        cmd.append(f"--{k}")
    else:
      cmd.append(f"--{k}={v}")

  return cmd


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
  try:
    set_env_variables(config)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
      subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)

    cmd = build_command(config)
    subprocess.run(cmd, check=True)
  except Exception as e:
    print(f"Error: {str(e)}")
    raise


if __name__ == "__main__":
  main()
