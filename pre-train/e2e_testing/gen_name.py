#!/usr/bin/env python3
import random
import sys


def gen_name(name: str | None):
  # 8 random characters from a-z and 0-9
  random_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
  name = random_id if name is None else f"{name}-{random_id}"
  return name


if __name__ == "__main__":
  if len(sys.argv) > 2:
    print("Usage: gen_name.py [name]")
    sys.exit(1)
  print(gen_name(sys.argv[1] if len(sys.argv) > 1 else None), flush=True, end="")
