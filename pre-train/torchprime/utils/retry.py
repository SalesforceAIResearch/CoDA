import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(fn: Callable[[], T], retry_count: int = 3, retry_delay: float = 10) -> T:
  logger = logging.getLogger(__name__)
  for attempt in range(retry_count):
    try:
      return fn()
    except Exception as e:
      if attempt < retry_count - 1:
        logger.warning(
          f"Error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retry_count})"
        )
        time.sleep(retry_delay + random.random() * retry_delay)
      else:
        logger.error(
          f"Failed to load tokenizer after {retry_count} attempts due to ReadTimeoutError: {e}"
        )
        raise  # Re-raise the exception after exhausting retries
  raise RuntimeError("Unreachable")
