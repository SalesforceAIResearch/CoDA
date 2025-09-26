from datetime import timedelta

from torchprime.metrics.metrics import Metrics


def test_print_metrics():
  """Test the string representation of the Metrics class."""

  # All fields are set.
  metrics = Metrics(
    model="llama-3-8b",
    train_runtime=timedelta(seconds=3600.0),
    step_execution_time=timedelta(seconds=63.1),
    mfu=0.30717649966,
  )
  expected_str = """
model                = llama-3-8b
train_runtime        = 1:00:00
step_execution_time  = 0:01:03.100000
mfu                  = 0.30717649966
""".lstrip()
  assert str(metrics) == expected_str

  # Missing step_execution_time.
  metrics = Metrics(
    model="llama-3.1-70b",
    train_runtime=timedelta(seconds=3600.0),
    step_execution_time=None,
    mfu=None,
  )
  expected_str = """
model                = llama-3.1-70b
train_runtime        = 1:00:00
step_execution_time  = N/A
mfu                  = N/A
""".lstrip()
  assert str(metrics) == expected_str


def test_metrics_to_json():
  metrics = Metrics(
    model="llama-3.1-70b",
    train_runtime=timedelta(seconds=1),
    step_execution_time=timedelta(seconds=1),
    mfu=1.0,
  )
  json_str = metrics.to_json()  # type: ignore
  metrics_back = Metrics.from_json(json_str)  # type: ignore
  assert metrics == metrics_back
