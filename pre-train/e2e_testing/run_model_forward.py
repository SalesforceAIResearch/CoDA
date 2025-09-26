from torchprime import models
from torchprime.launcher import run_model


def main(model_id, run_type="torch_xla", batch_size=2, number_of_runs=5):
  print("Running with flags:")
  for k, v in locals().items():
    print(f"{k}: {v}")

  model_factory = models.registry.get(model_id)
  assert model_factory is not None, "Model with id {model_id} not registered"

  model = model_factory()
  print("Model init successful")

  if run_type == "torch_xla":
    times = run_model.run_model_xla(model, batch_size, number_of_runs)
  elif run_type == "torchax_eager":
    times = run_model.run_model_torchax(model, batch_size, number_of_runs, eager=True)
  elif run_type == "torchax":
    times = run_model.run_model_torchax(model, batch_size, number_of_runs, eager=False)
  else:
    raise AssertionError(
      f"run_type: {run_type} unknown. Please pass in torch_xla, torchax or torchax_eager"
    )

  print("Model run times:")
  for i, t in enumerate(times):
    print(f"Iteration {i}: {t}s")

  print("Model run successful")


if __name__ == "__main__":
  import fire

  fire.Fire(main)
