import pytest

from torchprime.metrics.mfu import compute_mfu


def test_llama2_7b_v5p_mfu():
  config = {
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=1024,
    sequence_length=4096,
    step_duration=2.801027417,
    tpu_name="foobar-v5p-512",
  )
  assert result.mfu == pytest.approx(0.5872846948, rel=0.01, abs=0.005)


def test_llama2_70b_v5e_mfu():
  config = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=512,
    sequence_length=2048,
    step_duration=14.97010803,
    tpu_name="v5e-256-stuff",
  )
  assert result.mfu == pytest.approx(0.5950, rel=0.01, abs=0.005)


def test_llama3_70b_v6e_mfu():
  config = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 128256,
  }
  result = compute_mfu(
    config,
    batch_size=128,
    sequence_length=8192,
    step_duration=16.992,
    tpu_name="abc-v6e-128-stuff",
  )
  assert result.mfu == pytest.approx(0.2562, rel=0.01, abs=0.005)


def test_llama3_405b_v6e_2_pods_mfu():
  config = {
    "hidden_size": 16384,
    "intermediate_size": 53248,
    "num_attention_heads": 128,
    "num_hidden_layers": 126,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 128256,
  }
  result = compute_mfu(
    config,
    batch_size=512,
    sequence_length=8192,
    step_duration=84.168,
    tpu_name="v6e-256",
    num_slices=2,
  )
  assert result.mfu == pytest.approx(0.2792, rel=0.01, abs=0.005)


def test_mixtral_8x7b_v5p_mfu():
  config = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=4608,
    sequence_length=4096,
    step_duration=51.876,
    tpu_name="abc-v5p-256-stuff",
  )
  assert result.mfu == pytest.approx(0.51, rel=0.01, abs=0.005)


def test_mixtral_8x7b_v6e_mfu():
  config = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=1024,
    sequence_length=4096,
    step_duration=5.04,
    tpu_name="v6e-256",
  )
  assert result.mfu == pytest.approx(0.2962, rel=0.01, abs=0.005)


def test_mixtral_8x7b_v6e_2_pods_mfu():
  config = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=2048,
    sequence_length=4096,
    step_duration=5.36,
    tpu_name="v6e-256",
    num_slices=2,
  )
  assert result.mfu == pytest.approx(0.281, rel=0.01, abs=0.005)
