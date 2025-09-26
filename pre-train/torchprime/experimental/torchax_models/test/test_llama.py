import unittest

import torch

from torchprime.experimental.torchax_models.llama import model as llama_model


class LlamaTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.max_seq_len = 512  # 8192
    self.vocab_size = 128  # 32000
    self.n_layer = 1
    self.n_heads = 4
    self.dim = 8
    self.block_size = 16  # 2048
    with torch.no_grad():
      self.input = torch.randint(0, self.vocab_size, (1, self.max_seq_len))
      self.model_args = llama_model.ModelArgs(
        block_size=self.block_size,
        vocab_size=self.vocab_size,
        n_layer=self.n_layer,
        n_heads=self.n_heads,
        dim=self.dim,
        max_seq_len=self.max_seq_len,
      )
      self.freqs_cis = llama_model.precompute_freqs_cis(
        self.model_args.dim // self.model_args.n_heads,
        self.model_args.max_seq_len,
        self.model_args.rope_theta,
        self.model_args.use_scaled_rope,
      ).to(torch.bfloat16)
      self.m = llama_model.Transformer(self.model_args)
      self.m.to(torch.bfloat16)
      self.native_output = self.m(
        self.input, 0, freqs_cis=self.freqs_cis, mask=torch.ones_like(self.input)
      )

  def test_forward_torchax_against_native(self):
    import torchax
    import torchax.config

    torchax.enable_accuracy_mode()
    env = torchax.default_env()
    # TODO(zpcore): uncomment the following once torchax support config input
    # torch_xla2_config = torchax.config.Configuration()
    # torch_xla2_config.use_tpu_flash_attention = True
    # env = torchax.default_env(torch_xla2_config)
    with env:
      input_tensor = self.input.to("jax")
      freqs_cis = self.freqs_cis.to("jax")
      self.m.to("jax")
      output = (
        self.m(input_tensor, 0, freqs_cis=freqs_cis, mask=torch.ones_like(self.input))
        .to("cpu")
        .to(torch.bfloat16)
      )
    self.assertTrue(
      torch.allclose(output.to("cpu"), self.native_output, atol=1e-1),
      "pytorch native and torchax are not equal",
    )


"""
# (zpcore): This test is commented out for testing purposes
  def test_forward_torchxla_against_native(self):
    import torch_xla
    torch_xla.manual_seed(42)
    device = torch_xla.core.xla_model.xla_device()
    input_tensor = self.input.to(device)
    freqs_cis = self.freqs_cis.to(device)
    m = self.m.to(device)
    output = m(
        input_tensor, 0, freqs_cis=freqs_cis, mask=torch.ones_like(self.input))
    self.assertTrue(
        torch.allclose(output.to("cpu"), self.native_output, atol=1),
        f"pytorch native and torch_xla are not equal")
"""

if __name__ == "__main__":
  unittest.main()
