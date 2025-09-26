import torch
from functorch.compile import aot_function, aot_module, make_boxed_func  # type:ignore

from torchprime.torch_xla_models.remat_all import remat_all_partition_fn


def test_remat_all():
  """Test that remat_all_partition_fn reruns all the forward operations during the backward."""

  def fn(x, y):
    a = x @ y
    b = torch.sin(a)
    c = torch.exp(b)
    return c

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  x = torch.randn((1, 10), requires_grad=True)
  y = torch.randn((10, 1), requires_grad=True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=remat_all_partition_fn,
  )
  out = fn_compiled(x, y)
  out.backward()

  fw_graph = get_fwd()
  bw_graph = get_bwd()

  # Forward graph does these ops
  assert _count_function_calls(fw_graph, "mm") == 1
  assert _count_function_calls(fw_graph, "sin") == 1
  assert _count_function_calls(fw_graph, "exp") == 1

  # Backward graph does these ops
  assert _count_function_calls(bw_graph, "mm") == 3  # 1 forward + 2 backward
  assert _count_function_calls(bw_graph, "sin") == 1  # 1 forward
  # 1 backward (gradient of exp). There is no forward exp because the output
  # of the last node `c` in the `fn` is not required to compute gradients for nodes
  # within the `fn`.
  assert _count_function_calls(bw_graph, "exp") == 1
  assert _count_function_calls(bw_graph, "cos") == 1  # 1 backward (gradient of sin)

  # Test that forward outputs 2 residuals that is the two inputs. In other word,
  # nothing else is saved.
  c, residual_1, residual_2 = fw_graph(x, y)
  torch.testing.assert_close(residual_1, x)
  torch.testing.assert_close(residual_2, y)

  # We can plug these into the backward to compute the gradients functionally.
  c_grad = torch.tensor(1.0)
  x_grad, y_grad = bw_graph(residual_1, residual_2, c_grad)
  torch.testing.assert_close(x_grad, x.grad)
  torch.testing.assert_close(y_grad, y.grad)


def test_remat_all_view():
  """Test rematerialization of view operations."""

  def fn(x, y):
    a = x.view(2, 5)
    b = a.view(5, 2)
    c = b.view(1, 10)

    d = y.view(2, 5)
    e = d.view(5, 2)
    f = e.view(10, 1)

    g = c @ f
    h = torch.sin(g)
    i = h.view(1)
    j = torch.exp(i)
    return j

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  x = torch.randn((1, 10), requires_grad=True)
  y = torch.randn((10, 1), requires_grad=True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=remat_all_partition_fn,
  )
  out = fn_compiled(x, y)
  out.backward()

  fw_graph = get_fwd()
  bw_graph = get_bwd()

  # Seven view in total counted from `fn`.
  assert _count_function_calls(fw_graph, "view") == 7

  # All forward views should be rematerialized in the backward, plus each of their gradients.
  assert _count_function_calls(bw_graph, "view") == 7 * 2


def test_remat_all_reductions():
  """Test that even reductions can be rematerialized.

  What is a "reduction"? Basically going from large input to small output, c.f.
  https://github.com/pytorch/pytorch/blob/38e81a53324146d445a81eb8f80bccebe623eb35/torch/_functorch/partitioners.py#L1101
  """

  def fn(x, y):
    # Reduce x from 64x64 to 2x2.
    x = x.view(2, 2, 32, 32).sum(dim=-1).mean(dim=-1)
    x = torch.sin(x)
    # Reduce y from 64x64 to 2x2.
    y = y.view(2, 2, 32, 32).sum(dim=-1).mean(dim=-1)
    y = torch.sin(y)
    # Multiply them into a scalar.
    return x.reshape(1, 4) @ y.reshape(4, 1)

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  x = torch.randn((64, 64), requires_grad=True)
  y = torch.randn((64, 64), requires_grad=True)
  fn_compiled = aot_function(
    fn,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=remat_all_partition_fn,
  )
  out = fn_compiled(x, y)
  out.backward()

  fw_graph = get_fwd()
  bw_graph = get_bwd()

  # Sum and mean are all "reductions" and they should all still be recomputed.
  assert _count_function_calls(fw_graph, "sum") == 2
  assert _count_function_calls(fw_graph, "mean") == 2
  assert _count_function_calls(bw_graph, "sum") == 2
  assert _count_function_calls(bw_graph, "mean") == 2


def test_remat_all_llama3():
  """Stress test that we can remat the decoder layer of a Llama 3 model."""
  from .test_llama import get_llama_3_8b

  fixture = get_llama_3_8b()
  decoder_layer = fixture.model.model.layers[0]
  input_size = 32
  seq_len = input_size // 2
  in_embedding = torch.randn((2, seq_len, fixture.model.config.hidden_size))
  position_ids = torch.arange(in_embedding.shape[1]).unsqueeze(0)
  position_embeddings = fixture.model.model.rotary_emb(in_embedding, position_ids)

  fw_compiler, get_fwd = _make_get_graph_compiler()
  bw_compiler, get_bwd = _make_get_graph_compiler()

  decoder_layer_compiled = aot_module(
    decoder_layer,
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=remat_all_partition_fn,
  )

  # causal_mask dimentsions:
  # bz 1 seq_len seq_len
  causal_mask = torch.ones((2, 1, seq_len, seq_len))

  out_embedding = decoder_layer_compiled(
    in_embedding,
    attention_mask=causal_mask,
    position_ids=position_ids,
    position_embeddings=position_embeddings,
  )
  out_embedding.sum().backward()

  fw_graph = get_fwd()
  bw_graph = get_bwd()

  num_params_buffers = len(list(decoder_layer.parameters())) + len(
    list(decoder_layer.buffers())
  )

  compiled_out_embedding, *residuals = fw_graph(
    *[
      *decoder_layer.parameters(),
      *decoder_layer.buffers(),
    ],
    in_embedding,
    causal_mask,
    position_ids,
    *position_embeddings,
  )
  torch.testing.assert_close(out_embedding, compiled_out_embedding)

  # Verify that there are no intermediate activations in the residuals. In other words,
  # the residuals consist of weights/buffers plus the 4 inputs:
  # (in_embedding, position_ids, position_embeddings[0], position_embeddings[1]).
  assert len(residuals) == num_params_buffers + 4

  # We can also run the backward graph.
  _ = bw_graph(*residuals, torch.ones_like(out_embedding))


def _make_get_graph_compiler():
  """Creates a compiler that records the graph, and a getter function to retrieve them."""
  graph: list[torch.fx.GraphModule | None] = [None]

  def forward_comp(fx_module: torch.fx.GraphModule, _):
    assert graph[0] is None
    graph[0] = fx_module
    return make_boxed_func(fx_module)

  def get_graph():
    g = graph[0]
    assert g is not None
    return g

  return forward_comp, get_graph


def _count_function_calls(gm: torch.fx.GraphModule, pattern: str) -> int:
  count = 0
  for node in gm.graph.nodes:
    if node.op == "call_function":
      target = node.target
      if callable(target):
        target = target.__qualname__
      if pattern in target:
        count += 1
  return count
