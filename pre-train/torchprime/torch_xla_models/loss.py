import torch
from torch.nn import CrossEntropyLoss


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int):
  """
  Computes cross entropy loss of `logits` against the ground truth `labels` during
  next token prediction.

  Useful as the loss function of a LLM in pretraining or supervised finetuning.
  """
  # Shift so that tokens < n predict n
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  # Flatten the tokens
  loss_fct = CrossEntropyLoss()
  shift_logits = shift_logits.view(-1, vocab_size)
  shift_labels = shift_labels.view(-1)
  shift_labels = shift_labels.to(shift_logits.device)
  return loss_fct(shift_logits, shift_labels)
