from .model_original import TransAct
from .transact_config import TransActConfig


def get_model():
  action_vocab = list(range(0, 20))
  full_seq_len = 100
  action_emb_dim = 32
  item_emb_dim = 32
  time_window_ms = 1000 * 60 * 60 * 1  # 1 hr
  latest_n_emb = 10

  transact_config = TransActConfig(
    action_vocab=action_vocab,
    seq_len=full_seq_len,
    action_emb_dim=action_emb_dim,
    item_emb_dim=item_emb_dim,
    time_window_ms=time_window_ms,
    latest_n_emb=latest_n_emb,
  )

  return TransAct(transact_config)
