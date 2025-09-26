import numpy as np
from typing import Dict, Sequence, Tuple, Union
import torch.nn.functional as F

def f1_score(preds, labels):
    f1 = []
    for pred, label in zip(preds, labels):
        f1.append(len(np.intersect1d(pred, label))/len(pred))
    return np.mean(f1)

def compute_nll(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, labels = eval_preds
    f1 = f1_score(preds, labels)
    return {"eval_f1": f1}

def gsm_eval(preds, labels):
    score_dict = {"acc": []}

    for pred, label in zip(preds, labels):
        pred = pred.split('####')[-1].strip().replace(',', '')
        label = label.split('####')[-1].strip().replace(',', '')

        score_dict["acc"].append(pred==label)

    return {k: float(np.mean(v)) for k, v in score_dict.items()}