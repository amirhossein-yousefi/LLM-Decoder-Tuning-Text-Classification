from __future__ import annotations

from typing import Dict

import torch
from torchmetrics.classification import MultilabelF1Score
from transformers import EvalPrediction


class MultiLabelMetrics:
    """
    Computes micro and macro F1 for multi-label classification.
    Applies sigmoid to logits, thresholds at 0.5 by default.
    """
    def __init__(self, num_labels: int, threshold: float = 0.5) -> None:
        self.micro = MultilabelF1Score(num_labels=num_labels, threshold=threshold, average="micro")
        self.macro = MultilabelF1Score(num_labels=num_labels, threshold=threshold, average="macro")

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = torch.sigmoid(torch.tensor(logits))
        labels = torch.tensor(p.label_ids)
        f1_micro = self.micro(probs, labels).item()
        f1_macro = self.macro(probs, labels).item()
        return {"f1_micro": f1_micro, "f1_macro": f1_macro}
