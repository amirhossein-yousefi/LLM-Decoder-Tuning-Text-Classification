import numpy as np
from transformers import EvalPrediction
from llm_cls.metrics import MultiLabelMetrics

def test_multilabel_metrics_basic():
    logits = np.array([[2.0, -2.0], [-2.0, 2.0]])  # -> probs ~ [0.88, 0.12], [0.12, 0.88]
    labels = np.array([[1, 0], [0, 1]], dtype=float)
    m = MultiLabelMetrics(num_labels=2, threshold=0.5)
    out = m(EvalPrediction(predictions=logits, label_ids=labels))
    assert 0.9 <= out["f1_micro"] <= 1.0
    assert 0.9 <= out["f1_macro"] <= 1.0
