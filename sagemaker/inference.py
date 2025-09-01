
import os
import json
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKENIZER = None
_MODEL = None
_SIGMOID = torch.nn.Sigmoid()

def _load(model_dir: str):
    global _TOKENIZER, _MODEL
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    try:
        _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir, config=cfg, trust_remote_code=True)
    except Exception as e:
        if not _HAS_PEFT:
            raise e
        # Fallback: if the artifact contains only adapters, merge them into base (name stored in config)
        base_id = getattr(cfg, "_name_or_path", None)
        if not base_id:
            raise e
        base = AutoModelForSequenceClassification.from_pretrained(base_id, trust_remote_code=True)
        _MODEL = PeftModel.from_pretrained(base, model_dir)
        try:
            _MODEL = _MODEL.merge_and_unload()
        except Exception:
            pass
    _MODEL.to(_DEVICE).eval()
    return {"id2label": getattr(cfg, "id2label", None)}

# ---- SageMaker-required functions ----
def model_fn(model_dir: str):
    """Load model and tokenizer from model_dir."""
    return _load(model_dir)

def input_fn(input_data, content_type="application/json"):
    """Accept {"text": "..."} or {"text": [...]} or a raw JSON list of strings."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(input_data)
    texts = data.get("text", data) if isinstance(data, dict) else data
    if isinstance(texts, str):
        texts = [texts]
    return texts

@torch.inference_mode()
def predict_fn(data: List[str], model_artifacts: Dict[str, Any]):
    """Batch predict with multi-label/softmax logic."""
    max_length = int(os.environ.get("MAX_LENGTH", "512"))
    bs = int(os.environ.get("BATCH_SIZE", "16"))
    multi_label = os.environ.get("MULTILABEL", "true").lower() in {"1", "true", "yes"}
    threshold = float(os.environ.get("MULTILABEL_THRESHOLD", "0.5"))
    id2label = model_artifacts.get("id2label", None)

    results = []
    for i in range(0, len(data), bs):
        batch = data[i : i + bs]
        enc = _TOKENIZER(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(_DEVICE)
        logits = _MODEL(**enc).logits
        if id2label is None:
            id2label = {i: f"LABEL_{i}" for i in range(logits.shape[-1])}

        if multi_label:
            probs = _SIGMOID(logits).cpu()
            for row in probs:
                row = row.tolist()
                outs = [{"label": id2label.get(i, f"LABEL_{i}"), "score": float(s)} for i, s in enumerate(row) if s >= threshold]
                if not outs:
                    j = int(torch.tensor(row).argmax().item())
                    outs = [{"label": id2label.get(j, f"LABEL_{j}"), "score": float(row[j])}]
                results.append(outs)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().tolist()
            for row in probs:
                j = int(torch.tensor(row).argmax().item())
                results.append([{"label": id2label.get(j, f"LABEL_{j}"), "score": float(row[j])}])

    return results

def output_fn(prediction, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction), accept
