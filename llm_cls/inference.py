from __future__ import annotations

import json
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import AppConfig
from .data import build_text_column


@torch.no_grad()
def predict_csv(input_csv: str, cfg: AppConfig, output_jsonl: str) -> None:
    """
    Loads a fine-tuned checkpoint from exp.output_root / model / dataset / run_0 (by default),
    runs inference on input_csv, and writes JSONL with id + probabilities.
    """
    # pick run_0 as default; change if needed
    checkpoint_dir = cfg.output_dir(run_number=0)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    df = pd.read_csv(input_csv)
    df = build_text_column(df, cfg.data.text_fields, cfg.data.target_text_column)
    texts = df[cfg.data.target_text_column].tolist()
    ids = df[cfg.data.id_column].tolist() if cfg.data.id_column in df.columns else list(range(len(df)))

    with open(output_jsonl, "w") as f:
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, truncation=True, max_length=cfg.data.max_length, return_tensors="pt", padding=True)
            logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, p in enumerate(probs):
                record = {
                    "id": ids[i+j],
                    "probs": p.tolist(),
                    "labels": list(cfg.data.label_columns)
                }
                f.write(json.dumps(record) + "\n")
