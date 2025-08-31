from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from .config import AppConfig


def build_text_column(df: pd.DataFrame, fields: Sequence[str], target_col: str) -> pd.DataFrame:
    """
    Concatenates provided text fields into `target_col`. Skips empty values; enforces trailing period.
    """
    if not fields:
        if target_col not in df.columns:
            raise ValueError(f"No text_fields specified and '{target_col}' not present.")
        return df

    present = [c for c in fields if c in df.columns]
    if not present:
        raise ValueError(f"None of {fields} present in DataFrame columns {list(df.columns)}.")

    def _join_row(row: pd.Series) -> str:
        parts = [str(v).strip() for v in row if isinstance(v, str) and v.strip()]
        if not parts:
            return " "
        text = ". ".join(parts)
        return text if text.endswith(".") else text + "."

    df = df.copy()
    df[target_col] = df[present].fillna("").apply(_join_row, axis=1)
    return df


def create_tokenizer(cfg: AppConfig) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        token=cfg.model.hf_token,
        use_fast=True,
        add_prefix_space=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    return tok

def make_encoded_dataset(df: pd.DataFrame, cfg: AppConfig, tokenizer: AutoTokenizer) -> Dataset:
    """
    Returns a HF Dataset with tokenized inputs and a float32 `labels` matrix.
    """
    df = df.copy()
    df = build_text_column(df, cfg.data.text_fields, cfg.data.target_text_column)

    required = [cfg.data.id_column, cfg.data.target_text_column] + list(cfg.data.label_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df[required].copy()
    work[cfg.data.label_columns] = work[cfg.data.label_columns].astype(np.float32)

    ds = Dataset.from_pandas(work, preserve_index=False)
    ds = ds.rename_column(cfg.data.id_column, "id")

    def _tokenize(batch: Dict[str, List]) -> Dict[str, List]:
        enc = tokenizer(
            batch[cfg.data.target_text_column],
            truncation=True,
            max_length=cfg.data.max_length,
            padding=False,
        )
        labels_mat = np.stack([batch[col] for col in cfg.data.label_columns], axis=1).astype(np.float32)
        enc["labels"] = labels_mat
        return enc

    ds = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    return ds


def data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)
