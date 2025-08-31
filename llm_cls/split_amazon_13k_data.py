from __future__ import annotations
import os, json, math, argparse, ast
from pathlib import Path
from typing import Sequence, List, Dict, Optional, Tuple
import re
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

def sanitize_label_for_column(name: str) -> str:
    """
    Make a safe column name while keeping it recognizable.
    - Keep letters/digits/underscore.
    - Replace other chars with '_'.
    - Collapse multiple '_' and strip edges.
    """
    s = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "label"

def load_label_vocab() -> List[str]:
    """
    Returns index->label_name list from the dataset's 'labels' config.
    """
    vocab_ds = load_dataset("pietrolesci/amazoncat-13k", "labels", split="train")
    return list(vocab_ds["labels"])

def compute_label_frequencies(ds_split, num_labels: int) -> np.ndarray:
    """
    Count how often each label index appears in ds_split['target_ind'].
    Memory-friendly single pass.
    """
    counts = np.zeros(num_labels, dtype=np.int64)
    for ex in ds_split:
        inds = ex.get("target_ind", [])
        if inds:
            counts[np.asarray(inds, dtype=np.int64)] += 1
    return counts

def pick_topk_indices(counts: np.ndarray, k: int) -> List[int]:
    k = int(min(k, len(counts)))
    if k <= 0:
        return []
    # argsort descending
    return np.argsort(-counts)[:k].tolist()

def add_topk_indicator_columns(
    split_ds,
    top_indices: List[int],
    top_colnames: List[str],
    filter_rows: bool,
    batch_size: int = 1000,
):
    """
    Adds one 0/1 column per top label (columns named by 'top_colnames').
    Optionally filters rows that don't have any of those labels.
    """
    idx_and_cols = list(zip(top_indices, top_colnames))
    def _batch_fn(batch: Dict[str, List]):
        bsz = len(batch["target_ind"])
        out = {col: [0] * bsz for _, col in idx_and_cols}
        any_top = [0] * bsz
        for i, inds in enumerate(batch["target_ind"]):
            s = set(inds)
            hit = 0
            for idx, col in idx_and_cols:
                if idx in s:
                    out[col][i] = 1
                    hit = 1
            any_top[i] = hit
        out["_has_any_top_"] = any_top
        return out

    ds_plus = split_ds.map(_batch_fn, batched=True, batch_size=batch_size, desc="Adding top-k label indicators")

    if filter_rows:
        ds_plus = ds_plus.filter(lambda ex: ex["_has_any_top_"] == 1, desc="Filtering rows without top-k labels")

    ds_plus = ds_plus.remove_columns(["_has_any_top_"])
    return ds_plus

# =========================
# Main workflow
# =========================
def build_topk_csvs(
    out_dir: Path,
    top_k: int,
    val_size: float = 0.1,
    seed: int = 42,
    derive_topk_from: str = "train",  # 'train' | 'train+test' | 'all'
    filter_rows: bool = True,         # keep only rows that contain ≥1 top label
    include_text_cols: Sequence[str] = ("uid", "title", "content", "text"),
    add_prefix: Optional[str] = None, # optionally prefix label columns (e.g., "y_")
) -> Tuple[List[str], List[str]]:
    """
    Returns (top_label_names, top_label_columns) and writes CSVs:
      train_topk.csv, validation_topk.csv, test_topk.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base dataset and make validation split
    base = load_dataset("pietrolesci/amazoncat-13k")  # default config with 'train' and 'test'
    tv = base["train"].train_test_split(test_size=val_size, seed=seed)
    dsd = DatasetDict({"train": tv["train"], "validation": tv["test"], "test": base["test"]})

    # 2) Label vocab and top-K selection
    label_vocab = load_label_vocab()
    num_labels = len(label_vocab)

    if derive_topk_from == "all":
        pool_splits = ["train", "validation", "test"]
    elif derive_topk_from == "train+test":
        pool_splits = ["train", "test"]
    else:
        pool_splits = ["train"]

    counts = np.zeros(num_labels, dtype=np.int64)
    for sp in pool_splits:
        counts += compute_label_frequencies(dsd[sp], num_labels)

    top_indices = pick_topk_indices(counts, top_k)
    top_label_names = [label_vocab[i] for i in top_indices]

    # Sanitize columns (and ensure uniqueness)
    seen = set()
    top_label_columns = []
    for name in top_label_names:
        col = sanitize_label_for_column(name)
        if add_prefix:
            col = f"{add_prefix}{col}"
        # Ensure no dup after sanitization
        base_col = col
        j = 1
        while col in seen:
            col = f"{base_col}_{j}"
            j += 1
        seen.add(col)
        top_label_columns.append(col)

    # 3) Add indicator columns per split and write CSVs
    for split in ["train", "validation", "test"]:
        ds_aug = add_topk_indicator_columns(
            dsd[split],
            top_indices=top_indices,
            top_colnames=top_label_columns,
            filter_rows=filter_rows,
            batch_size=1000,
        )
        # Keep only desired columns
        keep_cols = [c for c in include_text_cols if c in ds_aug.column_names] + top_label_columns
        ds_final = ds_aug.remove_columns([c for c in ds_aug.column_names if c not in keep_cols])
        ds_final.to_csv(out_dir / f"{split}_top{top_k}.csv")

    # 4) Save a mapping file for traceability
    mapping = {
        "top_k": top_k,
        "derive_topk_from": derive_topk_from,
        "filter_rows": filter_rows,
        "label_vocab_size": num_labels,
        "top_indices": top_indices,
        "top_label_names": top_label_names,      # original names from vocab
        "top_label_columns": top_label_columns,  # sanitized/possibly prefixed column names in the CSVs
        "text_columns": list(include_text_cols),
    }
    (out_dir / f"top{top_k}_labels_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    return top_label_names, top_label_columns
# ['books', 'music', 'movies & tv', 'pop', 'literature & fiction', 'movies', 'education & reference', 'rock', 'used & rental textbooks', 'new']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=Path("data"), help="Where to write CSVs + mapping JSON")
    ap.add_argument("--top_k", type=int, default=10, help="Number of most frequent labels to keep")
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--derive_topk_from", choices=["train", "train+test", "all"], default="train",
                    help="Which splits to use to compute top-K frequencies")
    ap.add_argument("--keep_all_rows", action="store_true", help="If set, do NOT filter rows without any top label")
    ap.add_argument("--prefix", type=str, default=None, help="Optional prefix for label columns (e.g., 'y_')")
    ap.add_argument("--no_title_content", action="store_true",
                    help="If set, only keep 'uid' and 'text' (plus label columns) in CSVs")
    ap.add_argument("--encode_example", action="store_true",
                    help="If set, will immediately encode the train CSV as a quick sanity check")
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    include_cols = ["uid", "text"] if args.no_title_content else ["uid", "title", "content", "text"]

    top_names, top_cols = build_topk_csvs(
        out_dir=args.out_dir,
        top_k=args.top_k,
        val_size=args.val_size,
        seed=args.seed,
        derive_topk_from=args.derive_topk_from,
        filter_rows=(not args.keep_all_rows),
        include_text_cols=include_cols,
        add_prefix=args.prefix,
    )

    print(f"Top-{args.top_k} label names (first 10): {top_names[:10]}")
    print(f"CSV label columns (first 10): {top_cols[:10]}")

if __name__ == '__main__':
    main()
