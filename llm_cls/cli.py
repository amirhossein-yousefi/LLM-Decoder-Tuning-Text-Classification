from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import AppConfig
from .training import run_training
from .utils import setup_logging, ensure_dir, save_json
from .inference import predict_csv

app = typer.Typer(help="LLM decoder-based classification (LoRA + 4-bit)")

@app.command()
def train(
    config: str = typer.Option(..., exists=True, readable=True, help="Path to YAML config."),
    override_output_root: Optional[str] = typer.Option(None, help="Override exp.output_root."),
):
    """Train + evaluate across seeds configured in the YAML."""
    setup_logging(logging.INFO)
    cfg = AppConfig.from_yaml(config)
    if override_output_root:
        cfg.exp.output_root = override_output_root

    # Load data
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    all_train, all_test = [], []
    for i, seed in enumerate(cfg.exp.seeds):
        run_dir = cfg.output_dir(i)
        ensure_dir(run_dir)
        typer.echo(f"=== Run {i} (seed={seed}) -> {run_dir}")
        tr, te = run_training(cfg, train_df, val_df, test_df, run_number=i, seed=seed)
        all_train.append(tr)
        all_test.append(te)
        save_json(te, str(Path(run_dir) / "test_metrics.json"))

    summary = {"train": all_train, "test": all_test}
    print(json.dumps(summary, indent=2))


@app.command()
def predict(
    config: str = typer.Option(..., exists=True, help="Path to YAML config."),
    input_csv: str = typer.Option(..., exists=True, help="CSV with raw fields to score."),
    output_jsonl: str = typer.Option("preds.jsonl", help="Where to write predictions."),
):
    """Batch inference on a CSV using the best checkpoint from run_0."""
    setup_logging(logging.INFO)
    cfg = AppConfig.from_yaml(config)
    predict_csv(input_csv=input_csv, cfg=cfg, output_jsonl=output_jsonl)
    typer.echo(f"Wrote predictions to: {output_jsonl}")


if __name__ == "__main__":
    app()
