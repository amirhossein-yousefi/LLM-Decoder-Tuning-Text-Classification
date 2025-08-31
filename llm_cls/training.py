from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import pandas as pd
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from .config import AppConfig
from .data import create_tokenizer, data_collator, make_encoded_dataset
from .metrics import MultiLabelMetrics
from .model import create_model
from .utils import set_global_seed

logger = logging.getLogger(__name__)


def run_training(
    cfg: AppConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_number: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Trains and evaluates once. Returns (train_metrics, test_metrics).
    """
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_global_seed(seed)

    tokenizer = create_tokenizer(cfg)
    collator = data_collator(tokenizer)

    train_ds = make_encoded_dataset(train_df, cfg, tokenizer)
    val_ds = make_encoded_dataset(val_df, cfg, tokenizer)
    test_ds = make_encoded_dataset(test_df, cfg, tokenizer)

    train_ds=train_ds.select(range(2000))
    val_ds = val_ds.select(range(500))
    test_ds = test_ds.select(range(500))

    optim = "adamw_8bit" if cfg.model.use_4bit and cfg.train.optim_8bit_when_4bit else "adamw_torch"

    args = TrainingArguments(
        output_dir=cfg.output_dir(run_number),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=max(cfg.train.batch_size, 1),
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        save_total_limit=cfg.train.save_total_limit,
        metric_for_best_model=cfg.train.metric_for_best_model,
        load_best_model_at_end=True,
        seed=seed,
        data_seed=seed,
        optim=optim,
        report_to=list(cfg.train.report_to),
        logging_dir=os.path.join(cfg.output_dir(run_number), "logs"),
    )

    metrics_fn = MultiLabelMetrics(num_labels=len(cfg.data.label_columns))
    trainer = Trainer(
        model_init=lambda: create_model(cfg),
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)],
    )

    logger.info("Starting training...")
    train_out = trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    train_metrics = dict(getattr(train_out, "metrics", {}) or {})
    logger.info("Finished. Test metrics: %s", test_metrics)

    return train_metrics, test_metrics
