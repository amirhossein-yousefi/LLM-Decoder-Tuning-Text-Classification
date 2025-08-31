from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from .config import AppConfig

logger = logging.getLogger(__name__)


def _dtype_from_str(s: str):
    s = s.lower()
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch_dtype: {s}")


def create_model(cfg: AppConfig) -> torch.nn.Module:
    """
    Builds a quantized sequence classifier with a LoRA adapter on decoder-only LMs.
    """
    quant_cfg: Optional[BitsAndBytesConfig] = None
    if cfg.model.use_4bit:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_dtype_from_str(cfg.model.torch_dtype),
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            logger.error("Could not initialize 4-bit quantization; ensure bitsandbytes is installed.")
            raise

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        token=cfg.model.hf_token,
        num_labels=len(cfg.data.label_columns),
        problem_type=cfg.model.problem_type,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=_dtype_from_str(cfg.model.torch_dtype),
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        target_modules=list(cfg.model.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)

    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    return model
