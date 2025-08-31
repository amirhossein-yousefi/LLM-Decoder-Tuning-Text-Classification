from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
