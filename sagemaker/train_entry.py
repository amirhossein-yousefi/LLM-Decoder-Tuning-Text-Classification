
import os
import sys
import argparse
import subprocess
from pathlib import Path

def _run_cli(config_path: str) -> int:
    cmd = [sys.executable, "-m", "llm_cls.cli", "train", "--config", config_path]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True).returncode

def _candidate_model_dirs(root: Path):
    for p in root.rglob("*"):
        if p.is_dir() and (p / "config.json").exists():
            yield p

def _choose_latest(dirs):
    dirs = list(dirs)
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None

def _export_to_sm_model_dir(checkpoint_dir: Path, sm_model_dir: Path, base_model: str | None):
    sm_model_dir.mkdir(parents=True, exist_ok=True)
    # Try to load a complete HF model first; if only LoRA adapters are present, merge them.
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir), ignore_mismatched_sizes=True, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    except Exception:
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel  # requires peft in requirements.txt
        if not base_model:
            raise RuntimeError("base_model is required to merge LoRA adapters into base weights.")
        base = AutoModelForSequenceClassification.from_pretrained(base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model.save_pretrained(sm_model_dir, safe_serialization=True)
    tok.save_pretrained(sm_model_dir)
    print(f"[train_entry] Exported HF model to {sm_model_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_key", default="default.yaml", help="Which file under configs/ to use")
    parser.add_argument("--base_model", default=None, help="HF model id (only needed if training produced LoRA adapters)")
    args, _ = parser.parse_known_args()

    # Your repo (whole repo is packaged as source_dir=".")
    config_path = Path("/opt/ml/code") / "configs" / args.config_key
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}. Package configs/ in source_dir or set --config_key accordingly.")

    # 1) Run your existing CLI exactly as documented in the README
    #    (python -m llm_cls.cli train --config configs/default.yaml)
    _ = _run_cli(str(config_path))  # README shows this usage.

    # 2) Find the latest checkpoint folder that looks like a HF model dir and export to SM_MODEL_DIR
    outputs_root = Path.cwd() / "outputs"
    best = _choose_latest(_candidate_model_dirs(outputs_root))
    if best is None:
        raise RuntimeError(f"Could not find a saved model directory with config.json under {outputs_root}")

    sm_model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    _export_to_sm_model_dir(best, sm_model_dir, base_model=args.base_model)

if __name__ == "__main__":
    main()
