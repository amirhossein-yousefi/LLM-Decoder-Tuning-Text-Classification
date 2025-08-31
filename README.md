# TExt Classification using LLMs (Decoder-only, LoRA + 4-bit)

Modular training & inference stack for multi-label classification using decoder-based LLMs (e.g., Llama) with LoRA adapters and 4-bit quantization.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Place your CSVs (train/validation/test) and set paths in `configs/default.yaml`.

## Train
```bash
python -m llm_cls.cli train --config configs/default.yaml
```

Outputs will land under:
```
outputs/<model_name>/<dataset_name>/run_<i>/
```

## Predict
```bash
python -m llm_cls.cli predict --config configs/default.yaml   --input_csv data/test.csv   --output_jsonl preds.jsonl
```

## Notes
- Set `HF_TOKEN` if the model requires gated access.
- If `bitsandbytes` is unavailable (or Windows), set `model.use_4bit=false` in config.
- Switch models by editing `model.model_name` (e.g. `anferico/bert-for-patents`) and set `use_4bit=false` for non-decoder baselines.
