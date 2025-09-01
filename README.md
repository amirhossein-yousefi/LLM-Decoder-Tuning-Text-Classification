# LLM Decoder Tuning for Text Classification
_Modular, memory‑efficient fine‑tuning of decoder‑only LLMs (LoRA + optional 4‑bit) for multi‑label classification._

<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue"></a>
  <a href="https://huggingface.co/docs/transformers/index"><img alt="Transformers" src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow"></a>
  <a href="https://huggingface.co/docs/peft/index"><img alt="PEFT" src="https://img.shields.io/badge/PEFT-LoRA-orange"></a>
  <a href="https://github.com/bitsandbytes-foundation/bitsandbytes"><img alt="bitsandbytes" src="https://img.shields.io/badge/bitsandbytes-4bit-green"></a>
</p>

---

## 🚀 Model on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-LLM--Tuning--Text--Classification-yellow.svg)](https://huggingface.co/Amirhossein75/LLM-Decoder-Tuning-Text-Classification)
<p align="center">
  <a href="https://huggingface.co/Amirhossein75/LLM-Decoder-Tuning-Text-Classification">
    <img src="https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-blueviolet?style=for-the-badge" alt="Hugging Face Repo">
  </a>
</p>

---

## TL;DR
- Fine‑tune decoder‑only LLMs (e.g., Llama‑style) for text classification using **LoRA** adapters.
- Go **memory‑light** with optional **4‑bit quantization** (QLoRA‑style) when supported by your system.
- **CSV in → metrics + JSONL out**. One‑line **train** and **predict** via a clean CLI and YAML config.
- Reproducible **run folders** under `outputs/<model_name>/<dataset_name>/run_<i>/`.

> This README is written for the repository structure containing `configs/`, `llm_cls/`, `scripts/`, and `tests/`, with a CLI entry at `llm_cls/cli.py` and a default config at `configs/default.yaml`.

---

## Why this repo?
Encoder models dominate text classification, but decoder‑only LLMs can be strong multi‑label classifiers—especially when you **don’t full‑tune** them. With **LoRA**, you update only tiny adapter matrices, keeping compute and storage small. Add **4‑bit quantization** and you can train larger models on a single GPU.

**Use this project when you want:** fast iteration, small checkpoints, simple CSV‑based datasets, and a batteries‑included CLI that “just works.”

---

## Features
- **Decoder‑only LLMs** via Hugging Face Transformers.
- **Parameter‑Efficient Fine‑Tuning (PEFT)** with LoRA adapters.
- **Optional 4‑bit quantization** (when available) for training/inference to fit bigger models on modest GPUs.
- **Config‑driven runs** (YAML) for data paths, model choice, hyper‑params, and LoRA knobs.
- **Predict to JSONL** (friendly for downstream evaluation or serving).
- **Reproducible outputs**: each run logs under `outputs/<model_name>/<dataset_name>/run_<i>/`.

---

## Project layout
```
.
├── configs/          # YAML configs (edit default.yaml for your data & model)
├── llm_cls/          # Python package: CLI, data pipeline, model wrapper, trainer, metrics
├── scripts/          # Convenience scripts for experimentation
├── tests/            # Unit tests
├── requirements.txt  # Core dependencies
├── pyproject.toml    # Build & project metadata
├── Makefile          # Common developer tasks
├─  README.md
├─ sagemaker/
│  ├─ train_entry.py          # wraps your CLI, then exports a HF model to SM_MODEL_DIR
│  ├─ inference.py            # SageMaker handler: model_fn / input_fn / predict_fn / output_fn
│  ├─ requirements.txt        # extras (peft, bitsandbytes if needed)
│  ├─ launch_training.py      # starts a SageMaker training job (HuggingFace Estimator)
│  └─ deploy_endpoint.py      # creates a real-time endpoint and returns a Predictor
```

---

## Quickstart

### 1) Install
Create a virtual environment and install in editable mode with dev extras:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

If your base model is gated on Hugging Face, set an access token:
```bash
export HF_TOKEN=hf_xxx   # or set it in your shell profile
```

### 2) Point to your data
Place your CSV files and update paths in `configs/default.yaml`:
- Train CSV
- Validation CSV
- (Optional) Test CSV
Note: you can use the [split_amazon_13k_data.py](llm_cls/split_amazon_13k_data.py) to create you these CSV
Your CSV should contain at least a **text column** and one or more **label fields** (multi‑label supported). Configure column names inside the YAML.

### 3) Train
```bash
python -m llm_cls.cli train --config configs/default.yaml
```
Artifacts (checkpoints, logs, metrics) land under:
```
outputs/<model_name>/<dataset_name>/run_<i>/
```

### 4) Predict
```bash
python -m llm_cls.cli predict   --config configs/default.yaml   --input_csv data/test.csv   --output_jsonl preds.jsonl
```

---

## Configuration (example)
> Treat this as a template; adapt field names to match `configs/default.yaml` in the repo.

```yaml
# configs/default.yaml (illustrative)
data:
  dataset_name: "amazon_13k"
  id_column: "uid"
  text_fields:
  target_text_column: "text"
  max_length: 1024
  train_csv: "llm_cls/data/train_top10.csv"
  val_csv: "llm_cls/data/validation_top10.csv"
  test_csv: "llm_cls/data/test_top10.csv"
  label_columns:
    - "books"
    - "movies_tv"
    - "music"
    - "pop"
    - "literature_fiction"
    - "movies"
    - "education_reference"
    - "rock"
    - "used_rental_textbooks"
    - "new"


model:
  model_name: "meta-llama/Llama-3.2-1B"
  hf_token: null         # or set HF_TOKEN env var
  problem_type: "multi_label_classification"
  use_4bit: true
  torch_dtype: "bfloat16"
  lora_r: 2
  lora_alpha: 2
  lora_dropout: 0.05
  lora_target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]

train:
  batch_size: 4
  num_train_epochs: 20
  learning_rate: 0.0002
  weight_decay: 0.01
  gradient_accumulation_steps: 8
  early_stopping_patience: 2
  metric_for_best_model: "f1_micro"
  save_total_limit: 1
  optim_8bit_when_4bit: true
  report_to: ['tensorboard']  # e.g., ["wandb"]

exp:
  output_root: "outputs"
  seeds: [1, 3, 4, 5]
  run_name: null

```

**Model switching.** Change `model.model_name` to try different backbones. For *encoder* baselines (e.g., `anferico/bert-for-patents`), set `use_4bit: false` and adjust expectations—4‑bit quantization in this project is mainly for decoder LLMs.

---
## 🖥️ Training Hardware & Environment

- **Device:** Laptop (Windows, WDDM driver model)  
- **GPU:** NVIDIA GeForce **RTX 3080 Ti Laptop GPU** (16 GB VRAM)  
- **Driver:** **576.52**  
- **CUDA (driver):** **12.9**  
- **PyTorch:** **2.8.0+cu129**  
- **CUDA available:** ✅ 


### 📊 Training Logs & Metrics

- **Total FLOPs (training):** `28,655,955,226,918,910`  
- **Training runtime:** `1,309.6418` seconds  
- **Logging:** TensorBoard-compatible logs in `.../outputs/meta-llama__Llama-3.2-1B/amazon_13k/run_0/logs`  

You can monitor training live with:

```bash
tensorboard --logdir .../outputs/meta-llama__Llama-3.2-1B/amazon_13k/run_0/logs
```


### Metrics

- **Best overall (micro-F1):** **0.830** at **5 epochs**  
- **Best minority‑class sensitivity (macro-F1):** **0.752** at **6 epochs**  
- **Average across 4 runs:** micro‑F1 **0.824**, macro‑F1 **0.741**, eval loss **0.161**  
- **Throughput:** train ≈ **0.784 steps/s** (**24.9 samples/s**) ; eval time ≈ **34.0s** per run.

> Interpretation: going from **4 → 5 epochs** gives the best **micro‑F1**; **6 epochs** squeezes out the top **macro‑F1**, hinting at slightly better coverage of minority classes with a tiny trade‑off in micro‑F1.

---
### 📈 Per‑run metrics
| Run | Epochs | Train Loss | Eval Loss | F1 (micro) | F1 (macro) | Train Time (s) | Train steps/s | Train samples/s | Eval Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1.400 | 0.157 | 0.824 | 0.738 | 1309.6 | 0.962 | 30.543 | 33.6 |
| 2 | 5 | 1.220 | 0.159 | 0.830 | 0.743 | 1640.3 | 0.768 | 24.385 | 34.0 |
| 3 | 6 | 1.063 | 0.162 | 0.826 | 0.752 | 1984.2 | 0.635 | 20.159 | 34.4 |
| 4 | 5 | 1.265 | 0.165 | 0.816 | 0.729 | 1639.3 | 0.769 | 24.401 | 34.0 |

<sub>*F1(micro)* aggregates decisions over all samples; *F1(macro)* averages per‑class F1 equally, highlighting minority‑class performance.</sub>

## 🧪 How these were produced

The repository exposes a minimal CLI for training and prediction:

```bash
# setup (Unix)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# train with your config
python -m llm_cls.cli train --config configs/default.yaml

# predict on a CSV
python -m llm_cls.cli predict \  --config configs/default.yaml \  --input_csv data/test.csv \  --output_jsonl preds.jsonl
```

- Place your `train/validation/test` CSVs and set paths in `configs/default.yaml`.
- Outputs are written under `outputs/<model_name>/<dataset_name>/run_<i>/`.

> Notes from the repo:
> - If the base model is gated, export `HF_TOKEN`.
> - If `bitsandbytes` isn't available (e.g., on Windows), set `model.use_4bit=false` in the config.
> - To try a non‑decoder baseline (e.g., `anferico/bert-for-patents`), set `model.model_name` and `use_4bit=false`.

---


## SageMaker Integration for `LLM-Decoder-Tuning-Text-Classification`

This guide documents the **Amazon SageMaker** training & inference workflow added under the `sagemaker/` folder. It lets you:

- **Train** your existing project on SageMaker using the Hugging Face Deep Learning Containers (DLCs).
- **Export** a `save_pretrained/` model into `SM_MODEL_DIR` so SageMaker can package it as `model.tar.gz`.
- **Deploy** a real‑time endpoint with a robust, multi‑label‑friendly inference handler.

> The integration is **non‑intrusive**: it wraps your current CLI (`python -m llm_cls.cli ...`) and does not change your training logic.


## Folder layout

```
sagemaker/
├─ train_entry.py          # wraps your CLI then exports a HF model to SM_MODEL_DIR
├─ inference.py            # model_fn / input_fn / predict_fn / output_fn (JSON in/out)
├─ requirements.txt        # extras for training/serving (peft, bitsandbytes)
├─ launch_training.py      # starts a SageMaker training job (HuggingFace Estimator)
└─ deploy_endpoint.py      # creates a real-time endpoint (HuggingFaceModel)
```

- **`train_entry.py`** calls your existing training command (see your root README), then finds the latest HF-style checkpoint under `outputs/` and saves a complete `save_pretrained` model and tokenizer into `SM_MODEL_DIR` (usually `/opt/ml/model`).  
- **LoRA/QLoRA:** If training produced **adapters** only, set `BASE_MODEL` so `train_entry.py` can merge adapters into base weights before export.  
- **`inference.py`** implements the standard SageMaker handler contract and returns **JSON**. It supports batched inputs, multi‑label with configurable thresholds, and GPU/CPU.


## Prerequisites

- **AWS** account + **IAM role** with permissions for SageMaker, S3, and CloudWatch (export ARN as `SM_EXECUTION_ROLE_ARN`).  
- **Data in S3** (CSV files). Example S3 layout:
  - `s3://<bucket>/<prefix>/data/train/train.csv`
  - `s3://<bucket>/<prefix>/data/validation/validation.csv`
- **Optional:** `HF_TOKEN` in the environment if you use gated models on the Hugging Face Hub.


## Configure your dataset paths (important)

SageMaker maps `Estimator.fit(inputs={"train": ..., "validation": ...})` to **channel directories** like:

```
/opt/ml/input/data/train/       # SM_CHANNEL_TRAIN
/opt/ml/input/data/validation/  # SM_CHANNEL_VALIDATION
```

Update your **config YAML** to point at those files *inside the container*. For example (adapt to your schema):

```yaml
# Example only — adjust keys to match your config
data:
  train_csv: /opt/ml/input/data/train/train.csv
  validation_csv: /opt/ml/input/data/validation/validation.csv
  # test_csv: /opt/ml/input/data/test/test.csv
```

> Tip: If your config supports `${env:VAR}` interpolation, you can use `${env:SM_CHANNEL_TRAIN}/train.csv` and `${env:SM_CHANNEL_VALIDATION}/validation.csv`.


## Quick start

### 1) Launch training

From the repo root (after adding this `sagemaker/` folder):

```bash
# Required: your SageMaker execution role
export SM_EXECUTION_ROLE_ARN="arn:aws:iam::<account>:role/<YourSageMakerRole>"

# Optional: pick a bucket/prefix for artifacts + data channels (defaults are fine)
export SM_BUCKET="<your-bucket>"              # defaults to session.default_bucket()
export SM_PREFIX="llm-decoder-cls"            # folder prefix

# Optional: S3 locations for data channels (if you used different keys)
export SM_TRAIN_S3="s3://$SM_BUCKET/$SM_PREFIX/data/train/"
export SM_VAL_S3="s3://$SM_BUCKET/$SM_PREFIX/data/validation/"

# Optional: DLC versions (check the HF DLC matrix for valid combos)
export TRANSFORMERS_VERSION="4.41"            # example
export PYTORCH_VERSION="2.3"
export PY_VERSION="py311"

# Optional: instance sizing
export SM_TRAIN_INSTANCE="ml.g5.2xlarge"
export SM_TRAIN_INSTANCE_COUNT=1

# Which config file under configs/ to pass to your CLI
export CONFIG_KEY="default.yaml"

# Only if training produced LoRA adapters that need merging
# export BASE_MODEL="meta-llama/Llama-3.1-8B"

python sagemaker/launch_training.py
```

When training finishes, the script prints the S3 URI for your model artifact (a `model.tar.gz`). Save it; you’ll need it to deploy.


### 2) Deploy a real‑time endpoint

```bash
# Use the S3 path printed at the end of training:
export SM_TRAINED_MODEL_S3="s3://<bucket>/<prefix>/output/<job-id>/output/model.tar.gz"

# Inference runtime toggles (all optional)
export MULTILABEL="true"
export MULTILABEL_THRESHOLD="0.5"
export MAX_LENGTH="512"
export BATCH_SIZE="16"
# export HF_TOKEN="..."    # if needed for gated models

# Instance sizing
export SM_INF_INSTANCE="ml.g5.xlarge"
export SM_INF_INSTANCE_COUNT=1

# Endpoint name (optional)
export SM_ENDPOINT_NAME="llm-decoder-cls-endpoint"

python sagemaker/deploy_endpoint.py
```

The script prints your **endpoint name**.


### 3) Invoke the endpoint

Use the Python SDK’s `Predictor`. Because the handler expects **JSON**, set the serializer & deserializer:

```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = "llm-decoder-cls-endpoint"  # or whatever was printed
p = Predictor(endpoint_name, serializer=JSONSerializer(), deserializer=JSONDeserializer())

p.predict({"text": ["great product!", "terrible support"]})
# → [[{"label": "POSITIVE", "score": 0.98}], [{"label": "NEGATIVE", "score": 0.97}]]
```

The handler also supports a single string or a raw list of strings as input.


## How it works under the hood

- **Training** uses the `HuggingFace` Estimator in the SageMaker Python SDK. Your repository is sent as `source_dir`, `sagemaker/train_entry.py` is the `entry_point`, and S3 inputs are mounted under **channels** (e.g., `/opt/ml/input/data/train/`).  
- `train_entry.py` runs your **existing CLI**:  
  `python -m llm_cls.cli train --config /opt/ml/code/configs/<CONFIG_KEY>`  
  Then it locates the most recent HF‑style checkpoint under `outputs/` and **exports** a `save_pretrained` model to `SM_MODEL_DIR` so SageMaker can package it as `model.tar.gz` for deployment.
- **Serving** uses `HuggingFaceModel(...).deploy(...)` and your custom `inference.py` implementing `model_fn`, `input_fn`, `predict_fn`, and `output_fn`.  
  - JSON shape: `{"text": "..."} | {"text": ["...","..."]} | ["...","..."]`.  
  - Multi‑label mode applies a sigmoid and threshold per class; otherwise softmax.  
  - LoRA adapters are detected and merged when possible.


## Choose compatible DLC versions

Set `TRANSFORMERS_VERSION`, `PYTORCH_VERSION`, and `PY_VERSION` in the environment to match **supported** Hugging Face DLC tags for your region. See the “Available DLCs” matrix and pick a trio that exists in your AWS region.


## Common customizations

- **Different data layouts** — Add more channels in `launch_training.py` and reference them from your config (e.g., `test`, `aux`).  
- **Larger batch sizes** — Increase `BATCH_SIZE` for inference.  
- **Thresholding** — Adjust `MULTILABEL_THRESHOLD` per your validation metrics.  
- **Gated models** — Export `HF_TOKEN` to allow the DLC to pull private/gated models.  
- **No custom handler** — If you don’t need post‑processing, you can deploy “zero‑code” by setting `HF_TASK="text-classification"` and omitting `entry_point` (not used by these scripts).


## Troubleshooting

- **“File not found: /opt/ml/input/data/train/train.csv”** — Ensure your S3 path is correct and that `launch_training.py` passes an `"inputs"` dict with a `"train"` key.  
- **“No model found to export”** — Confirm your training wrote an HF model directory containing `config.json` somewhere under `outputs/`.  
- **Adapter‑only checkpoints** — Provide `BASE_MODEL` so the export step can merge adapters.  
- **Serialization errors when invoking** — Make sure you used `JSONSerializer()`/`JSONDeserializer()` with `Predictor`.  
- **Version mismatches** — Align `transformers_version`, `pytorch_version`, and `py_version` to a valid DLC trio.


## Cost & cleanup

SageMaker **real‑time endpoints accrue charges while running**. After testing, delete the endpoint (and optionally the model & endpoint config) to stop billing. You can safely keep the `model.tar.gz` in S3.


## Security

Keep tokens and secrets (e.g., `HF_TOKEN`) in **environment variables** or a secrets manager. Do **not** commit secrets to the repository.


---

### At a glance: environment variables used by these scripts

| Variable | Purpose |
|---|---|
| `SM_EXECUTION_ROLE_ARN` | IAM role used by SageMaker (required). |
| `SM_BUCKET`, `SM_PREFIX` | Optional S3 bucket/prefix for artifacts. |
| `SM_TRAIN_S3`, `SM_VAL_S3` | S3 URIs for data channels. |
| `CONFIG_KEY` | Which file under `configs/` to pass to your CLI. |
| `BASE_MODEL` | Base HF model id used to merge LoRA adapters (optional). |
| `TRANSFORMERS_VERSION`, `PYTORCH_VERSION`, `PY_VERSION` | DLC versions. |
| `SM_TRAIN_INSTANCE(_COUNT)` | Training instance type & count. |
| `SM_INF_INSTANCE(_COUNT)` | Inference instance type & count. |
| `SM_ENDPOINT_NAME` | Name of the endpoint to create. |
| `HF_TOKEN` | Optional access token for gated Hub models. |
| `MULTILABEL`, `MULTILABEL_THRESHOLD`, `MAX_LENGTH`, `BATCH_SIZE` | Inference behavior toggles. |


## Practical guidance

### Choosing a base model
- Start with 7–8B‑parameter models for strong baselines and reasonable VRAM needs.
- Prefer instruction‑tuned variants for cleaner zero‑shot and better decoder behavior.
- For multilingual tasks, pick a multilingual decoder (or evaluate an encoder baseline as a control).

### Multi‑label specifics
- Use threshold‑based decoding (e.g., sigmoid + class‑wise threshold) to improve F1.
- Track **micro/macro F1** and per‑class metrics; long‑tail classes benefit from class weights and augmentation.

### Hardware & performance
- 4‑bit quantization dramatically lowers memory pressure; if you hit errors or you’re on Windows, set `model.use_4bit: false` and fall back to standard precision LoRA or move to Linux for bitsandbytes support.
- To avoid OOM: reduce `batch_size` and `max_length`, use gradient accumulation, and prefer bf16/fp16 if available.

---

## Troubleshooting

**“4‑bit not available” / bitsandbytes import errors**
- Ensure a recent CUDA‑compatible PyTorch and the latest `bitsandbytes`.
- On Windows, native bitsandbytes support is limited; either switch to Linux or disable 4‑bit by setting `model.use_4bit: false`.
- If problems persist, train without 4‑bit and re‑enable once your toolchain is set.

**Gated model access**
- Export `HF_TOKEN` before running if the model requires it.

**Unexpected metrics**
- Verify label column(s) and delimiter for multi‑label data.
- Check that `max_length` doesn’t truncate crucial content; try 512–1024 for long texts.

---

## Roadmap ideas
- Early stopping & LR scheduling callbacks
- Mixed precision (bf16) toggles and CPU offload knobs
- Ready‑made dataset loaders (ag_news, yelp, etc.)
- Better multi‑label threshold tuning utilities

---

## References & acknowledgments
- **LoRA**: Low‑Rank Adaptation for efficient fine‑tuning of large models.
- **QLoRA / 4‑bit**: 4‑bit quantization with LoRA adapters for single‑GPU training.
- **Ecosystem**: Hugging Face Transformers & PEFT, bitsandbytes.

---

## Citation
Please cite the underlying techniques if you use them in academic work:

```
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```
