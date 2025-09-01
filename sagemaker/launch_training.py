import os, time
import sagemaker
from sagemaker.huggingface import HuggingFace

# ---- Fill (or export) these before running ----
role = os.environ.get("SM_EXECUTION_ROLE_ARN", "arn:aws:iam::<account>:role/<YourSageMakerRole>")

sess = sagemaker.Session()
bucket = os.environ.get("SM_BUCKET", sess.default_bucket())
prefix = os.environ.get("SM_PREFIX", "llm-decoder-cls")

# S3 locations containing train.csv and validation.csv
train_s3 = os.environ.get("SM_TRAIN_S3", f"s3://{bucket}/{prefix}/data/train/")
val_s3   = os.environ.get("SM_VAL_S3",   f"s3://{bucket}/{prefix}/data/validation/")

# Pick supported versions from the DLC matrix:
# https://huggingface.co/docs/sagemaker/en/dlcs/available
transformers_version = os.environ.get("TRANSFORMERS_VERSION", "4.41")
pytorch_version      = os.environ.get("PYTORCH_VERSION", "2.3")
py_version           = os.environ.get("PY_VERSION", "py311")

estimator = HuggingFace(
    entry_point      ="sagemaker/train_entry.py",
    source_dir       =".",  # include the whole repo so configs/ and llm_cls/ are available
    role             = role,
    instance_type    = os.environ.get("SM_TRAIN_INSTANCE", "ml.g5.2xlarge"),
    instance_count   = int(os.environ.get("SM_TRAIN_INSTANCE_COUNT", "1")),
    transformers_version = transformers_version,
    pytorch_version      = pytorch_version,
    py_version           = py_version,
    hyperparameters = {
        # which config under configs/ to pass to your CLI (adjust to your file)
        "config_key": os.environ.get("CONFIG_KEY", "default.yaml"),
        # only needed if your checkpoint contains LoRA adapters to be merged
        "base_model": os.environ.get("BASE_MODEL", ""),  # e.g. meta-llama/Llama-3.1-8B
    },
    environment = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # set if you use gated models
    },
)

inputs = {
    "train": sagemaker.inputs.TrainingInput(train_s3, distribution="FullyReplicated"),
    "validation": sagemaker.inputs.TrainingInput(val_s3, distribution="FullyReplicated"),
}

job_name = os.environ.get("SM_JOB_NAME", f"llm-decoder-cls-{int(time.time())}")
estimator.fit(inputs=inputs, job_name=job_name)
print("Model artifacts (S3):", estimator.model_data)
