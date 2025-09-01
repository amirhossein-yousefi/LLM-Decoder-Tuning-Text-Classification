import os
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

role = os.environ.get("SM_EXECUTION_ROLE_ARN", "arn:aws:iam::<account>:role/<YourSageMakerRole>")
model_data = os.environ["SM_TRAINED_MODEL_S3"]  # set to estimator.model_data from training

transformers_version = os.environ.get("TRANSFORMERS_VERSION", "4.41")
pytorch_version      = os.environ.get("PYTORCH_VERSION", "2.3")
py_version           = os.environ.get("PY_VERSION", "py311")

env = {
    "MULTILABEL": os.environ.get("MULTILABEL", "true"),
    "MULTILABEL_THRESHOLD": os.environ.get("MULTILABEL_THRESHOLD", "0.5"),
    "MAX_LENGTH": os.environ.get("MAX_LENGTH", "512"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE", "16"),
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
}

model = HuggingFaceModel(
    model_data = model_data,
    role       = role,
    transformers_version = transformers_version,
    pytorch_version      = pytorch_version,
    py_version           = py_version,
    entry_point = "sagemaker/inference.py",
    source_dir  = ".",
    env         = env,
)

predictor = model.deploy(
    initial_instance_count = int(os.environ.get("SM_INF_INSTANCE_COUNT", "1")),
    instance_type          = os.environ.get("SM_INF_INSTANCE", "ml.g5.xlarge"),
    endpoint_name          = os.environ.get("SM_ENDPOINT_NAME", "llm-decoder-cls-endpoint"),
)

print("Endpoint:", predictor.endpoint_name)
