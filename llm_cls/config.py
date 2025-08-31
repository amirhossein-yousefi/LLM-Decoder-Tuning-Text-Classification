from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "WIPO_Emerging_Vision"
    id_column: str = "Publication Number"
    # Label columns are 0/1 per class
    label_columns: Sequence[str] = field(default_factory=lambda: [
        "Adaptive focus",
        "Artificial Iris",
        "Artificial silicon retina (ASR)/Retinal prostheses",
        "Augmented Reality Devices",
        "Bionic eye (system)",
        "Cortical Implants",
        "Drug delivery",
        "Hand Wearables",
        "IOL with Sensors",
        "Intracorneal lenses",
        "Multifocal",
        "Smart Eyewear",
        "Telescopic Lenses",
        "Virtual Reality Devices",
    ])
    text_fields: Sequence[str] = field(default_factory=lambda: ["title", "fclaim", "abstract"])
    target_text_column: str = "text"
    max_length: int = 1024
    # Paths
    train_csv: str = "data/train.csv"
    val_csv: str = "data/validation.csv"
    test_csv: str = "data/test.csv"


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    problem_type: str = "multi_label_classification"
    use_4bit: bool = True
    torch_dtype: str = "bfloat16"  # "bfloat16" or "float16"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
    ])


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_train_epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1_micro"
    save_total_limit: int = 1
    # Optimizer automatically switches to 8-bit if quantized
    optim_8bit_when_4bit: bool = True
    report_to: Sequence[str] = field(default_factory=list)  # ["wandb"] if you want


@dataclass
class ExperimentConfig:
    output_root: str = "outputs"
    seeds: List[int] = field(default_factory=lambda: [1, 3, 4, 5])
    # optional run tag
    run_name: Optional[str] = None


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)

    @staticmethod
    def from_yaml(path: str) -> "AppConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        # yaml->dataclasses
        return AppConfig(
            data=DataConfig(**cfg.get("data", {})),
            model=ModelConfig(**cfg.get("model", {})),
            train=TrainConfig(**cfg.get("train", {})),
            exp=ExperimentConfig(**cfg.get("exp", {})),
        )

    def to_dict(self):
        return asdict(self)

    def output_dir(self, run_number: int) -> str:
        import os as _os
        return _os.path.join(
            self.exp.output_root, self.model.model_name.replace("/", "__"),
            self.data.dataset_name, f"run_{run_number}"
        )
