"""
Configuration management for the LLM Knowledge Distillation project.

This module provides centralized configuration handling using YAML files
with support for environment variable overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class HardwareConfig:
    """Hardware-related configuration."""
    device: str = "auto"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    compile_model: bool = False


@dataclass
class ModelConfig:
    """Model configuration for teacher or student."""
    name: str
    quantization: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    device_map: str = "auto"
    trust_remote_code: bool = True
    torch_dtype: Optional[str] = None
    use_cache: bool = True


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None
    fan_in_fan_out: bool = False


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "databricks/databricks-dolly-15k"
    task_type: str = "instruction_following"
    text_column: str = "instruction"
    context_column: str = "context"
    response_column: str = "response"
    category_column: str = "category"
    train_split: str = "train"
    validation_split: float = 0.1
    test_split: float = 0.05
    max_samples: Optional[int] = None


@dataclass
class TokenizationConfig:
    """Tokenization configuration."""
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./artifacts/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "none"


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    temperature: float = 2.0
    alpha: float = 0.3  # CE loss weight
    beta: float = 0.7   # KD loss weight
    use_kd: bool = True


@dataclass
class OptunaRoundConfig:
    """Configuration for a single Optuna optimization round."""
    n_trials: int = 20
    timeout: int = 3600
    study_name: str = "optuna_study"
    direction: str = "minimize"


@dataclass
class OptunaSearchSpace:
    """Search space for Optuna hyperparameter optimization."""
    learning_rate: List[float] = field(default_factory=lambda: [1.0e-5, 5.0e-4])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 0.1])
    lora_r: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    lora_dropout: List[float] = field(default_factory=lambda: [0.0, 0.1])
    temperature: List[float] = field(default_factory=lambda: [1.0, 5.0])
    alpha: List[float] = field(default_factory=lambda: [0.1, 0.5])
    beta: List[float] = field(default_factory=lambda: [0.5, 0.9])
    per_device_train_batch_size: List[int] = field(default_factory=lambda: [1, 2, 4])
    gradient_accumulation_steps: List[int] = field(default_factory=lambda: [4, 8, 16])
    num_train_epochs: List[int] = field(default_factory=lambda: [1, 2, 3])
    warmup_ratio: List[float] = field(default_factory=lambda: [0.0, 0.1])
    max_length: List[int] = field(default_factory=lambda: [256, 512, 1024])


@dataclass
class OptunaConfig:
    """Complete Optuna configuration."""
    round1: OptunaRoundConfig = field(default_factory=lambda: OptunaRoundConfig(
        n_trials=20, timeout=3600, study_name="kd_round1"
    ))
    round2: OptunaRoundConfig = field(default_factory=lambda: OptunaRoundConfig(
        n_trials=75, timeout=28800, study_name="kd_round2"
    ))
    search_space: OptunaSearchSpace = field(default_factory=OptunaSearchSpace)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1", "precision", "recall", "perplexity"
    ])
    generate_during_eval: bool = True
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "./artifacts/logs"
    tensorboard_dir: str = "./artifacts/logs/tensorboard"
    wandb_enabled: bool = False
    wandb_project: str = "llm-distillation"
    wandb_entity: Optional[str] = None


@dataclass
class ArtifactsConfig:
    """Artifacts storage configuration."""
    checkpoint_dir: str = "./artifacts/checkpoints"
    best_model_dir: str = "./artifacts/best_model"
    plots_dir: str = "./artifacts/plots"
    logs_dir: str = "./artifacts/logs"
    optuna_dir: str = "./artifacts/optuna"
    reports_dir: str = "./artifacts/reports"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 8
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    device: str = "auto"


@dataclass
class AppConfig:
    """Streamlit app configuration."""
    title: str = "LLM Knowledge Distillation Demo"
    description: str = "Interactive demo for distilled student model"
    max_input_length: int = 1024
    default_max_new_tokens: int = 128
    default_temperature: float = 0.7
    default_top_p: float = 0.9


@dataclass
class HubConfig:
    """HuggingFace Hub upload configuration."""
    enabled: bool = False
    repo_id: Optional[str] = None
    private: bool = False
    token: Optional[str] = None
    merge_lora: bool = True
    commit_message: str = "Upload distilled student model"
    tags: List[str] = field(default_factory=lambda: ["distillation", "lora"])
    license: str = "mit"
    base_model: Optional[str] = None
    teacher_model: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class Config:
    """Main configuration class aggregating all sub-configs."""
    project: Dict[str, Any] = field(default_factory=dict)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    app: AppConfig = field(default_factory=AppConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if "project" in data:
            config.project = data["project"]
        
        if "hardware" in data:
            config.hardware = HardwareConfig(**data["hardware"])
        
        if "models" in data:
            config.models = {
                name: ModelConfig(**cfg) 
                for name, cfg in data["models"].items()
            }
        
        if "lora" in data:
            config.lora = LoRAConfig(**data["lora"])
        
        if "dataset" in data:
            config.dataset = DatasetConfig(**data["dataset"])
        
        if "tokenization" in data:
            config.tokenization = TokenizationConfig(**data["tokenization"])
        
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        
        if "distillation" in data:
            config.distillation = DistillationConfig(**data["distillation"])
        
        if "optuna" in data:
            optuna_data = data["optuna"]
            config.optuna = OptunaConfig(
                round1=OptunaRoundConfig(**optuna_data.get("round1", {})),
                round2=OptunaRoundConfig(**optuna_data.get("round2", {})),
                search_space=OptunaSearchSpace(**optuna_data.get("search_space", {}))
            )
        
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])
        
        if "logging" in data:
            logging_data = data["logging"]
            config.logging = LoggingConfig(
                level=logging_data.get("level", "INFO"),
                log_dir=logging_data.get("log_dir", "./artifacts/logs"),
                tensorboard_dir=logging_data.get("tensorboard_dir", "./artifacts/logs/tensorboard"),
                wandb_enabled=logging_data.get("wandb", {}).get("enabled", False),
                wandb_project=logging_data.get("wandb", {}).get("project", "llm-distillation"),
                wandb_entity=logging_data.get("wandb", {}).get("entity")
            )
        
        if "artifacts" in data:
            config.artifacts = ArtifactsConfig(**data["artifacts"])
        
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])
        
        if "app" in data:
            config.app = AppConfig(**data["app"])
        
        if "hub" in data:
            config.hub = HubConfig(**data["hub"])
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project": self.project,
            "hardware": self.hardware.__dict__,
            "models": {k: v.__dict__ for k, v in self.models.items()},
            "lora": self.lora.__dict__,
            "dataset": self.dataset.__dict__,
            "tokenization": self.tokenization.__dict__,
            "training": self.training.__dict__,
            "distillation": self.distillation.__dict__,
            "optuna": {
                "round1": self.optuna.round1.__dict__,
                "round2": self.optuna.round2.__dict__,
                "search_space": self.optuna.search_space.__dict__
            },
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__,
            "artifacts": self.artifacts.__dict__,
            "inference": self.inference.__dict__,
            "app": self.app.__dict__,
            "hub": self.hub.__dict__
        }

    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
    
    Returns:
        Config object with all settings.
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            Path("configs/default.yaml"),
            Path("../configs/default.yaml"),
            Path("../../configs/default.yaml"),
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            # Return default config
            return Config()
    
    return Config.from_yaml(config_path)


def apply_env_overrides(config: Config) -> Config:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with KD_ and use __ as separator.
    Example: KD_TRAINING__LEARNING_RATE=1e-5
    """
    env_prefix = "KD_"
    
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Remove prefix and split by __
            config_key = key[len(env_prefix):].lower()
            parts = config_key.split("__")
            
            # Navigate to the correct attribute
            current = config
            for part in parts[:-1]:
                current = getattr(current, part, None)
                if current is None:
                    break
            
            if current is not None:
                # Try to convert value to appropriate type
                final_key = parts[-1]
                current_value = getattr(current, final_key, None)
                
                if current_value is not None:
                    if isinstance(current_value, bool):
                        value = value.lower() in ("true", "1", "yes")
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, list):
                        value = [v.strip() for v in value.split(",")]
                
                setattr(current, final_key, value)
    
    return config
