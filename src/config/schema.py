from pydantic import BaseModel, Field
from typing import Optional
import yaml

class ProjectConfig(BaseModel):
    name: str
    version: str
    seed: int

class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    outputs_path: str
    max_seq_length: int
    train_split: float
    val_split: float
    test_split: float

class ClassicalModelConfig(BaseModel):
    backbone: str
    num_labels: int
    dropout: float

class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    retry_attempts: int

class ModelConfig(BaseModel):
    classical: ClassicalModelConfig
    llm: LLMConfig

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    early_stopping_patience: int
    gradient_accumulation_steps: int
    fp16: bool

class ExplainabilityConfig(BaseModel):
    enable_attention: bool
    enable_integrated_gradients: bool
    enable_llm_rationales: bool
    top_k_tokens: int

class SafetyConfig(BaseModel):
    enable_guardrails: bool
    crisis_message: str

class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    explainability: ExplainabilityConfig
    safety: SafetyConfig

    @classmethod
    def load(cls, path: str = "configs/config.yaml") -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
