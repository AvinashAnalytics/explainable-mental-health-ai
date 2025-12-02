"""
Configuration management system.

Loads YAML config and provides type-safe access to settings.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Central configuration object."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_default(cls) -> Config:
        """Load default configuration."""
        default_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls({})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    @property
    def data_paths(self) -> Dict[str, str]:
        return self.get('data.paths', {})
    
    @property
    def model_name(self) -> str:
        return self.get('model.name', 'distilbert-base-uncased')
    
    @property
    def max_length(self) -> int:
        return self.get('model.max_length', 256)
    
    @property
    def batch_size(self) -> int:
        return self.get('training.batch_size', 16)
    
    @property
    def learning_rate(self) -> float:
        return self.get('training.learning_rate', 2e-5)
    
    @property
    def epochs(self) -> int:
        return self.get('training.epochs', 3)
    
    @property
    def seed(self) -> int:
        return self.get('training.seed', 42)
    
    @property
    def llm_provider(self) -> str:
        return self.get('llm.provider', 'openai')
    
    @property
    def llm_model(self) -> str:
        return self.get('llm.model', 'gpt-4o-mini')
    
    @property
    def llm_temperature(self) -> float:
        return self.get('llm.temperature', 0.2)
    
    @property
    def enable_safety_checks(self) -> bool:
        return self.get('safety.enable_checks', True)
    
    @property
    def enable_crisis_routing(self) -> bool:
        return self.get('safety.enable_crisis_routing', True)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config.from_default()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config
