"""Core module: Configuration and constants."""
from src.core.config import Config, get_config, set_config
from src.core.constants import DSM5_SYMPTOMS, get_severity_level, is_crisis_text

__all__ = ['Config', 'get_config', 'set_config', 'DSM5_SYMPTOMS', 'get_severity_level', 'is_crisis_text']
