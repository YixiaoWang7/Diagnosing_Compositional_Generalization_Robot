"""Project-owned simulator evaluation, loaded without simulator dependencies."""

from .config import EvaluationConfig
from .runner import evaluate

__all__ = ["EvaluationConfig", "evaluate"]
