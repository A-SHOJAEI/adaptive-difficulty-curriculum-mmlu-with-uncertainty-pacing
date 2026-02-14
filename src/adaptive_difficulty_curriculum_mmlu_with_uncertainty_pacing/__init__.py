"""Adaptive Difficulty Curriculum Learning for MMLU with Uncertainty-Based Pacing.

A novel curriculum learning framework that dynamically adjusts task difficulty
ordering using epistemic uncertainty estimates from Monte Carlo dropout.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
    UncertaintyAwareModel,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.training.trainer import (
    AdaptiveCurriculumTrainer,
)

__all__ = [
    "UncertaintyAwareModel",
    "AdaptiveCurriculumTrainer",
]
