"""Model components including uncertainty-aware architecture and custom components."""

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
    UncertaintyAwareModel,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
    UncertaintyQuantificationHead,
    CurriculumLoss,
    monte_carlo_dropout_inference,
)

__all__ = [
    "UncertaintyAwareModel",
    "UncertaintyQuantificationHead",
    "CurriculumLoss",
    "monte_carlo_dropout_inference",
]
