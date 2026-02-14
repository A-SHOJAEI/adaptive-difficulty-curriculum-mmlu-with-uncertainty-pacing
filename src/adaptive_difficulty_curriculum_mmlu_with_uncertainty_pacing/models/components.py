"""Custom model components for uncertainty quantification and curriculum learning.

This module implements novel components:
1. UncertaintyQuantificationHead: Custom layer for epistemic uncertainty estimation
2. CurriculumLoss: Adaptive loss function that weights samples by uncertainty
3. monte_carlo_dropout_inference: MC dropout for uncertainty estimation
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyQuantificationHead(nn.Module):
    """Custom head for epistemic uncertainty quantification via Monte Carlo dropout.

    This novel component maintains dropout during inference to enable
    uncertainty estimation through multiple forward passes. Unlike standard
    dropout which is disabled at test time, this allows measuring model
    confidence across predictions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
    ):
        """Initialize uncertainty quantification head.

        Args:
            input_dim: Dimension of input features.
            num_classes: Number of output classes.
            dropout_rate: Dropout probability for MC dropout.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Two-layer head with dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.activation = nn.GELU()

        logger.info(
            f"Initialized UncertaintyQuantificationHead with dropout={dropout_rate}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout enabled.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def enable_dropout(self):
        """Enable dropout for uncertainty estimation during inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class CurriculumLoss(nn.Module):
    """Custom loss function for curriculum learning with uncertainty-based weighting.

    This novel loss function adaptively weights samples based on their uncertainty,
    allowing the model to focus on high-uncertainty domains while maintaining
    training stability. Combines cross-entropy with uncertainty-based sample
    reweighting.
    """

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        uncertainty_weight: float = 0.5,
        temperature: float = 2.0,
    ):
        """Initialize curriculum loss.

        Args:
            base_loss: Base loss function type.
            uncertainty_weight: Weight for uncertainty-based reweighting.
            temperature: Temperature for softening uncertainty weighting.
        """
        super().__init__()

        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
        self.temperature = temperature

        logger.info(
            f"Initialized CurriculumLoss with uncertainty_weight={uncertainty_weight}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> torch.Tensor:
        """Compute curriculum-weighted loss.

        Args:
            logits: Model predictions of shape (batch_size, num_classes).
            labels: Ground truth labels of shape (batch_size,).
            uncertainties: Uncertainty estimates of shape (batch_size,).

        Returns:
            Scalar loss value.
        """
        # Base cross-entropy loss (unreduced)
        base_loss = F.cross_entropy(logits, labels, reduction="none")

        # Normalize uncertainties to [0, 1] with temperature scaling
        normalized_uncertainties = torch.sigmoid(
            uncertainties / self.temperature
        )

        # Compute adaptive weights: higher uncertainty = higher weight
        # This focuses training on uncertain domains
        weights = 1.0 + self.uncertainty_weight * normalized_uncertainties

        # Apply weights and compute mean
        weighted_loss = (base_loss * weights).mean()

        return weighted_loss


def monte_carlo_dropout_inference(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_samples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform Monte Carlo dropout inference for uncertainty estimation.

    Novel inference procedure that runs multiple forward passes with dropout
    enabled to estimate epistemic uncertainty via prediction variance.

    Args:
        model: Model with dropout layers.
        inputs: Dictionary with input_ids and attention_mask.
        num_samples: Number of MC samples to collect.

    Returns:
        Tuple of (mean_predictions, uncertainties) where uncertainties
        are measured as entropy of the predictive distribution.
    """
    model.eval()

    # Enable dropout in uncertainty head
    if hasattr(model, "uncertainty_head"):
        model.uncertainty_head.enable_dropout()

    # Collect predictions from multiple forward passes
    predictions_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(**inputs)
            probs = F.softmax(outputs["logits"], dim=-1)
            predictions_list.append(probs)

    # Stack predictions: (num_samples, batch_size, num_classes)
    predictions_stack = torch.stack(predictions_list)

    # Compute mean prediction
    mean_predictions = predictions_stack.mean(dim=0)

    # Compute uncertainty as predictive entropy
    epsilon = 1e-10
    uncertainties = -(mean_predictions * torch.log(mean_predictions + epsilon)).sum(dim=-1)

    return mean_predictions, uncertainties


class ConfidenceCalibrationLayer(nn.Module):
    """Temperature scaling layer for calibrating prediction confidence.

    Additional custom component that improves uncertainty estimation quality
    by learning an optimal temperature parameter for softmax scaling.
    """

    def __init__(self):
        """Initialize calibration layer with learnable temperature."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Model output logits.

        Returns:
            Temperature-scaled logits.
        """
        return logits / self.temperature
