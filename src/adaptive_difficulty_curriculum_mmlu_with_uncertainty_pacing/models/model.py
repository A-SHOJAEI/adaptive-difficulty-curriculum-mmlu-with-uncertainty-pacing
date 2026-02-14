"""Core model implementation with uncertainty-aware architecture."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
    UncertaintyQuantificationHead,
    ConfidenceCalibrationLayer,
)

logger = logging.getLogger(__name__)


class UncertaintyAwareModel(nn.Module):
    """Transformer-based model with uncertainty quantification for MMLU.

    Combines a pretrained language model backbone with a custom uncertainty
    quantification head that enables epistemic uncertainty estimation via
    Monte Carlo dropout.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        use_calibration: bool = True,
    ):
        """Initialize uncertainty-aware model.

        Args:
            model_name: Name of pretrained transformer model.
            num_classes: Number of output classes (4 for MMLU multiple choice).
            dropout_rate: Dropout rate for uncertainty estimation.
            use_calibration: Whether to use confidence calibration.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Custom uncertainty quantification head
        self.uncertainty_head = UncertaintyQuantificationHead(
            input_dim=self.config.hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Optional calibration layer
        self.use_calibration = use_calibration
        if use_calibration:
            self.calibration_layer = ConfidenceCalibrationLayer()

        logger.info(f"Initialized UncertaintyAwareModel with backbone={model_name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Optional labels of shape (batch_size,).

        Returns:
            Dictionary with logits, loss (if labels provided), and hidden states.
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Pass through uncertainty head
        logits = self.uncertainty_head(pooled_output)

        # Apply calibration if enabled
        if self.use_calibration:
            logits = self.calibration_layer(logits)

        result = {
            "logits": logits,
            "hidden_states": pooled_output,
        }

        # Compute loss if labels provided
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            result["loss"] = loss

        return result

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract embeddings from backbone.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Embeddings of shape (batch_size, hidden_size).
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def freeze_backbone(self):
        """Freeze backbone parameters for efficient fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze backbone parameters")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze backbone parameters")
