"""Tests for model components."""

import pytest
import torch
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
    UncertaintyAwareModel,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
    UncertaintyQuantificationHead,
    CurriculumLoss,
    monte_carlo_dropout_inference,
    ConfidenceCalibrationLayer,
)


class TestUncertaintyAwareModel:
    """Test cases for uncertainty-aware model."""

    def test_initialization(self):
        """Test model initialization."""
        model = UncertaintyAwareModel(
            model_name="bert-base-uncased",
            num_classes=4,
        )
        assert model.num_classes == 4
        assert model.uncertainty_head is not None

    def test_forward_pass(self, sample_batch, device):
        """Test forward pass."""
        model = UncertaintyAwareModel(num_classes=4)
        model.to(device)

        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape == (input_ids.shape[0], 4)

    def test_forward_with_labels(self, sample_batch, device):
        """Test forward pass with labels."""
        model = UncertaintyAwareModel(num_classes=4)
        model.to(device)

        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)
        labels = sample_batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert "loss" in outputs
        assert outputs["loss"].item() > 0


class TestUncertaintyQuantificationHead:
    """Test cases for uncertainty quantification head."""

    def test_initialization(self):
        """Test head initialization."""
        head = UncertaintyQuantificationHead(
            input_dim=768,
            num_classes=4,
            dropout_rate=0.3,
        )
        assert head.num_classes == 4
        assert head.dropout_rate == 0.3

    def test_forward(self):
        """Test forward pass."""
        head = UncertaintyQuantificationHead(input_dim=768, num_classes=4)
        x = torch.randn(8, 768)

        output = head(x)

        assert output.shape == (8, 4)

    def test_dropout_enabled(self):
        """Test that dropout can be enabled."""
        head = UncertaintyQuantificationHead(input_dim=768, num_classes=4)
        head.eval()
        head.enable_dropout()

        # Check that dropout modules are in training mode
        dropout_in_train_mode = False
        for module in head.modules():
            if isinstance(module, torch.nn.Dropout):
                if module.training:
                    dropout_in_train_mode = True
                    break

        assert dropout_in_train_mode


class TestCurriculumLoss:
    """Test cases for curriculum loss function."""

    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = CurriculumLoss(uncertainty_weight=0.5)
        assert loss_fn.uncertainty_weight == 0.5

    def test_forward(self):
        """Test loss computation."""
        loss_fn = CurriculumLoss()

        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        uncertainties = torch.rand(8)

        loss = loss_fn(logits, labels, uncertainties)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_uncertainty_weighting(self):
        """Test that uncertainty affects loss."""
        loss_fn = CurriculumLoss(uncertainty_weight=1.0)

        logits = torch.randn(4, 4)
        labels = torch.randint(0, 4, (4,))

        # High uncertainty should increase loss weight
        high_uncertainties = torch.ones(4) * 10.0
        low_uncertainties = torch.ones(4) * 0.1

        loss_high = loss_fn(logits, labels, high_uncertainties)
        loss_low = loss_fn(logits, labels, low_uncertainties)

        # With positive uncertainty weight, high uncertainty should give higher loss
        assert loss_high.item() >= loss_low.item()


class TestMonteCarloDropout:
    """Test cases for Monte Carlo dropout inference."""

    def test_mc_dropout_inference(self, sample_batch, device):
        """Test MC dropout inference."""
        model = UncertaintyAwareModel(num_classes=4)
        model.to(device)

        inputs = {
            "input_ids": sample_batch["input_ids"].to(device),
            "attention_mask": sample_batch["attention_mask"].to(device),
        }

        mean_preds, uncertainties = monte_carlo_dropout_inference(
            model, inputs, num_samples=5
        )

        assert mean_preds.shape == (inputs["input_ids"].shape[0], 4)
        assert uncertainties.shape == (inputs["input_ids"].shape[0],)
        assert torch.all(uncertainties >= 0)


class TestConfidenceCalibrationLayer:
    """Test cases for confidence calibration layer."""

    def test_initialization(self):
        """Test calibration layer initialization."""
        layer = ConfidenceCalibrationLayer()
        assert layer.temperature is not None

    def test_forward(self):
        """Test temperature scaling."""
        layer = ConfidenceCalibrationLayer()
        logits = torch.randn(4, 4)

        scaled_logits = layer(logits)

        assert scaled_logits.shape == logits.shape
