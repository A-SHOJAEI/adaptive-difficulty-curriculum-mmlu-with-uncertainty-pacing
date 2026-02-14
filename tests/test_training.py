"""Tests for training components."""

import pytest
import torch
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.training.trainer import (
    AdaptiveCurriculumTrainer,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
    UncertaintyAwareModel,
)


class TestAdaptiveCurriculumTrainer:
    """Test cases for adaptive curriculum trainer."""

    def test_initialization(self, device):
        """Test trainer initialization."""
        model = UncertaintyAwareModel(num_classes=4)

        trainer = AdaptiveCurriculumTrainer(
            model=model,
            device=device,
            learning_rate=0.0001,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None

    def test_training_step(self, device, sample_batch):
        """Test single training step."""
        model = UncertaintyAwareModel(num_classes=4)
        trainer = AdaptiveCurriculumTrainer(
            model=model,
            device=device,
            use_amp=False,  # Disable AMP for testing
        )

        # Prepare batch
        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)
        labels = sample_batch["labels"].to(device)

        # Forward pass
        outputs = trainer.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        uncertainties = torch.zeros(labels.size(0), device=device)
        loss = trainer.criterion(outputs["logits"], labels, uncertainties)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_checkpoint_save_load(self, device, tmp_path):
        """Test checkpoint saving and loading."""
        model = UncertaintyAwareModel(num_classes=4)
        trainer = AdaptiveCurriculumTrainer(model=model, device=device)

        # Save checkpoint
        checkpoint_path = str(tmp_path / "test_checkpoint.pt")
        metrics = {"accuracy": 0.75, "loss": 0.5}
        trainer.save_checkpoint(checkpoint_path, epoch=1, metrics=metrics)

        # Create new trainer and load
        model_new = UncertaintyAwareModel(num_classes=4)
        trainer_new = AdaptiveCurriculumTrainer(model=model_new, device=device)

        epoch, loaded_metrics = trainer_new.load_checkpoint(checkpoint_path)

        assert epoch == 1
        assert loaded_metrics["accuracy"] == 0.75
