"""Adaptive curriculum trainer with uncertainty-based pacing."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
    CurriculumLoss,
    monte_carlo_dropout_inference,
)

logger = logging.getLogger(__name__)


class AdaptiveCurriculumTrainer:
    """Trainer implementing adaptive curriculum learning with uncertainty-based pacing.

    This novel training approach dynamically reorders training samples based on
    epistemic uncertainty estimates, prioritizing domains where the model exhibits
    high uncertainty to maximize learning efficiency.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.00003,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        curriculum_update_freq: int = 100,
        uncertainty_weight: float = 0.5,
        mc_samples: int = 10,
    ):
        """Initialize adaptive curriculum trainer.

        Args:
            model: Model to train.
            device: Device to train on (cuda/cpu).
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for AdamW optimizer.
            warmup_steps: Number of warmup steps for learning rate.
            max_grad_norm: Maximum gradient norm for clipping.
            use_amp: Whether to use automatic mixed precision.
            curriculum_update_freq: Steps between curriculum updates.
            uncertainty_weight: Weight for uncertainty-based loss.
            mc_samples: Number of MC dropout samples for uncertainty.
        """
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.curriculum_update_freq = curriculum_update_freq
        self.mc_samples = mc_samples

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Initialize curriculum loss
        self.criterion = CurriculumLoss(
            uncertainty_weight=uncertainty_weight,
        )

        # Learning rate scheduler with warmup
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            T_mult=2,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.domain_uncertainties: Dict[str, float] = {}

        logger.info("Initialized AdaptiveCurriculumTrainer")

    def train_epoch(
        self,
        train_dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with adaptive curriculum.

        Args:
            train_dataloader: DataLoader for training data.
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Estimate uncertainties for curriculum weighting
            if self.global_step % self.curriculum_update_freq == 0:
                uncertainties = self._estimate_batch_uncertainties(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
            else:
                # Use zeros when not updating curriculum
                uncertainties = torch.zeros(labels.size(0), device=self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = self.criterion(
                        outputs["logits"],
                        labels,
                        uncertainties,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.criterion(
                    outputs["logits"],
                    labels,
                    uncertainties,
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()

            predictions = torch.argmax(outputs["logits"], dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": total_correct / total_samples,
                "lr": self.scheduler.get_last_lr()[0],
            })

        metrics = {
            "train_loss": total_loss / len(train_dataloader),
            "train_accuracy": total_correct / total_samples,
        }

        return metrics

    def evaluate(
        self,
        eval_dataloader,
        compute_uncertainty: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model on validation/test set.

        Args:
            eval_dataloader: DataLoader for evaluation data.
            compute_uncertainty: Whether to compute uncertainty estimates.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_uncertainties = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Standard forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs["loss"].item()

                predictions = torch.argmax(outputs["logits"], dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                # Compute uncertainty if requested
                if compute_uncertainty:
                    _, uncertainties = monte_carlo_dropout_inference(
                        self.model,
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                        num_samples=self.mc_samples,
                    )
                    all_uncertainties.extend(uncertainties.cpu().numpy())

        metrics = {
            "eval_loss": total_loss / len(eval_dataloader),
            "eval_accuracy": total_correct / total_samples,
        }

        if compute_uncertainty:
            metrics["mean_uncertainty"] = np.mean(all_uncertainties)
            metrics["std_uncertainty"] = np.std(all_uncertainties)

        return metrics

    def _estimate_batch_uncertainties(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate uncertainties for a batch using MC dropout.

        Args:
            inputs: Dictionary with input_ids and attention_mask.

        Returns:
            Uncertainty estimates for each sample in batch.
        """
        _, uncertainties = monte_carlo_dropout_inference(
            self.model,
            inputs,
            num_samples=self.mc_samples,
        )
        return uncertainties

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint.
            epoch: Current epoch number.
            metrics: Current metrics dictionary.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"], checkpoint["metrics"]
