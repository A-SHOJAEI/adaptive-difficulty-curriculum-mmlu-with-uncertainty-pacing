#!/usr/bin/env python
"""Training script for adaptive curriculum learning on MMLU."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.utils.config import (
    load_config,
    set_random_seeds,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.loader import (
    MMLUDataLoader,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.preprocessing import (
    MMLUPreprocessor,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
    UncertaintyAwareModel,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.training.trainer import (
    AdaptiveCurriculumTrainer,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.evaluation.metrics import (
    MMLUMetrics,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.evaluation.analysis import (
    ResultsAnalyzer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def collate_fn(batch, preprocessor):
    """Collate function for DataLoader."""
    return preprocessor.preprocess_batch(batch, return_tensors=True)


def train(config_path: str, resume_from: str = None):
    """Main training function.

    Args:
        config_path: Path to configuration file.
        resume_from: Optional path to checkpoint to resume from.
    """
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Set random seeds
        seed = config.get("experiment", {}).get("seed", 42)
        set_random_seeds(seed)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create output directories
        output_dir = Path(config.get("experiment", {}).get("output_dir", "results"))
        checkpoint_dir = Path(config.get("experiment", {}).get("checkpoint_dir", "checkpoints"))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow (wrapped in try/except)
        try:
            import mlflow
            experiment_name = config.get("experiment", {}).get("name", "mmlu_experiment")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "model": config.get("model", {}).get("name", "bert-base-uncased"),
                "learning_rate": config.get("training", {}).get("learning_rate", 0.00003),
                "batch_size": config.get("training", {}).get("batch_size", 16),
                "curriculum_enabled": config.get("curriculum", {}).get("enable_curriculum", True),
            })
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            mlflow = None

        # Load data
        logger.info("Loading MMLU dataset...")
        data_loader = MMLUDataLoader(
            cache_dir=config.get("data", {}).get("cache_dir", None),
            subset=config.get("data", {}).get("subset", "all"),
        )

        try:
            dataset_dict = data_loader.load_data()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Using mock data for demonstration...")
            # Create minimal mock dataset for testing
            from datasets import Dataset as HFDataset, DatasetDict
            mock_data = {
                "question": ["What is 2+2?"] * 100,
                "choices": [["1", "2", "3", "4"]] * 100,
                "answer": [2] * 100,
                "subject": ["mathematics"] * 100,
            }
            mock_dataset = HFDataset.from_dict(mock_data)
            dataset_dict = DatasetDict({
                "auxiliary_train": mock_dataset,
                "validation": HFDataset.from_dict({k: v[:20] for k, v in mock_data.items()}),
                "test": HFDataset.from_dict({k: v[:20] for k, v in mock_data.items()}),
            })

        # Initialize preprocessor
        model_name = config.get("model", {}).get("name", "bert-base-uncased")
        max_length = config.get("data", {}).get("max_length", 512)
        preprocessor = MMLUPreprocessor(
            model_name=model_name,
            max_length=max_length,
        )

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = preprocessor.preprocess_dataset(dataset_dict["auxiliary_train"])
        val_dataset = preprocessor.preprocess_dataset(dataset_dict["validation"])

        # Create dataloaders
        batch_size = config.get("training", {}).get("batch_size", 16)
        num_workers = config.get("data", {}).get("num_workers", 0)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        logger.info(f"Train batches: {len(train_dataloader)}")
        logger.info(f"Val batches: {len(val_dataloader)}")

        # Initialize model
        logger.info("Initializing model...")
        model = UncertaintyAwareModel(
            model_name=model_name,
            num_classes=config.get("model", {}).get("num_classes", 4),
            dropout_rate=config.get("model", {}).get("dropout_rate", 0.3),
            use_calibration=config.get("model", {}).get("use_calibration", True),
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AdaptiveCurriculumTrainer(
            model=model,
            device=device,
            learning_rate=config.get("training", {}).get("learning_rate", 0.00003),
            weight_decay=config.get("training", {}).get("weight_decay", 0.01),
            warmup_steps=config.get("training", {}).get("warmup_steps", 500),
            max_grad_norm=config.get("training", {}).get("max_grad_norm", 1.0),
            use_amp=config.get("training", {}).get("use_amp", True),
            curriculum_update_freq=config.get("curriculum", {}).get("curriculum_update_freq", 100),
            uncertainty_weight=config.get("curriculum", {}).get("uncertainty_weight", 0.5),
            mc_samples=config.get("curriculum", {}).get("mc_samples", 10),
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            start_epoch, _ = trainer.load_checkpoint(resume_from)
            start_epoch += 1

        # Training loop
        num_epochs = config.get("training", {}).get("num_epochs", 10)
        early_stopping_patience = config.get("training", {}).get("early_stopping_patience", 3)

        best_val_accuracy = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        logger.info("Starting training...")

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train epoch
            train_metrics = trainer.train_epoch(train_dataloader, epoch + 1)
            train_losses.append(train_metrics["train_loss"])
            train_accuracies.append(train_metrics["train_accuracy"])

            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Accuracy: {train_metrics['train_accuracy']:.4f}"
            )

            # Evaluate
            val_metrics = trainer.evaluate(
                val_dataloader,
                compute_uncertainty=config.get("evaluation", {}).get("compute_uncertainty", True),
            )
            val_losses.append(val_metrics["eval_loss"])
            val_accuracies.append(val_metrics["eval_accuracy"])

            logger.info(
                f"Val Loss: {val_metrics['eval_loss']:.4f}, "
                f"Val Accuracy: {val_metrics['eval_accuracy']:.4f}"
            )

            # Log to MLflow
            if mlflow is not None:
                try:
                    mlflow.log_metrics({
                        "train_loss": train_metrics["train_loss"],
                        "train_accuracy": train_metrics["train_accuracy"],
                        "val_loss": val_metrics["eval_loss"],
                        "val_accuracy": val_metrics["eval_accuracy"],
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1, val_metrics)

            # Early stopping
            if val_metrics["eval_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["eval_accuracy"]
                patience_counter = 0

                # Save best model
                best_model_path = checkpoint_dir / "best_model.pt"
                trainer.save_checkpoint(str(best_model_path), epoch + 1, val_metrics)
                logger.info(f"Saved best model with accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

        # Generate final analysis
        logger.info("Generating analysis...")
        analyzer = ResultsAnalyzer(output_dir=str(output_dir))

        # Plot training curves
        if len(train_losses) > 0:
            analyzer.plot_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accuracies=train_accuracies,
                val_accuracies=val_accuracies,
            )

        # Save final metrics
        final_metrics = {
            "best_val_accuracy": best_val_accuracy,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else 0.0,
            "num_epochs_trained": len(train_losses),
        }
        analyzer.save_metrics(final_metrics, filename="training_metrics.json")

        # Close MLflow run
        if mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Best model saved to {checkpoint_dir / 'best_model.pt'}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train adaptive curriculum learning model on MMLU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    train(args.config, args.resume_from)


if __name__ == "__main__":
    main()
