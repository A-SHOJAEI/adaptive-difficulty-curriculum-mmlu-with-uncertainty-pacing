#!/usr/bin/env python
"""Evaluation script for trained MMLU model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
    monte_carlo_dropout_inference,
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


def evaluate(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    output_dir: str = "results",
):
    """Evaluate trained model on MMLU.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to configuration file.
        split: Dataset split to evaluate on ('validation' or 'test').
        output_dir: Directory to save evaluation results.
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

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading MMLU {split} dataset...")
        data_loader = MMLUDataLoader(
            cache_dir=config.get("data", {}).get("cache_dir", None),
            subset=config.get("data", {}).get("subset", "all"),
        )

        try:
            dataset_dict = data_loader.load_data()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Using mock data for demonstration...")
            from datasets import Dataset as HFDataset, DatasetDict
            mock_data = {
                "question": ["What is 2+2?"] * 50,
                "choices": [["1", "2", "3", "4"]] * 50,
                "answer": [2] * 50,
                "subject": ["mathematics"] * 50,
            }
            dataset_dict = DatasetDict({
                "test": HFDataset.from_dict(mock_data),
                "validation": HFDataset.from_dict(mock_data),
            })

        # Initialize preprocessor
        model_name = config.get("model", {}).get("name", "bert-base-uncased")
        max_length = config.get("data", {}).get("max_length", 512)
        preprocessor = MMLUPreprocessor(
            model_name=model_name,
            max_length=max_length,
        )

        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        split_map = {"validation": "validation", "test": "test"}
        eval_dataset = preprocessor.preprocess_dataset(dataset_dict[split_map[split]])

        # Create dataloader
        batch_size = config.get("training", {}).get("batch_size", 16)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(f"Evaluation batches: {len(eval_dataloader)}")

        # Initialize model
        logger.info("Initializing model...")
        model = UncertaintyAwareModel(
            model_name=model_name,
            num_classes=config.get("model", {}).get("num_classes", 4),
            dropout_rate=config.get("model", {}).get("dropout_rate", 0.3),
            use_calibration=config.get("model", {}).get("use_calibration", True),
        )

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Initialize metrics
        metrics_calculator = MMLUMetrics()

        # Evaluation loop
        logger.info("Running evaluation...")
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_domains = []

        compute_uncertainty = config.get("evaluation", {}).get("compute_uncertainty", True)
        mc_samples = config.get("curriculum", {}).get("mc_samples", 10)

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Get predictions
                if compute_uncertainty:
                    mean_preds, uncertainties = monte_carlo_dropout_inference(
                        model,
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                        num_samples=mc_samples,
                    )
                    predictions = torch.argmax(mean_preds, dim=-1)
                    all_uncertainties.extend(uncertainties.cpu().numpy())
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs["logits"], dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Update metrics
        metrics_calculator.update(
            predictions=all_predictions,
            labels=all_labels,
            uncertainties=all_uncertainties if compute_uncertainty else None,
        )

        # Compute all metrics
        logger.info("Computing metrics...")
        metrics = metrics_calculator.compute()

        # Log metrics
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_score']:.4f}")

        if compute_uncertainty:
            logger.info(f"\nUncertainty Statistics:")
            logger.info(f"  Mean Uncertainty: {metrics.get('mean_uncertainty', 0):.4f}")
            logger.info(f"  Std Uncertainty: {metrics.get('std_uncertainty', 0):.4f}")
            if 'uncertainty_correct' in metrics:
                logger.info(f"  Uncertainty (Correct): {metrics['uncertainty_correct']:.4f}")
            if 'uncertainty_incorrect' in metrics:
                logger.info(f"  Uncertainty (Incorrect): {metrics['uncertainty_incorrect']:.4f}")

        logger.info("="*80 + "\n")

        # Save metrics to JSON
        metrics_path = output_path / f"evaluation_metrics_{split}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Generate analysis
        logger.info("Generating analysis plots...")
        analyzer = ResultsAnalyzer(output_dir=str(output_path))

        # Save summary report
        analyzer.generate_summary_report(metrics, filename=f"evaluation_report_{split}.txt")

        # Plot confusion matrix
        confusion_mat = metrics_calculator.get_confusion_matrix()
        analyzer.plot_confusion_matrix(
            confusion_mat,
            class_names=["A", "B", "C", "D"],
            filename=f"confusion_matrix_{split}.png",
        )

        # Plot uncertainty distribution
        if compute_uncertainty and all_uncertainties:
            correct_mask = np.array(all_predictions) == np.array(all_labels)
            analyzer.plot_uncertainty_distribution(
                all_uncertainties,
                correct_mask=correct_mask,
                filename=f"uncertainty_distribution_{split}.png",
            )

        # Save predictions
        if config.get("evaluation", {}).get("save_predictions", True):
            predictions_data = {
                "predictions": [int(p) for p in all_predictions],
                "labels": [int(l) for l in all_labels],
                "uncertainties": [float(u) for u in all_uncertainties] if all_uncertainties else [],
            }
            predictions_path = output_path / f"predictions_{split}.json"
            with open(predictions_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            logger.info(f"Saved predictions to {predictions_path}")

        logger.info(f"\nEvaluation complete! Results saved to {output_path}")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained MMLU model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
