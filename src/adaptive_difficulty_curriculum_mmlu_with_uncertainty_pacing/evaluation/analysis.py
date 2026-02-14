"""Results analysis and visualization for curriculum learning evaluation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for curriculum learning results with visualization.

    Generates plots and statistical analyses comparing curriculum learning
    approaches, including domain-wise performance and uncertainty evolution.
    """

    def __init__(self, output_dir: str = "results"):
        """Initialize results analyzer.

        Args:
            output_dir: Directory to save analysis results and plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = "metrics.json",
    ):
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save.
            filename: Output filename.
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        filename: str = "training_curves.png",
    ):
        """Plot training and validation curves.

        Args:
            train_losses: List of training losses per epoch.
            val_losses: List of validation losses per epoch.
            train_accuracies: List of training accuracies per epoch.
            val_accuracies: List of validation accuracies per epoch.
            filename: Output filename for plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
        ax2.plot(epochs, val_accuracies, label='Val Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {output_path}")

    def plot_domain_performance(
        self,
        domain_accuracies: Dict[str, float],
        filename: str = "domain_performance.png",
    ):
        """Plot per-domain accuracy comparison.

        Args:
            domain_accuracies: Dictionary mapping domains to accuracies.
            filename: Output filename for plot.
        """
        domains = list(domain_accuracies.keys())
        accuracies = list(domain_accuracies.values())

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        domains = [domains[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, max(6, len(domains) * 0.3)))

        colors = plt.cm.viridis(np.linspace(0, 1, len(domains)))
        ax.barh(domains, accuracies, color=colors)

        ax.set_xlabel('Accuracy')
        ax.set_title('Per-Domain Accuracy')
        ax.set_xlim(0, 1.0)

        # Add value labels
        for i, (domain, acc) in enumerate(zip(domains, accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved domain performance plot to {output_path}")

    def plot_uncertainty_distribution(
        self,
        uncertainties: List[float],
        correct_mask: Optional[List[bool]] = None,
        filename: str = "uncertainty_distribution.png",
    ):
        """Plot uncertainty distribution.

        Args:
            uncertainties: List of uncertainty values.
            correct_mask: Optional mask indicating correct predictions.
            filename: Output filename for plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if correct_mask is not None:
            uncertainties_array = np.array(uncertainties)
            correct_mask_array = np.array(correct_mask)

            ax.hist(
                uncertainties_array[correct_mask_array],
                bins=30,
                alpha=0.6,
                label='Correct Predictions',
                color='green',
            )
            ax.hist(
                uncertainties_array[~correct_mask_array],
                bins=30,
                alpha=0.6,
                label='Incorrect Predictions',
                color='red',
            )
            ax.legend()
        else:
            ax.hist(uncertainties, bins=30, alpha=0.7, color='blue')

        ax.set_xlabel('Uncertainty (Entropy)')
        ax.set_ylabel('Count')
        ax.set_title('Uncertainty Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved uncertainty distribution plot to {output_path}")

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        class_names: Optional[List[str]] = None,
        filename: str = "confusion_matrix.png",
    ):
        """Plot confusion matrix heatmap.

        Args:
            confusion_mat: Confusion matrix array.
            class_names: Optional names for classes.
            filename: Output filename for plot.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        if class_names is None:
            class_names = [f"Class {i}" for i in range(confusion_mat.shape[0])]

        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix to {output_path}")

    def generate_summary_report(
        self,
        metrics: Dict[str, float],
        filename: str = "summary_report.txt",
    ):
        """Generate text summary report.

        Args:
            metrics: Dictionary of computed metrics.
            filename: Output filename for report.
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MMLU Adaptive Curriculum Learning - Results Summary\n")
            f.write("=" * 80 + "\n\n")

            # Overall performance
            f.write("Overall Performance:\n")
            f.write("-" * 40 + "\n")
            for key in ["accuracy", "precision", "recall", "f1_score"]:
                if key in metrics:
                    f.write(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}\n")
            f.write("\n")

            # Domain statistics
            f.write("Domain Statistics:\n")
            f.write("-" * 40 + "\n")
            domain_keys = [
                "mean_domain_accuracy",
                "std_domain_accuracy",
                "min_domain_accuracy",
                "max_domain_accuracy",
                "low_resource_accuracy",
            ]
            for key in domain_keys:
                if key in metrics:
                    f.write(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}\n")
            f.write("\n")

            # Uncertainty statistics
            f.write("Uncertainty Statistics:\n")
            f.write("-" * 40 + "\n")
            uncertainty_keys = [
                "mean_uncertainty",
                "std_uncertainty",
                "uncertainty_correct",
                "uncertainty_incorrect",
                "uncertainty_error_correlation",
            ]
            for key in uncertainty_keys:
                if key in metrics:
                    f.write(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}\n")
            f.write("\n")

        logger.info(f"Generated summary report at {output_path}")
