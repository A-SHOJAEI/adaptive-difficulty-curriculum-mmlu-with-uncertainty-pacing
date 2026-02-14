"""Evaluation metrics for MMLU including domain-specific analysis."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class MMLUMetrics:
    """Comprehensive metrics for MMLU evaluation.

    Computes overall accuracy, per-domain accuracy, low-resource domain
    improvement, and other metrics specific to curriculum learning evaluation.
    """

    def __init__(self, domain_names: Optional[List[str]] = None):
        """Initialize MMLU metrics.

        Args:
            domain_names: List of domain names for per-domain analysis.
        """
        self.domain_names = domain_names or []
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.labels = []
        self.domains = []
        self.uncertainties = []

    def update(
        self,
        predictions: List[int],
        labels: List[int],
        domains: Optional[List[str]] = None,
        uncertainties: Optional[List[float]] = None,
    ):
        """Update metrics with new predictions.

        Args:
            predictions: List of predicted class indices.
            labels: List of ground truth labels.
            domains: Optional list of domain names for each sample.
            uncertainties: Optional uncertainty estimates for each sample.
        """
        self.predictions.extend(predictions)
        self.labels.extend(labels)

        if domains is not None:
            self.domains.extend(domains)

        if uncertainties is not None:
            self.uncertainties.extend(uncertainties)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary containing computed metrics.
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to compute metrics")
            return {}

        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = accuracy_score(self.labels, self.predictions)

        # Precision, recall, F1 (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels,
            self.predictions,
            average="macro",
            zero_division=0,
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = (
            precision_recall_fscore_support(
                self.labels,
                self.predictions,
                average=None,
                zero_division=0,
            )
        )

        for i in range(len(per_class_precision)):
            metrics[f"class_{i}_precision"] = per_class_precision[i]
            metrics[f"class_{i}_recall"] = per_class_recall[i]
            metrics[f"class_{i}_f1"] = per_class_f1[i]

        # Per-domain accuracy
        if self.domains:
            domain_metrics = self._compute_domain_metrics()
            metrics.update(domain_metrics)

        # Uncertainty metrics
        if self.uncertainties:
            uncertainty_metrics = self._compute_uncertainty_metrics()
            metrics.update(uncertainty_metrics)

        logger.info(f"Computed metrics: accuracy={metrics['accuracy']:.4f}")
        return metrics

    def _compute_domain_metrics(self) -> Dict[str, float]:
        """Compute per-domain accuracy and statistics.

        Returns:
            Dictionary with domain-specific metrics.
        """
        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)

        for pred, label, domain in zip(self.predictions, self.labels, self.domains):
            domain_total[domain] += 1
            if pred == label:
                domain_correct[domain] += 1

        metrics = {}
        domain_accuracies = []

        for domain in domain_total:
            accuracy = domain_correct[domain] / domain_total[domain]
            metrics[f"domain_{domain}_accuracy"] = accuracy
            domain_accuracies.append(accuracy)

        # Aggregate domain metrics
        if domain_accuracies:
            metrics["mean_domain_accuracy"] = np.mean(domain_accuracies)
            metrics["std_domain_accuracy"] = np.std(domain_accuracies)
            metrics["min_domain_accuracy"] = np.min(domain_accuracies)
            metrics["max_domain_accuracy"] = np.max(domain_accuracies)

            # Low-resource domain performance (bottom 25%)
            sorted_accuracies = sorted(domain_accuracies)
            low_resource_threshold = int(len(sorted_accuracies) * 0.25)
            if low_resource_threshold > 0:
                low_resource_accuracies = sorted_accuracies[:low_resource_threshold]
                metrics["low_resource_accuracy"] = np.mean(low_resource_accuracies)

        return metrics

    def _compute_uncertainty_metrics(self) -> Dict[str, float]:
        """Compute uncertainty-related metrics.

        Returns:
            Dictionary with uncertainty metrics.
        """
        metrics = {}

        uncertainties_array = np.array(self.uncertainties)
        predictions_array = np.array(self.predictions)
        labels_array = np.array(self.labels)

        # Mean uncertainty
        metrics["mean_uncertainty"] = float(np.mean(uncertainties_array))
        metrics["std_uncertainty"] = float(np.std(uncertainties_array))

        # Uncertainty for correct vs incorrect predictions
        correct_mask = predictions_array == labels_array
        incorrect_mask = ~correct_mask

        if correct_mask.sum() > 0:
            metrics["uncertainty_correct"] = float(
                np.mean(uncertainties_array[correct_mask])
            )

        if incorrect_mask.sum() > 0:
            metrics["uncertainty_incorrect"] = float(
                np.mean(uncertainties_array[incorrect_mask])
            )

        # Uncertainty calibration (correlation with errors)
        if len(uncertainties_array) > 1:
            errors = (predictions_array != labels_array).astype(float)
            correlation = np.corrcoef(uncertainties_array, errors)[0, 1]
            metrics["uncertainty_error_correlation"] = float(correlation)

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix.

        Returns:
            Confusion matrix as numpy array.
        """
        return confusion_matrix(self.labels, self.predictions)
