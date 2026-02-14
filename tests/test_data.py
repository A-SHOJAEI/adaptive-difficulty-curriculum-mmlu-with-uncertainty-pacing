"""Tests for data loading and preprocessing."""

import pytest
import torch
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.loader import (
    MMLUDataLoader,
)
from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.preprocessing import (
    MMLUPreprocessor,
)


class TestMMLUDataLoader:
    """Test cases for MMLU data loader."""

    def test_initialization(self):
        """Test loader initialization."""
        loader = MMLUDataLoader()
        assert loader.subset == "all"
        assert loader.dataset_dict is None

    def test_format_question(self, mock_mmlu_samples):
        """Test question formatting."""
        preprocessor = MMLUPreprocessor()
        formatted = preprocessor.format_question(mock_mmlu_samples[0])

        assert "Question:" in formatted
        assert "Choices:" in formatted
        assert "Answer:" in formatted
        assert "2 + 2" in formatted


class TestMMLUPreprocessor:
    """Test cases for MMLU preprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = MMLUPreprocessor(max_length=256)
        assert preprocessor.max_length == 256
        assert preprocessor.tokenizer is not None

    def test_preprocess_batch(self, mock_mmlu_samples):
        """Test batch preprocessing."""
        preprocessor = MMLUPreprocessor()
        batch_output = preprocessor.preprocess_batch(mock_mmlu_samples)

        assert "input_ids" in batch_output
        assert "attention_mask" in batch_output
        assert "labels" in batch_output
        assert isinstance(batch_output["input_ids"], torch.Tensor)
        assert batch_output["labels"].shape[0] == len(mock_mmlu_samples)

    def test_format_consistency(self, mock_mmlu_samples):
        """Test consistent formatting."""
        preprocessor = MMLUPreprocessor()

        formatted1 = preprocessor.format_question(mock_mmlu_samples[0])
        formatted2 = preprocessor.format_question(mock_mmlu_samples[0])

        assert formatted1 == formatted2
