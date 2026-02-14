"""Pytest fixtures for testing."""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    batch_size = 4
    seq_length = 128

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
        "labels": torch.randint(0, 4, (batch_size,)),
        "subjects": ["math", "history", "science", "literature"],
    }


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "model": {
            "name": "bert-base-uncased",
            "num_classes": 4,
            "dropout_rate": 0.3,
        },
        "training": {
            "learning_rate": 0.00003,
            "batch_size": 16,
            "num_epochs": 3,
            "warmup_steps": 100,
        },
        "data": {
            "max_length": 512,
            "cache_dir": "./cache",
        },
    }


@pytest.fixture
def tokenizer():
    """Load tokenizer for testing."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def mock_mmlu_samples():
    """Create mock MMLU samples."""
    return [
        {
            "question": "What is 2 + 2?",
            "choices": ["2", "3", "4", "5"],
            "answer": 2,
            "subject": "mathematics",
        },
        {
            "question": "Who wrote Hamlet?",
            "choices": ["Dickens", "Shakespeare", "Austen", "Hemingway"],
            "answer": 1,
            "subject": "literature",
        },
        {
            "question": "What is H2O?",
            "choices": ["Oxygen", "Hydrogen", "Water", "Carbon"],
            "answer": 2,
            "subject": "chemistry",
        },
    ]
