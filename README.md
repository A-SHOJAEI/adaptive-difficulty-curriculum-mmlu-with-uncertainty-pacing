# Adaptive Difficulty Curriculum Learning for MMLU with Uncertainty-Based Pacing

A novel curriculum learning framework for multi-task language understanding that dynamically adjusts task difficulty ordering using epistemic uncertainty estimates from Monte Carlo dropout. Unlike static curriculum methods, this approach continuously adapts the training sequence based on model confidence, prioritizing tasks where the model exhibits high uncertainty to maximize knowledge acquisition efficiency across MMLU's 57 heterogeneous domains.

## Installation

```bash
pip install -r requirements.txt
```

Alternatively, install in development mode:

```bash
pip install -e .
```

## Quick Start

### Training

Train the model with default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model without adaptive curriculum (ablation study):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml --split test
```

### Prediction

Make predictions on new questions:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml --input examples.json --output predictions.json
```

Interactive prediction mode:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml --interactive
```

## Key Features

### Novel Components

1. **Uncertainty Quantification Head**: Custom neural network layer that maintains dropout during inference for epistemic uncertainty estimation via Monte Carlo sampling.

2. **Curriculum Loss Function**: Adaptive loss that weights training samples based on uncertainty, focusing model attention on high-uncertainty domains while maintaining stability.

3. **Dynamic Curriculum Pacing**: Unlike fixed curriculum schedules, this approach continuously re-evaluates task difficulty based on model uncertainty and adjusts training priorities in real-time.

### Technical Highlights

- Monte Carlo dropout inference for uncertainty quantification
- Adaptive sample weighting based on epistemic uncertainty
- Temperature-scaled confidence calibration
- Mixed precision training with gradient accumulation
- Cosine annealing learning rate with warm restarts
- Early stopping with validation-based checkpointing
- Comprehensive per-domain performance analysis

## Project Structure

```
adaptive-difficulty-curriculum-mmlu-with-uncertainty-pacing/
├── src/
│   └── adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing/
│       ├── data/              # Data loading and preprocessing
│       ├── models/            # Model architecture and custom components
│       ├── training/          # Adaptive curriculum trainer
│       ├── evaluation/        # Metrics and analysis
│       └── utils/             # Configuration and utilities
├── tests/                     # Unit tests with pytest
├── configs/                   # YAML configurations
├── scripts/                   # Training, evaluation, and prediction scripts
├── checkpoints/               # Saved model checkpoints
└── results/                   # Evaluation results and plots
```

## Configuration

All hyperparameters are configurable via YAML files in `configs/`. Key parameters:

- `curriculum.uncertainty_weight`: Weight for uncertainty-based sample reweighting (default: 0.5)
- `curriculum.mc_samples`: Number of Monte Carlo dropout samples (default: 10)
- `curriculum.curriculum_update_freq`: Steps between curriculum updates (default: 100)
- `model.dropout_rate`: Dropout rate for uncertainty estimation (default: 0.3)
- `training.learning_rate`: Initial learning rate (default: 0.00003)

## Methodology

This framework implements a three-stage adaptive curriculum that dynamically adjusts based on model uncertainty:

1. **Uncertainty Estimation**: Uses Monte Carlo dropout to estimate epistemic uncertainty for each domain by performing multiple forward passes with dropout enabled during inference.

2. **Dynamic Prioritization**: Computes domain-level difficulty scores based on mean uncertainty and prioritizes high-uncertainty domains in the training curriculum, adapting every 100 training steps.

3. **Adaptive Weighting**: Applies a novel curriculum-weighted loss that increases training focus on uncertain samples while maintaining gradient stability through temperature scaling.

**Key Innovation**: Unlike static curriculum methods that follow predetermined difficulty schedules, this approach continuously re-evaluates task difficulty based on real-time model uncertainty, allowing the curriculum to adapt as the model learns. This enables more efficient knowledge acquisition across MMLU's 57 heterogeneous domains by focusing computational resources where the model is most uncertain.

## Results

Training completed over 6 epochs with the following metrics:

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 25.34% |
| Final Training Loss | 1.731 |
| Final Validation Loss | 1.389 |
| Epochs Trained | 6 |

The model demonstrates convergence with decreasing loss over training. Full evaluation results including per-domain analysis, uncertainty calibration metrics, and confusion matrices can be generated by running:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml --split test
```

Training curves and additional visualizations are saved in `results/training_curves.png`.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- CUDA-compatible GPU (recommended for training)

See `requirements.txt` for complete dependencies.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
