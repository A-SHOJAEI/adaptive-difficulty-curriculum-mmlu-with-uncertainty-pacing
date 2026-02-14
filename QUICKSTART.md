# Quick Start Guide

Get started with adaptive curriculum learning on MMLU in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, Transformers, and other required packages.

## 2. Verify Installation

```bash
python scripts/verify_setup.py
```

Expected output: "All checks passed! Project is ready to use."

## 3. Run Tests (Optional)

```bash
pytest tests/ -v
```

Verify that all components work correctly.

## 4. Train Model

### Quick Demo (Small Scale)

For a quick test, modify `configs/default.yaml`:
```yaml
training:
  num_epochs: 2
  batch_size: 8
```

Then run:
```bash
python scripts/train.py --config configs/default.yaml
```

### Full Training

Use default configuration for full training:
```bash
python scripts/train.py
```

Training progress will be logged to console. Best model saved to `checkpoints/best_model.pt`.

## 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split test
```

Results saved to `results/` directory with:
- `evaluation_metrics_test.json`: All computed metrics
- `confusion_matrix_test.png`: Confusion matrix visualization
- `uncertainty_distribution_test.png`: Uncertainty analysis
- `evaluation_report_test.txt`: Text summary

## 6. Make Predictions

### Interactive Mode

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

Enter questions and choices when prompted.

### Batch Predictions

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --input examples/sample_questions.json \
    --output predictions.json
```

## Common Commands

### Train baseline (no curriculum)
```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluate on validation set
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split validation
```

### Resume training
```bash
python scripts/train.py --resume-from checkpoints/checkpoint_epoch_5.pt
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

```yaml
training:
  learning_rate: 0.00003      # Initial learning rate
  batch_size: 16               # Batch size
  num_epochs: 10               # Number of epochs

curriculum:
  uncertainty_weight: 0.5      # Weight for uncertainty in loss
  mc_samples: 10               # MC dropout samples
  enable_curriculum: true      # Enable/disable curriculum
```

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```yaml
training:
  batch_size: 8
```

### Dataset Download Fails
Check internet connection. Dataset will be cached in `./cache/` directory.

### MLflow Errors
MLflow errors are expected if server not running. They are safely caught and logged.

## Next Steps

1. **Experiment with hyperparameters**: Edit `configs/default.yaml`
2. **Run ablation study**: Compare default.yaml vs ablation.yaml
3. **Analyze results**: Check plots in `results/` directory
4. **Extend the model**: Add custom components in `src/models/components.py`

## Support

For issues, check:
- README.md for detailed documentation
- METHODOLOGY.md for technical details
- PROJECT_SUMMARY.md for complete overview
