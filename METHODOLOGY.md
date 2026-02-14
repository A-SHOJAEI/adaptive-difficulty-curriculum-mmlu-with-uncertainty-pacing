# Methodology

## Overview

This project implements a novel adaptive curriculum learning approach for the MMLU benchmark that uses epistemic uncertainty quantification to dynamically adjust task difficulty ordering during training.

## Key Innovation

Traditional curriculum learning uses fixed difficulty schedules based on heuristics or predefined task orderings. This approach dynamically adapts the curriculum based on real-time uncertainty estimates, allowing the model to focus computational resources on domains where it is most uncertain.

## Technical Components

### 1. Uncertainty Quantification via Monte Carlo Dropout

**Implementation**: `src/models/components.py:monte_carlo_dropout_inference()`

The model estimates epistemic uncertainty by:
- Enabling dropout during inference (contrary to standard practice)
- Performing multiple forward passes (default: 10 samples)
- Computing prediction variance across samples
- Measuring uncertainty as predictive entropy

**Mathematical formulation**:
```
Uncertainty(x) = -Σ p(y|x) log p(y|x)
where p(y|x) = (1/T) Σ_t p(y|x, θ_t)
```

### 2. Adaptive Curriculum Loss

**Implementation**: `src/models/components.py:CurriculumLoss`

The loss function combines standard cross-entropy with uncertainty-based sample reweighting:

```
L(θ) = E[(1 + α·σ(u/τ)) · CE(y, ŷ)]
```

Where:
- `α` is the uncertainty weight (default: 0.5)
- `u` is the uncertainty estimate
- `τ` is the temperature parameter (default: 2.0)
- `σ` is the sigmoid function for normalization

This ensures higher-uncertainty samples receive more training attention.

### 3. Dynamic Curriculum Pacing

**Implementation**: `src/training/trainer.py:AdaptiveCurriculumTrainer`

The trainer periodically (default: every 100 steps):
1. Evaluates model uncertainty on current batch
2. Computes domain-level uncertainty aggregates
3. Adjusts sample weights for next training steps
4. Updates learning curriculum based on uncertainty distribution

## Training Pipeline

### Stage 1: Initialization
- Load pretrained BERT-base model
- Initialize uncertainty quantification head
- Set up adaptive curriculum loss

### Stage 2: Adaptive Training
- For each batch:
  - Estimate sample uncertainties via MC dropout
  - Weight samples by uncertainty
  - Compute curriculum-weighted loss
  - Update model parameters
  - Track domain-level uncertainty evolution

### Stage 3: Evaluation
- Compute per-domain accuracy
- Measure low-resource domain improvement
- Analyze uncertainty calibration
- Generate comparative visualizations

## Ablation Studies

The project includes two configurations for controlled comparison:

### Full Model (`configs/default.yaml`)
- Adaptive curriculum enabled
- Uncertainty weight: 0.5
- MC dropout samples: 10
- Dropout rate: 0.3
- Confidence calibration enabled

### Baseline (`configs/ablation.yaml`)
- No adaptive curriculum (uncertainty weight: 0.0)
- Single forward pass (MC samples: 1)
- Lower dropout rate: 0.1
- No confidence calibration

This allows quantifying the contribution of each component to final performance.

## Expected Outcomes

Based on curriculum learning literature, we expect:

1. **Higher MMLU Average**: 2-5% improvement over baseline through better knowledge transfer
2. **Low-Resource Gains**: 10-15% improvement on domains with <100 training samples
3. **Training Efficiency**: Reach target accuracy 20-25% faster than uniform sampling
4. **Better Calibration**: Improved alignment between confidence and accuracy

## Reproducibility

All experiments use:
- Fixed random seed (42)
- Deterministic CUDA operations
- Logged hyperparameters (MLflow when available)
- Checkpoint saving at each epoch
- Comprehensive metrics logging

Run `python scripts/train.py` to reproduce results.

## References

This work builds on:
- Monte Carlo Dropout (Gal & Ghahramani, 2016)
- Curriculum Learning (Bengio et al., 2009)
- MMLU Benchmark (Hendrycks et al., 2021)
- Uncertainty-based active learning principles
