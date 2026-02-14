#!/usr/bin/env python
"""Verification script to check project setup."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_imports():
    """Check that all imports work."""
    print("Checking imports...")

    try:
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.model import (
            UncertaintyAwareModel,
        )
        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False

    try:
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.models.components import (
            UncertaintyQuantificationHead,
            CurriculumLoss,
            monte_carlo_dropout_inference,
        )
        print("✓ Components imports successful")
    except ImportError as e:
        print(f"✗ Components import failed: {e}")
        return False

    try:
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.training.trainer import (
            AdaptiveCurriculumTrainer,
        )
        print("✓ Trainer imports successful")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        return False

    try:
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.loader import (
            MMLUDataLoader,
        )
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.data.preprocessing import (
            MMLUPreprocessor,
        )
        print("✓ Data imports successful")
    except ImportError as e:
        print(f"✗ Data import failed: {e}")
        return False

    try:
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.evaluation.metrics import (
            MMLUMetrics,
        )
        from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.evaluation.analysis import (
            ResultsAnalyzer,
        )
        print("✓ Evaluation imports successful")
    except ImportError as e:
        print(f"✗ Evaluation import failed: {e}")
        return False

    return True


def check_configs():
    """Check that config files are valid."""
    print("\nChecking configuration files...")

    import yaml

    try:
        with open("configs/default.yaml") as f:
            config = yaml.safe_load(f)

        # Check key fields
        assert "model" in config
        assert "training" in config
        assert "curriculum" in config
        assert config["training"]["learning_rate"] == 0.00003
        print("✓ default.yaml is valid")
    except Exception as e:
        print(f"✗ default.yaml check failed: {e}")
        return False

    try:
        with open("configs/ablation.yaml") as f:
            config = yaml.safe_load(f)

        assert config["curriculum"]["enable_curriculum"] == False
        print("✓ ablation.yaml is valid")
    except Exception as e:
        print(f"✗ ablation.yaml check failed: {e}")
        return False

    return True


def check_file_structure():
    """Check that all required files exist."""
    print("\nChecking file structure...")

    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "configs/default.yaml",
        "configs/ablation.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/predict.py",
        "tests/test_data.py",
        "tests/test_model.py",
        "tests/test_training.py",
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"✗ Missing file: {file_path}")
            return False

    print(f"✓ All {len(required_files)} required files present")
    return True


def check_dependencies():
    """Check that required packages are importable."""
    print("\nChecking dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "sklearn",
        "yaml",
        "tqdm",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print(f"✓ All {len(required_packages)} core dependencies available")
    return True


def main():
    """Run all checks."""
    print("="*80)
    print("Project Setup Verification")
    print("="*80)

    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Configuration Files", check_configs),
        ("Python Imports", check_imports),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append(False)

    print("\n" + "="*80)
    if all(results):
        print("✓ All checks passed! Project is ready to use.")
        print("="*80)
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: pytest tests/")
        print("  3. Start training: python scripts/train.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
