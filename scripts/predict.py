#!/usr/bin/env python
"""Prediction script for inference on new MMLU questions."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from adaptive_difficulty_curriculum_mmlu_with_uncertainty_pacing.utils.config import (
    load_config,
    set_random_seeds,
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def predict_single(
    model: UncertaintyAwareModel,
    preprocessor: MMLUPreprocessor,
    question: str,
    choices: List[str],
    device: torch.device,
    compute_uncertainty: bool = True,
    mc_samples: int = 10,
) -> Dict:
    """Make prediction on a single question.

    Args:
        model: Trained model.
        preprocessor: Data preprocessor.
        question: Question text.
        choices: List of answer choices.
        device: Device to run inference on.
        compute_uncertainty: Whether to compute uncertainty.
        mc_samples: Number of MC samples for uncertainty.

    Returns:
        Dictionary with prediction, probabilities, and uncertainty.
    """
    # Format sample
    sample = {
        "question": question,
        "choices": choices,
        "answer": 0,  # Dummy value
        "subject": "unknown",
    }

    # Preprocess
    batch = preprocessor.preprocess_batch([sample], return_tensors=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Get prediction
    model.eval()

    with torch.no_grad():
        if compute_uncertainty:
            mean_probs, uncertainties = monte_carlo_dropout_inference(
                model,
                {"input_ids": input_ids, "attention_mask": attention_mask},
                num_samples=mc_samples,
            )
            prediction = torch.argmax(mean_probs, dim=-1).item()
            probabilities = mean_probs[0].cpu().numpy()
            uncertainty = uncertainties[0].item()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)[0].cpu().numpy()
            prediction = int(np.argmax(probabilities))
            uncertainty = None

    # Format result
    result = {
        "question": question,
        "choices": choices,
        "prediction": prediction,
        "predicted_answer": choices[prediction],
        "probabilities": {
            chr(65 + i): float(probabilities[i])
            for i in range(len(probabilities))
        },
        "confidence": float(probabilities[prediction]),
    }

    if uncertainty is not None:
        result["uncertainty"] = float(uncertainty)

    return result


def predict_from_file(
    checkpoint_path: str,
    config_path: str,
    input_file: str,
    output_file: str = None,
    compute_uncertainty: bool = True,
):
    """Make predictions on questions from a JSON file.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to configuration file.
        input_file: Path to input JSON file with questions.
        output_file: Optional path to save predictions.
        compute_uncertainty: Whether to compute uncertainty estimates.
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

        # Initialize preprocessor
        model_name = config.get("model", {}).get("name", "bert-base-uncased")
        max_length = config.get("data", {}).get("max_length", 512)
        preprocessor = MMLUPreprocessor(
            model_name=model_name,
            max_length=max_length,
        )

        # Initialize and load model
        logger.info("Loading model...")
        model = UncertaintyAwareModel(
            model_name=model_name,
            num_classes=config.get("model", {}).get("num_classes", 4),
            dropout_rate=config.get("model", {}).get("dropout_rate", 0.3),
            use_calibration=config.get("model", {}).get("use_calibration", True),
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        # Load input questions
        logger.info(f"Loading questions from {input_file}")
        with open(input_file, 'r') as f:
            questions_data = json.load(f)

        # Make predictions
        logger.info(f"Making predictions on {len(questions_data)} questions...")

        predictions = []
        mc_samples = config.get("curriculum", {}).get("mc_samples", 10)

        for i, item in enumerate(questions_data):
            logger.info(f"Processing question {i+1}/{len(questions_data)}")

            result = predict_single(
                model=model,
                preprocessor=preprocessor,
                question=item["question"],
                choices=item["choices"],
                device=device,
                compute_uncertainty=compute_uncertainty,
                mc_samples=mc_samples,
            )

            predictions.append(result)

            # Print result
            logger.info(f"  Question: {result['question'][:100]}...")
            logger.info(f"  Predicted: {result['predicted_answer']}")
            logger.info(f"  Confidence: {result['confidence']:.4f}")
            if 'uncertainty' in result:
                logger.info(f"  Uncertainty: {result['uncertainty']:.4f}")

        # Save predictions
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Saved predictions to {output_file}")

        return predictions

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_interactive(
    checkpoint_path: str,
    config_path: str,
):
    """Run interactive prediction mode.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to configuration file.
    """
    # Load configuration
    config = load_config(config_path)
    set_random_seeds(config.get("experiment", {}).get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize preprocessor and model
    model_name = config.get("model", {}).get("name", "bert-base-uncased")
    preprocessor = MMLUPreprocessor(model_name=model_name)

    model = UncertaintyAwareModel(
        model_name=model_name,
        num_classes=4,
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.3),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Interactive prediction mode. Type 'quit' to exit.")

    while True:
        # Get question
        print("\n" + "="*80)
        question = input("Enter question (or 'quit'): ").strip()

        if question.lower() == 'quit':
            break

        # Get choices
        choices = []
        for i in range(4):
            choice = input(f"Enter choice {chr(65+i)}: ").strip()
            choices.append(choice)

        # Make prediction
        result = predict_single(
            model=model,
            preprocessor=preprocessor,
            question=question,
            choices=choices,
            device=device,
            compute_uncertainty=True,
        )

        # Display result
        print("\n" + "-"*80)
        print("PREDICTION:")
        print(f"  Answer: {result['predicted_answer']} (Option {chr(65 + result['prediction'])})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Uncertainty: {result.get('uncertainty', 0):.4f}")
        print("\nProbabilities:")
        for option, prob in result['probabilities'].items():
            print(f"  {option}: {prob:.4f}")
        print("-"*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained MMLU model"
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
        "--input",
        type=str,
        help="Path to input JSON file with questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions JSON",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Disable uncertainty computation",
    )

    args = parser.parse_args()

    if args.interactive:
        predict_interactive(args.checkpoint, args.config)
    elif args.input:
        predict_from_file(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            input_file=args.input,
            output_file=args.output,
            compute_uncertainty=not args.no_uncertainty,
        )
    else:
        parser.error("Either --input or --interactive must be specified")


if __name__ == "__main__":
    main()
