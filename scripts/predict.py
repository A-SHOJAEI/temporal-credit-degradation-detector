#!/usr/bin/env python3
"""Prediction script for temporal credit degradation detector.

This script loads a trained model and generates predictions on new data,
demonstrating inference capabilities and prediction format.
"""

import argparse
import logging
import sys
from pathlib import Path
import pickle
import warnings
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_credit_degradation_detector.data.loader import DataLoader

warnings.filterwarnings('ignore')


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/best_model_cv.pkl',
        help='Path to trained model file'
    )

    parser.add_argument(
        '--preprocessor-path',
        type=str,
        default='models/preprocessor.pkl',
        help='Path to preprocessor file'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of samples to predict on'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """Load trained model and preprocessor.

    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file

    Returns:
        Tuple of (model, preprocessor)
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Loading preprocessor from {preprocessor_path}")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def main():
    """Main prediction pipeline."""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting prediction pipeline")

    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(
        args.model_path,
        args.preprocessor_path
    )

    # Load sample data
    logger.info(f"Loading sample data (n={args.sample_size})")
    loader = DataLoader()
    X, y = loader.load_home_credit_data(sample_size=args.sample_size)

    # Create temporal splits to get test data
    splits = loader.create_temporal_splits(X, y, test_size=0.3)
    X_test, y_test = splits['test']

    # Preprocess data
    logger.info("Preprocessing data")
    X_test_processed = preprocessor.transform(X_test)

    # Generate predictions
    logger.info("Generating predictions")
    predictions_proba = model.predict_proba(X_test_processed)[:, 1]
    predictions_class = model.predict(X_test_processed)

    # Create results dataframe
    results = pd.DataFrame({
        'actual': y_test,
        'predicted_class': predictions_class,
        'predicted_probability': predictions_proba,
        'default_risk': pd.cut(
            predictions_proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })

    # Add decision recommendation
    results['recommendation'] = results['predicted_probability'].apply(
        lambda p: 'Approve' if p < 0.5 else 'Reject'
    )

    # Save predictions
    logger.info(f"Saving predictions to {args.output_path}")
    results.to_csv(args.output_path, index=False)

    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Actual default rate: {results['actual'].mean():.2%}")
    logger.info(f"Predicted default rate: {results['predicted_class'].mean():.2%}")
    logger.info(f"\nPredicted probability statistics:")
    logger.info(f"  Mean: {results['predicted_probability'].mean():.4f}")
    logger.info(f"  Std:  {results['predicted_probability'].std():.4f}")
    logger.info(f"  Min:  {results['predicted_probability'].min():.4f}")
    logger.info(f"  Max:  {results['predicted_probability'].max():.4f}")
    logger.info(f"\nRisk distribution:")
    logger.info(results['default_risk'].value_counts().to_string())
    logger.info(f"\nRecommendations:")
    logger.info(results['recommendation'].value_counts().to_string())
    logger.info("="*50)

    # Sample predictions
    logger.info("\nSample predictions (first 10):")
    logger.info(results.head(10).to_string(index=False))

    logger.info(f"\nPredictions saved successfully to {args.output_path}")


if __name__ == '__main__':
    main()
