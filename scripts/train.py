#!/usr/bin/env python3
"""Training script for temporal credit degradation detector.

This script handles the complete training pipeline including data loading,
preprocessing, model training with cross-validation, and model persistence.
"""

import argparse
import logging
import sys
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.training.trainer import ModelTrainer
from temporal_credit_degradation_detector.utils.config import load_config, load_config_from_env
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator

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
        description="Train temporal credit degradation detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data directory (overrides config)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        help='MLflow experiment name (overrides config)'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        help='Number of cross-validation folds (overrides config)'
    )

    parser.add_argument(
        '--optimize-hp',
        action='store_true',
        help='Enable hyperparameter optimization'
    )

    parser.add_argument(
        '--no-optimize-hp',
        action='store_false',
        dest='optimize_hp',
        help='Disable hyperparameter optimization'
    )

    parser.add_argument(
        '--hp-trials',
        type=int,
        help='Number of hyperparameter optimization trials (overrides config)'
    )

    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Use early stopping during training'
    )

    parser.add_argument(
        '--max-updates',
        type=int,
        help='Maximum weight updates for early stopping'
    )

    parser.add_argument(
        '--patience',
        type=int,
        help='Early stopping patience'
    )

    parser.add_argument(
        '--train-samples',
        type=int,
        help='Number of training samples to use'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        help='Random state for reproducibility (overrides config)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--save-artifacts',
        action='store_true',
        default=True,
        help='Save training artifacts'
    )

    parser.set_defaults(optimize_hp=True)
    return parser.parse_args()


def create_overrides(args: argparse.Namespace) -> dict:
    """Create configuration overrides from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}

    if args.data_path:
        overrides.setdefault('data', {})['data_path'] = args.data_path

    if args.experiment_name:
        overrides.setdefault('experiment', {})['experiment_name'] = args.experiment_name

    if args.cv_folds:
        overrides.setdefault('training', {})['cv_folds'] = args.cv_folds

    if args.hp_trials:
        overrides.setdefault('training', {})['n_hp_trials'] = args.hp_trials

    if args.max_updates:
        overrides.setdefault('training', {})['max_updates'] = args.max_updates

    if args.patience:
        overrides.setdefault('training', {})['patience'] = args.patience

    if args.train_samples:
        overrides.setdefault('data', {})['home_credit_samples'] = args.train_samples

    if args.random_state:
        overrides['random_state'] = args.random_state

    if args.output_dir:
        overrides['output_dir'] = args.output_dir

    # Handle optimize_hp flag
    overrides.setdefault('training', {})['optimize_hyperparameters'] = args.optimize_hp

    # Handle early stopping
    overrides.setdefault('training', {})['early_stopping'] = args.early_stopping

    return overrides


def load_and_split_data(data_loader: DataLoader, config) -> dict:
    """Load and split data for training.

    Args:
        data_loader: Data loader instance
        config: Configuration object

    Returns:
        Dictionary with data splits
    """
    logger = logging.getLogger(__name__)

    # Load Home Credit data (primary dataset)
    logger.info("Loading Home Credit data...")
    X_home, y_home = data_loader.load_home_credit_data(
        sample_size=config.data.home_credit_samples
    )

    # Create temporal splits
    logger.info("Creating temporal splits...")
    home_splits = data_loader.create_temporal_splits(
        X_home, y_home,
        time_column=config.data.temporal_column,
        train_months=config.data.train_months,
        val_months=config.data.val_months,
        test_months=config.data.test_months
    )

    # Load Lending Club data for additional evaluation (optional)
    try:
        logger.info("Loading Lending Club data...")
        X_lc, y_lc = data_loader.load_lending_club_data(
            sample_size=config.data.lending_club_samples
        )

        lc_splits = data_loader.create_temporal_splits(
            X_lc, y_lc,
            time_column='ISSUE_MONTH',
            train_months=config.data.train_months,
            val_months=config.data.val_months,
            test_months=config.data.test_months
        )

        return {
            'home_credit': home_splits,
            'lending_club': lc_splits
        }

    except Exception as e:
        logger.warning(f"Could not load Lending Club data: {e}")
        return {
            'home_credit': home_splits
        }


def train_model(trainer: ModelTrainer, data_splits: dict, config) -> dict:
    """Train the model using the specified approach.

    Args:
        trainer: Model trainer instance
        data_splits: Dictionary with data splits
        config: Configuration object

    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)

    # Use primary dataset (Home Credit) for training
    primary_splits = data_splits['home_credit']
    X_train, y_train = primary_splits['train']
    X_val, y_val = primary_splits['val']

    if config.training.early_stopping:
        logger.info("Training with early stopping...")
        results = trainer.train_with_early_stopping(
            X_train, y_train, X_val, y_val,
            max_updates=config.training.max_updates,
            patience=config.training.patience
        )
    else:
        logger.info("Training with cross-validation...")
        # Combine train and validation for CV
        import pandas as pd
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)

        results = trainer.train_cross_validation(
            X_combined, y_combined,
            cv_folds=config.training.cv_folds,
            optimize_hp=config.training.optimize_hyperparameters,
            n_hp_trials=config.training.n_hp_trials
        )

    return results


def evaluate_model(trainer: ModelTrainer, data_splits: dict) -> dict:
    """Evaluate the trained model on test sets.

    Args:
        trainer: Model trainer instance
        data_splits: Dictionary with data splits

    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    all_results = {}

    for dataset_name, splits in data_splits.items():
        logger.info(f"Evaluating on {dataset_name} test set...")

        try:
            X_test, y_test = splits['test']
            results = trainer.evaluate_on_test(
                X_test, y_test,
                save_predictions=True
            )

            all_results[dataset_name] = results

            # Log key metrics
            metrics = results['metrics']
            logger.info(f"{dataset_name} Results:")
            logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            logger.info(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
            logger.info(f"  Calibration Error: {metrics.get('calibration_error', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate on {dataset_name}: {e}")
            logger.warning("Skipping this dataset (likely different feature schema)")

    return all_results


def perform_drift_analysis(trainer: ModelTrainer, data_splits: dict) -> dict:
    """Perform drift analysis across datasets.

    Args:
        trainer: Model trainer instance
        data_splits: Dictionary with data splits

    Returns:
        Drift analysis results
    """
    logger = logging.getLogger(__name__)

    if len(data_splits) < 2:
        logger.info("Skipping drift analysis - only one dataset available")
        return {}

    try:
        logger.info("Performing drift analysis...")

        # Use Home Credit as reference, Lending Club as current
        home_splits = data_splits['home_credit']
        lc_splits = data_splits['lending_club']

        X_ref, y_ref = home_splits['test']
        X_curr, y_curr = lc_splits['test']

        # Get predictions
        y_pred_ref = trainer.model.predict_proba(trainer.preprocessor.transform(X_ref))[:, 1]
        y_pred_curr = trainer.model.predict_proba(trainer.preprocessor.transform(X_curr))[:, 1]

        # Create drift report
        evaluator = ModelEvaluator()
        drift_report = evaluator.create_drift_report(
            X_ref, X_curr, y_pred_ref, y_pred_curr, y_ref.values, y_curr.values
        )

        logger.info("Drift Analysis Results:")
        logger.info(f"  Overall Drift Score: {drift_report.get('overall_drift_score', 0):.4f}")
        logger.info(f"  Feature Drift Ratio (KS): {drift_report.get('feature_drift_summary', {}).get('ks_drift_ratio', 0):.4f}")
        logger.info(f"  Prediction Drift: {drift_report.get('prediction_drift', {}).get('prediction_drift', False)}")

        recommendations = drift_report.get('recommendations', [])
        if recommendations:
            logger.info("  Recommendations:")
            for rec in recommendations:
                logger.info(f"    - {rec}")

        return drift_report
    except Exception as e:
        logger.warning(f"Drift analysis failed: {e}")
        logger.warning("Skipping drift analysis (likely different feature schemas between datasets)")
        return {}


def main():
    """Main training function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")

        # Create overrides from command line and environment
        cli_overrides = create_overrides(args)
        env_overrides = load_config_from_env()

        # Combine overrides (CLI takes precedence)
        combined_overrides = {**env_overrides, **cli_overrides}

        config = load_config(args.config, combined_overrides)
        logger.info("Configuration loaded successfully")

        # Initialize data loader
        data_loader = DataLoader(config.data.data_path)

        # Load and split data
        logger.info("Loading and splitting data...")
        data_splits = load_and_split_data(data_loader, config)

        # Initialize trainer
        trainer = ModelTrainer(config.to_dict())

        # Train model
        logger.info("Starting model training...")
        training_results = train_model(trainer, data_splits, config)

        # Evaluate model
        logger.info("Evaluating trained model...")
        evaluation_results = evaluate_model(trainer, data_splits)

        # Perform drift analysis
        drift_results = perform_drift_analysis(trainer, data_splits)

        # Save artifacts
        if args.save_artifacts:
            logger.info("Saving training artifacts...")
            trainer.save_training_artifacts(training_results)

            # Save configuration used
            config_path = Path(config.output_dir) / "final_config.yaml"
            config.save(config_path)

        # Print summary
        logger.info("Training completed successfully!")

        # Print key results
        primary_dataset = list(data_splits.keys())[0]
        primary_metrics = evaluation_results[primary_dataset]['metrics']

        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Primary Dataset: {primary_dataset}")
        print(f"AUC-ROC: {primary_metrics.get('auc_roc', 0):.4f}")
        print(f"Brier Score: {primary_metrics.get('brier_score', 0):.4f}")
        print(f"Calibration Error: {primary_metrics.get('calibration_error', 0):.4f}")

        if drift_results:
            print(f"Overall Drift Score: {drift_results.get('overall_drift_score', 0):.4f}")

        print(f"Model saved to: {trainer.model_dir}")
        print("="*50)

        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())