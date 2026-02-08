#!/usr/bin/env python3
"""Evaluation script for temporal credit degradation detector.

This script provides comprehensive evaluation capabilities including
temporal stability analysis, drift detection, and performance benchmarking.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.training.trainer import ModelTrainer
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator
from temporal_credit_degradation_detector.utils.config import load_config
from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble

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
        description="Evaluate temporal credit degradation detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model file'
    )

    parser.add_argument(
        '--preprocessor-path',
        type=str,
        required=True,
        help='Path to saved preprocessor file'
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
        help='Path to data directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--evaluation-type',
        choices=['basic', 'temporal', 'drift', 'comprehensive'],
        default='comprehensive',
        help='Type of evaluation to perform'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['home_credit', 'lending_club', 'both'],
        default=['both'],
        help='Datasets to evaluate on'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        help='Sample size for evaluation (for faster testing)'
    )

    parser.add_argument(
        '--generate-plots',
        action='store_true',
        default=True,
        help='Generate evaluation plots'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Save predictions for further analysis'
    )

    parser.add_argument(
        '--benchmark-targets',
        type=str,
        help='Path to JSON file with benchmark targets'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str) -> tuple:
    """Load saved model and preprocessor.

    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor

    Returns:
        Tuple of (model, preprocessor)
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")
    model = StabilityWeightedEnsemble.load_model(model_path)

    logger.info(f"Loading preprocessor from {preprocessor_path}")
    import joblib
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def load_evaluation_data(data_loader: DataLoader, datasets: list, config, sample_size=None) -> Dict[str, Dict]:
    """Load data for evaluation.

    Args:
        data_loader: Data loader instance
        datasets: List of datasets to load
        config: Configuration object
        sample_size: Optional sample size limit

    Returns:
        Dictionary with loaded datasets and splits
    """
    logger = logging.getLogger(__name__)
    data_splits = {}

    if 'home_credit' in datasets or 'both' in datasets:
        logger.info("Loading Home Credit data...")
        X_home, y_home = data_loader.load_home_credit_data(sample_size=sample_size)

        home_splits = data_loader.create_temporal_splits(
            X_home, y_home,
            time_column=config.data.temporal_column,
            train_months=config.data.train_months,
            val_months=config.data.val_months,
            test_months=config.data.test_months
        )

        data_splits['home_credit'] = home_splits

    if 'lending_club' in datasets or 'both' in datasets:
        try:
            logger.info("Loading Lending Club data...")
            X_lc, y_lc = data_loader.load_lending_club_data(sample_size=sample_size)

            lc_splits = data_loader.create_temporal_splits(
                X_lc, y_lc,
                time_column='ISSUE_MONTH',
                train_months=config.data.train_months,
                val_months=config.data.val_months,
                test_months=config.data.test_months
            )

            data_splits['lending_club'] = lc_splits

        except Exception as e:
            logger.warning(f"Could not load Lending Club data: {e}")

    return data_splits


def perform_basic_evaluation(model, preprocessor, data_splits: Dict, output_dir: Path) -> Dict[str, Any]:
    """Perform basic model evaluation.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        data_splits: Data splits for evaluation
        output_dir: Output directory for results

    Returns:
        Basic evaluation results
    """
    logger = logging.getLogger(__name__)
    evaluator = ModelEvaluator()
    results = {}

    for dataset_name, splits in data_splits.items():
        logger.info(f"Performing basic evaluation on {dataset_name}")

        # Evaluate on all splits
        for split_name, (X, y) in splits.items():
            logger.info(f"  Evaluating {split_name} split...")

            # Preprocess data
            X_processed = preprocessor.transform(X)

            # Get predictions
            y_prob = model.predict_proba(X_processed)[:, 1]
            y_pred = model.predict(X_processed)

            # Calculate metrics
            metrics = evaluator.calculate_metrics(y.values, y_prob, y_pred)

            # Store results
            key = f"{dataset_name}_{split_name}"
            results[key] = {
                'metrics': metrics,
                'predictions': y_prob.tolist(),
                'true_labels': y.values.tolist()
            }

            logger.info(f"    AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            logger.info(f"    Brier Score: {metrics.get('brier_score', 0):.4f}")

    # Save results
    results_file = output_dir / "basic_evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            json_results[key] = {
                'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in value['metrics'].items()},
                'predictions': value['predictions'],
                'true_labels': value['true_labels']
            }
        json.dump(json_results, f, indent=2)

    logger.info(f"Basic evaluation results saved to {results_file}")
    return results


def perform_temporal_evaluation(model, preprocessor, data_splits: Dict, output_dir: Path) -> Dict[str, Any]:
    """Perform temporal stability evaluation.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        data_splits: Data splits for evaluation
        output_dir: Output directory for results

    Returns:
        Temporal evaluation results
    """
    logger = logging.getLogger(__name__)
    evaluator = ModelEvaluator()
    results = {}

    for dataset_name, splits in data_splits.items():
        logger.info(f"Performing temporal evaluation on {dataset_name}")

        # Create time-based predictions for temporal analysis
        predictions_by_time = {}

        # For this demo, we'll simulate temporal periods using different splits
        time_periods = ['early', 'middle', 'late']
        split_names = ['train', 'val', 'test']

        for i, (period, split_name) in enumerate(zip(time_periods, split_names)):
            if split_name in splits:
                X, y = splits[split_name]
                X_processed = preprocessor.transform(X)
                y_prob = model.predict_proba(X_processed)[:, 1]

                predictions_by_time[period] = (y.values, y_prob)

        # Evaluate temporal stability
        if len(predictions_by_time) > 1:
            stability_results = evaluator.evaluate_temporal_stability(
                predictions_by_time, time_periods
            )

            results[dataset_name] = stability_results

            logger.info(f"  Temporal stability score: {stability_results.get('overall_stability_score', 0):.4f}")

    # Save results
    results_file = output_dir / "temporal_evaluation_results.json"
    with open(results_file, 'w') as f:
        json_results = {}
        for key, value in results.items():
            json_results[key] = _convert_numpy_types(value)
        json.dump(json_results, f, indent=2)

    logger.info(f"Temporal evaluation results saved to {results_file}")
    return results


def perform_drift_evaluation(model, preprocessor, data_splits: Dict, output_dir: Path) -> Dict[str, Any]:
    """Perform drift detection evaluation.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        data_splits: Data splits for evaluation
        output_dir: Output directory for results

    Returns:
        Drift evaluation results
    """
    logger = logging.getLogger(__name__)
    evaluator = ModelEvaluator()
    results = {}

    # Perform drift analysis between datasets or time periods
    if len(data_splits) >= 2:
        dataset_names = list(data_splits.keys())
        reference_dataset = dataset_names[0]
        current_dataset = dataset_names[1]

        logger.info(f"Performing drift analysis: {reference_dataset} -> {current_dataset}")

        # Use test splits for drift analysis
        X_ref, y_ref = data_splits[reference_dataset]['test']
        X_curr, y_curr = data_splits[current_dataset]['test']

        # Preprocess data
        X_ref_processed = preprocessor.transform(X_ref)
        X_curr_processed = preprocessor.transform(X_curr)

        # Get predictions
        y_pred_ref = model.predict_proba(X_ref_processed)[:, 1]
        y_pred_curr = model.predict_proba(X_curr_processed)[:, 1]

        # Create drift report
        drift_report = evaluator.create_drift_report(
            X_ref, X_curr, y_pred_ref, y_pred_curr,
            y_ref.values, y_curr.values
        )

        results['cross_dataset_drift'] = drift_report

        logger.info(f"  Overall drift score: {drift_report.get('overall_drift_score', 0):.4f}")

    # Perform temporal drift analysis within each dataset
    for dataset_name, splits in data_splits.items():
        logger.info(f"Performing temporal drift analysis on {dataset_name}")

        # Use train as reference, test as current
        X_ref, y_ref = splits['train']
        X_curr, y_curr = splits['test']

        # Preprocess data
        X_ref_processed = preprocessor.transform(X_ref)
        X_curr_processed = preprocessor.transform(X_curr)

        # Get predictions
        y_pred_ref = model.predict_proba(X_ref_processed)[:, 1]
        y_pred_curr = model.predict_proba(X_curr_processed)[:, 1]

        # Create drift report
        drift_report = evaluator.create_drift_report(
            X_ref, X_curr, y_pred_ref, y_pred_curr,
            y_ref.values, y_curr.values
        )

        results[f'{dataset_name}_temporal_drift'] = drift_report

        logger.info(f"  {dataset_name} temporal drift score: {drift_report.get('overall_drift_score', 0):.4f}")

    # Save results
    results_file = output_dir / "drift_evaluation_results.json"
    with open(results_file, 'w') as f:
        json_results = {}
        for key, value in results.items():
            json_results[key] = _convert_numpy_types(value)
        json.dump(json_results, f, indent=2)

    logger.info(f"Drift evaluation results saved to {results_file}")
    return results


def generate_evaluation_plots(results: Dict, output_dir: Path) -> None:
    """Generate evaluation plots and visualizations.

    Args:
        results: Evaluation results
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating evaluation plots...")

    plt.style.use('default')
    sns.set_palette("husl")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Performance comparison across datasets
    if 'basic_results' in results:
        basic_results = results['basic_results']
        plot_performance_comparison(basic_results, plots_dir)

    # Plot 2: Temporal stability
    if 'temporal_results' in results:
        temporal_results = results['temporal_results']
        plot_temporal_stability(temporal_results, plots_dir)

    # Plot 3: Drift analysis
    if 'drift_results' in results:
        drift_results = results['drift_results']
        plot_drift_analysis(drift_results, plots_dir)

    logger.info(f"Evaluation plots saved to {plots_dir}")


def plot_performance_comparison(basic_results: Dict, plots_dir: Path) -> None:
    """Plot performance comparison across datasets and splits."""
    metrics_data = []

    for key, value in basic_results.items():
        if 'metrics' in value:
            parts = key.split('_')
            dataset = parts[0]
            split = '_'.join(parts[1:])

            metrics = value['metrics']
            metrics_data.append({
                'Dataset': dataset,
                'Split': split,
                'AUC-ROC': metrics.get('auc_roc', 0),
                'Brier Score': metrics.get('brier_score', 0),
                'Calibration Error': metrics.get('calibration_error', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0)
            })

    if not metrics_data:
        return

    df = pd.DataFrame(metrics_data)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    metrics_to_plot = ['AUC-ROC', 'Brier Score', 'Calibration Error', 'Precision', 'Recall']

    for i, metric in enumerate(metrics_to_plot):
        row = i // 3
        col = i % 3

        sns.barplot(data=df, x='Split', y=metric, hue='Dataset', ax=axes[row, col])
        axes[row, col].set_title(f'{metric} by Dataset and Split')
        axes[row, col].tick_params(axis='x', rotation=45)

    # Remove empty subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_temporal_stability(temporal_results: Dict, plots_dir: Path) -> None:
    """Plot temporal stability analysis."""
    if not temporal_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Temporal Stability Analysis', fontsize=16)

    dataset_names = list(temporal_results.keys())
    stability_scores = [temporal_results[dataset].get('overall_stability_score', 0)
                       for dataset in dataset_names]

    # Plot 1: Overall stability scores
    axes[0].bar(dataset_names, stability_scores, color='skyblue')
    axes[0].set_title('Overall Stability Score by Dataset')
    axes[0].set_ylabel('Stability Score (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Metric stability over time (if available)
    if dataset_names and 'metrics_by_time' in temporal_results[dataset_names[0]]:
        dataset = dataset_names[0]
        metrics_by_time = temporal_results[dataset]['metrics_by_time']

        time_periods = list(metrics_by_time.keys())
        auc_scores = [metrics_by_time[period].get('auc_roc', 0) for period in time_periods]

        axes[1].plot(time_periods, auc_scores, marker='o', linewidth=2)
        axes[1].set_title(f'AUC-ROC Over Time ({dataset})')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_xlabel('Time Period')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'temporal_stability.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_drift_analysis(drift_results: Dict, plots_dir: Path) -> None:
    """Plot drift analysis results."""
    if not drift_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Drift Analysis Results', fontsize=16)

    # Plot 1: Overall drift scores
    drift_names = []
    drift_scores = []

    for key, value in drift_results.items():
        if 'overall_drift_score' in value:
            drift_names.append(key.replace('_', ' ').title())
            drift_scores.append(value['overall_drift_score'])

    if drift_names:
        axes[0].bar(drift_names, drift_scores, color='coral')
        axes[0].set_title('Overall Drift Scores')
        axes[0].set_ylabel('Drift Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='High Drift Threshold')
        axes[0].legend()

    # Plot 2: Feature drift summary (if available)
    if drift_names and 'feature_drift_summary' in drift_results[list(drift_results.keys())[0]]:
        first_key = list(drift_results.keys())[0]
        drift_summary = drift_results[first_key]['feature_drift_summary']

        categories = ['KS Test', 'JS Divergence']
        drift_ratios = [
            drift_summary.get('ks_drift_ratio', 0),
            drift_summary.get('js_drift_ratio', 0)
        ]

        axes[1].bar(categories, drift_ratios, color='lightgreen')
        axes[1].set_title('Feature Drift Detection Methods')
        axes[1].set_ylabel('Fraction of Features with Drift')
        axes[1].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Significant Drift Threshold')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'drift_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def check_benchmark_targets(results: Dict, benchmark_file: str) -> Dict[str, bool]:
    """Check if evaluation results meet benchmark targets.

    Args:
        results: Evaluation results
        benchmark_file: Path to benchmark targets file

    Returns:
        Dictionary indicating which benchmarks are met
    """
    logger = logging.getLogger(__name__)

    try:
        with open(benchmark_file, 'r') as f:
            targets = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load benchmark targets: {e}")
        return {}

    benchmark_status = {}

    if 'basic_results' in results:
        for key, value in results['basic_results'].items():
            if 'test' in key and 'metrics' in value:
                metrics = value['metrics']

                # Check each target
                for metric, target_value in targets.items():
                    if metric in metrics:
                        actual_value = metrics[metric]

                        # Determine if target is met based on metric type
                        if metric in ['auc_roc', 'precision', 'recall', 'f1_score']:
                            # Higher is better
                            meets_target = actual_value >= target_value
                        else:
                            # Lower is better (brier_score, calibration_error, etc.)
                            meets_target = actual_value <= target_value

                        benchmark_status[f"{key}_{metric}"] = meets_target

                        if meets_target:
                            logger.info(f"✓ {key} {metric}: {actual_value:.4f} meets target {target_value}")
                        else:
                            logger.warning(f"✗ {key} {metric}: {actual_value:.4f} does not meet target {target_value}")

    return benchmark_status


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    """Main evaluation function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Load configuration
        config = load_config(args.config)

        # Override data path if provided
        if args.data_path:
            config.data.data_path = args.data_path

        # Load model and preprocessor
        model, preprocessor = load_model_and_preprocessor(args.model_path, args.preprocessor_path)

        # Initialize data loader
        data_loader = DataLoader(config.data.data_path)

        # Load evaluation data
        data_splits = load_evaluation_data(data_loader, args.datasets, config, args.sample_size)

        if not data_splits:
            logger.error("No data loaded for evaluation")
            return 1

        # Store all results
        all_results = {}

        # Perform evaluations based on type
        if args.evaluation_type in ['basic', 'comprehensive']:
            logger.info("Performing basic evaluation...")
            basic_results = perform_basic_evaluation(model, preprocessor, data_splits, output_dir)
            all_results['basic_results'] = basic_results

        if args.evaluation_type in ['temporal', 'comprehensive']:
            logger.info("Performing temporal evaluation...")
            temporal_results = perform_temporal_evaluation(model, preprocessor, data_splits, output_dir)
            all_results['temporal_results'] = temporal_results

        if args.evaluation_type in ['drift', 'comprehensive']:
            logger.info("Performing drift evaluation...")
            drift_results = perform_drift_evaluation(model, preprocessor, data_splits, output_dir)
            all_results['drift_results'] = drift_results

        # Generate plots
        if args.generate_plots:
            generate_evaluation_plots(all_results, output_dir)

        # Check benchmark targets
        if args.benchmark_targets:
            benchmark_status = check_benchmark_targets(all_results, args.benchmark_targets)
            benchmark_file = output_dir / "benchmark_status.json"
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_status, f, indent=2)

        # Save comprehensive results
        comprehensive_results_file = output_dir / "comprehensive_evaluation_results.json"
        with open(comprehensive_results_file, 'w') as f:
            json_results = _convert_numpy_types(all_results)
            json.dump(json_results, f, indent=2)

        logger.info(f"Evaluation completed successfully! Results saved to {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())