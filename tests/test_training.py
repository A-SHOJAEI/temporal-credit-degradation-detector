"""Tests for training pipeline and evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from temporal_credit_degradation_detector.training.trainer import ModelTrainer, EarlyStopping
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator, DriftDetector


class TestEarlyStopping:
    """Test cases for EarlyStopping class."""

    def test_init(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        assert es.patience == 5
        assert es.min_delta == 0.001
        assert es.mode == 'max'
        assert not es.early_stop

    def test_max_mode_improvement(self):
        """Test early stopping in max mode with improvements."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode='max')

        # Improving scores
        assert not es(0.5)  # Initial score
        assert not es(0.6)  # Improvement
        assert not es(0.65) # Small improvement
        assert not es(0.7)  # Improvement

        assert not es.early_stop

    def test_max_mode_no_improvement(self):
        """Test early stopping in max mode without improvements."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode='max')

        # No improvements
        assert not es(0.5)  # Initial score
        assert not es(0.5)  # No improvement (counter = 1)
        assert es(0.5)      # No improvement (counter = 2, triggers stop)

        assert es.early_stop

    def test_min_mode_improvement(self):
        """Test early stopping in min mode with improvements."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode='min')

        # Improving scores (lower is better)
        assert not es(1.0)  # Initial score
        assert not es(0.8)  # Improvement
        assert not es(0.6)  # Improvement

        assert not es.early_stop

    def test_min_mode_no_improvement(self):
        """Test early stopping in min mode without improvements."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode='min')

        # No improvements
        assert not es(1.0)  # Initial score
        assert not es(1.1)  # Worse (counter = 1)
        assert es(1.2)      # Worse (counter = 2, triggers stop)

        assert es.early_stop


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    def test_init(self, config, temp_dir):
        """Test ModelTrainer initialization."""
        config.model_dir = temp_dir
        trainer = ModelTrainer(config.to_dict())

        assert trainer.config == config.to_dict()
        assert trainer.experiment_name == config.experiment.experiment_name
        assert Path(trainer.model_dir).exists()

    @patch('temporal_credit_degradation_detector.training.trainer.mlflow')
    def test_setup_mlflow(self, mock_mlflow, config):
        """Test MLflow setup."""
        trainer = ModelTrainer(config.to_dict())
        # MLflow setup is called in __init__
        mock_mlflow.set_experiment.assert_called_once()

    def test_prepare_data(self, config, sample_temporal_data):
        """Test data preparation."""
        trainer = ModelTrainer(config.to_dict())
        X_train, y_train = sample_temporal_data['train']
        X_val, y_val = sample_temporal_data['val']

        X_train_prep, y_train_prep, X_val_prep, y_val_prep = trainer.prepare_data(
            X_train, y_train, X_val, y_val
        )

        # Check that preprocessor is fitted
        assert trainer.preprocessor is not None
        assert trainer.preprocessor.is_fitted

        # Check output shapes
        assert X_train_prep.shape[0] == X_train.shape[0]
        assert X_val_prep.shape[0] == X_val.shape[0]
        assert len(y_train_prep) == len(y_train)
        assert len(y_val_prep) == len(y_val)

        # Should have no missing values
        assert not X_train_prep.isnull().any().any()
        assert not X_val_prep.isnull().any().any()

    def test_prepare_data_without_validation(self, config, sample_temporal_data):
        """Test data preparation without validation data."""
        trainer = ModelTrainer(config.to_dict())
        X_train, y_train = sample_temporal_data['train']

        X_train_prep, y_train_prep, X_val_prep, y_val_prep = trainer.prepare_data(
            X_train, y_train
        )

        assert X_val_prep is None
        assert y_val_prep is None

    @patch('temporal_credit_degradation_detector.training.trainer.optuna')
    def test_optimize_hyperparameters(self, mock_optuna, config, sample_temporal_data):
        """Test hyperparameter optimization."""
        # Mock optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            'stability_alpha': 0.15,
            'min_weight': 0.03,
            'recalibration_threshold': 0.12,
            'calibration_window': 750
        }
        mock_study.best_value = 0.85
        mock_optuna.create_study.return_value = mock_study

        trainer = ModelTrainer(config.to_dict())
        X_train, y_train = sample_temporal_data['train']
        X_val, y_val = sample_temporal_data['val']

        # Prepare data first
        X_train_prep, y_train, X_val_prep, y_val = trainer.prepare_data(
            X_train, y_train, X_val, y_val
        )

        best_params = trainer.optimize_hyperparameters(
            X_train_prep, y_train, X_val_prep, y_val, n_trials=5
        )

        assert isinstance(best_params, dict)
        assert 'stability_alpha' in best_params
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()

    @patch('temporal_credit_degradation_detector.training.trainer.mlflow')
    def test_train_cross_validation(self, mock_mlflow, config, sample_temporal_data, temp_dir):
        """Test cross-validation training."""
        config.model_dir = temp_dir
        config.training.cv_folds = 3
        config.training.optimize_hyperparameters = False
        config.training.n_hp_trials = 5

        trainer = ModelTrainer(config.to_dict())
        X, y = sample_temporal_data['train']

        # Combine train and val for CV
        X_val, y_val = sample_temporal_data['val']
        X_combined = pd.concat([X, X_val], ignore_index=True)
        y_combined = pd.concat([y, y_val], ignore_index=True)

        results = trainer.train_cross_validation(
            X_combined, y_combined, cv_folds=3, optimize_hp=False
        )

        # Check results structure
        assert 'cv_scores' in results
        assert 'avg_metrics' in results
        assert 'best_fold' in results
        assert 'model' in results

        # Check that model is trained
        assert trainer.model is not None
        assert trainer.model.is_fitted_

        # Check CV scores
        assert len(results['cv_scores']) == 3

        # Check average metrics
        assert 'cv_auc_roc_mean' in results['avg_metrics']
        assert 'cv_auc_roc_std' in results['avg_metrics']

    @patch('temporal_credit_degradation_detector.training.trainer.mlflow')
    def test_train_with_early_stopping(self, mock_mlflow, config, sample_temporal_data, temp_dir):
        """Test training with early stopping."""
        config.model_dir = temp_dir
        config.training.max_updates = 10
        config.training.patience = 3

        trainer = ModelTrainer(config.to_dict())
        X_train, y_train = sample_temporal_data['train']
        X_val, y_val = sample_temporal_data['val']

        results = trainer.train_with_early_stopping(
            X_train, y_train, X_val, y_val,
            max_updates=10, patience=3
        )

        # Check results structure
        assert 'training_history' in results
        assert 'final_metrics' in results
        assert 'total_updates' in results
        assert 'model' in results

        # Check that model is trained
        assert trainer.model is not None
        assert trainer.model.is_fitted_

        # Check training history
        assert len(results['training_history']) > 0
        assert results['total_updates'] <= 10

    def test_evaluate_on_test(self, config, sample_temporal_data, temp_dir):
        """Test test set evaluation."""
        config.model_dir = temp_dir
        trainer = ModelTrainer(config.to_dict())

        # Train model first
        X_train, y_train = sample_temporal_data['train']
        X_val, y_val = sample_temporal_data['val']
        trainer.train_with_early_stopping(X_train, y_train, X_val, y_val, max_updates=5)

        # Evaluate on test set
        X_test, y_test = sample_temporal_data['test']
        results = trainer.evaluate_on_test(X_test, y_test, save_predictions=True)

        # Check results structure
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'feature_importance' in results
        assert 'model_weights' in results

        # Check metrics
        metrics = results['metrics']
        assert 'auc_roc' in metrics
        assert 'brier_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

        # Check predictions
        assert len(results['predictions']) == len(y_test)

        # Check that predictions file is saved
        pred_file = Path(temp_dir) / "test_predictions.csv"
        assert pred_file.exists()

    def test_evaluate_without_trained_model(self, config, sample_temporal_data):
        """Test evaluation without trained model."""
        trainer = ModelTrainer(config.to_dict())
        X_test, y_test = sample_temporal_data['test']

        with pytest.raises(ValueError, match="Model must be trained"):
            trainer.evaluate_on_test(X_test, y_test)

    def test_save_training_artifacts(self, config, temp_dir):
        """Test saving training artifacts."""
        config.model_dir = temp_dir
        trainer = ModelTrainer(config.to_dict())

        # Mock results
        results = {
            'training_history': [
                {'auc_roc': 0.75, 'brier_score': 0.18},
                {'auc_roc': 0.78, 'brier_score': 0.16}
            ],
            'avg_metrics': {
                'cv_auc_roc_mean': 0.77,
                'cv_auc_roc_std': 0.02
            }
        }

        trainer.save_training_artifacts(results, suffix="test")

        # Check that files are created
        history_file = Path(temp_dir) / "training_history_test.json"
        metrics_file = Path(temp_dir) / "cv_metrics_test.json"

        assert history_file.exists()
        assert metrics_file.exists()

    def test_load_model(self, config, fitted_ensemble, temp_dir):
        """Test model loading."""
        # Save a model first
        model_path = Path(temp_dir) / "model.pkl"
        preprocessor_path = Path(temp_dir) / "preprocessor.pkl"

        fitted_ensemble.save_model(str(model_path))

        # Create and save preprocessor
        from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor
        import joblib

        preprocessor = CreditDataPreprocessor()
        joblib.dump(preprocessor, preprocessor_path)

        # Load model
        trainer = ModelTrainer(config.to_dict())
        trainer.load_model(str(model_path), str(preprocessor_path))

        assert trainer.model is not None
        assert trainer.preprocessor is not None
        assert trainer.model.is_fitted_


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    def test_init(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(calibration_bins=5)
        assert evaluator.calibration_bins == 5

    def test_calculate_metrics(self, sample_predictions, performance_benchmarks):
        """Test comprehensive metrics calculation."""
        evaluator = ModelEvaluator()
        y_true, y_prob = sample_predictions

        metrics = evaluator.calculate_metrics(y_true, y_prob)

        # Check that all expected metrics are present
        expected_metrics = [
            'auc_roc', 'auc_pr', 'precision', 'recall', 'f1_score',
            'brier_score', 'log_loss', 'calibration_error',
            'specificity', 'sensitivity', 'false_positive_rate'
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Check metric ranges
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['brier_score'] <= 1
        assert 0 <= metrics['calibration_error'] <= 1

        # Check against benchmarks
        assert metrics['auc_roc'] >= performance_benchmarks['min_auc_roc'] * 0.8  # Allow some tolerance
        assert metrics['brier_score'] <= performance_benchmarks['max_brier_score'] * 1.5

    def test_calculate_calibration_error(self, sample_predictions):
        """Test calibration error calculation."""
        evaluator = ModelEvaluator(calibration_bins=10)
        y_true, y_prob = sample_predictions

        cal_error = evaluator.calculate_calibration_error(y_true, y_prob)

        assert isinstance(cal_error, float)
        assert 0 <= cal_error <= 1

    def test_calculate_business_metrics(self, sample_predictions):
        """Test business metrics calculation."""
        evaluator = ModelEvaluator()
        y_true, y_prob = sample_predictions

        business_metrics = evaluator.calculate_business_metrics(y_true, y_prob)

        # Check that business metrics are present
        expected_business_metrics = [
            'optimal_threshold', 'max_profit', 'profit_at_05_threshold',
            'approval_rate_optimal', 'default_rate_if_all_approved'
        ]

        for metric in expected_business_metrics:
            assert metric in business_metrics

        # Check metric ranges
        assert 0 <= business_metrics['optimal_threshold'] <= 1
        assert 0 <= business_metrics['approval_rate_optimal'] <= 1
        assert 0 <= business_metrics['default_rate_if_all_approved'] <= 1

    def test_evaluate_temporal_stability(self, sample_predictions):
        """Test temporal stability evaluation."""
        evaluator = ModelEvaluator()

        # Create temporal predictions
        y_true, y_prob = sample_predictions
        n_periods = 5
        period_size = len(y_true) // n_periods

        predictions_by_time = {}
        time_periods = [f'period_{i}' for i in range(n_periods)]

        for i, period in enumerate(time_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(y_true)
            predictions_by_time[period] = (y_true[start_idx:end_idx], y_prob[start_idx:end_idx])

        stability_results = evaluator.evaluate_temporal_stability(
            predictions_by_time, time_periods
        )

        # Check results structure
        assert 'metrics_by_time' in stability_results
        assert 'stability_stats' in stability_results
        assert 'overall_stability_score' in stability_results

        # Check that all periods are evaluated
        assert len(stability_results['metrics_by_time']) == n_periods

        # Check stability statistics
        stability_stats = stability_results['stability_stats']
        for metric in ['auc_roc', 'brier_score']:
            assert metric in stability_stats
            assert 'mean' in stability_stats[metric]
            assert 'std' in stability_stats[metric]
            assert 'cv' in stability_stats[metric]

    def test_create_drift_report(self, sample_drift_data):
        """Test comprehensive drift report creation."""
        evaluator = ModelEvaluator()
        X_ref, X_curr, y_pred_ref, y_pred_curr = sample_drift_data

        # Generate true labels for performance comparison
        y_true_ref = np.random.binomial(1, 0.3, len(y_pred_ref))
        y_true_curr = np.random.binomial(1, 0.4, len(y_pred_curr))  # Different base rate

        drift_report = evaluator.create_drift_report(
            X_ref, X_curr, y_pred_ref, y_pred_curr,
            y_true_ref, y_true_curr
        )

        # Check report structure
        assert 'feature_drift_ks' in drift_report
        assert 'feature_drift_js' in drift_report
        assert 'feature_drift_summary' in drift_report
        assert 'prediction_drift' in drift_report
        assert 'performance_comparison' in drift_report
        assert 'recommendations' in drift_report
        assert 'overall_drift_score' in drift_report

        # Check drift summary
        summary = drift_report['feature_drift_summary']
        assert 'total_features' in summary
        assert 'ks_drift_count' in summary
        assert 'ks_drift_ratio' in summary

        # Check recommendations
        recommendations = drift_report['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_metrics_edge_cases(self):
        """Test metrics calculation with edge cases."""
        evaluator = ModelEvaluator()

        # Perfect predictions
        y_true_perfect = np.array([0, 0, 1, 1])
        y_prob_perfect = np.array([0.0, 0.1, 0.9, 1.0])

        metrics_perfect = evaluator.calculate_metrics(y_true_perfect, y_prob_perfect)
        assert metrics_perfect['auc_roc'] > 0.9  # Should be very high

        # Random predictions
        y_true_random = np.array([0, 1, 0, 1])
        y_prob_random = np.array([0.5, 0.5, 0.5, 0.5])

        metrics_random = evaluator.calculate_metrics(y_true_random, y_prob_random)
        assert 0.4 <= metrics_random['auc_roc'] <= 0.6  # Should be around 0.5

    def test_empty_input(self):
        """Test evaluator with empty input."""
        evaluator = ModelEvaluator()

        # Should handle empty arrays gracefully
        y_true_empty = np.array([])
        y_prob_empty = np.array([])

        with pytest.raises((ValueError, ZeroDivisionError)):
            evaluator.calculate_metrics(y_true_empty, y_prob_empty)


class TestDriftDetector:
    """Test cases for DriftDetector class."""

    def test_init(self):
        """Test DriftDetector initialization."""
        detector = DriftDetector(window_size=500, significance_level=0.01)
        assert detector.window_size == 500
        assert detector.significance_level == 0.01
        assert detector.reference_data is None

    def test_set_reference(self, sample_drift_data):
        """Test setting reference data."""
        detector = DriftDetector()
        X_ref, _, y_pred_ref, _ = sample_drift_data

        detector.set_reference(X_ref, y_pred_ref)

        assert detector.reference_data is not None
        assert detector.reference_predictions is not None
        assert len(detector.reference_data) == len(X_ref)

    def test_detect_feature_drift(self, sample_drift_data):
        """Test feature drift detection."""
        detector = DriftDetector()
        X_ref, X_curr, y_pred_ref, _ = sample_drift_data

        detector.set_reference(X_ref, y_pred_ref)

        # Test different methods
        for method in ['ks_test', 'js_divergence', 'chi2_test']:
            drift_results = detector.detect_feature_drift(X_curr, method=method)

            assert isinstance(drift_results, dict)
            assert len(drift_results) > 0

            # Check that all features have results
            for feature_name, results in drift_results.items():
                assert 'is_drift' in results
                assert isinstance(results['is_drift'], bool)

    def test_detect_prediction_drift(self, sample_drift_data):
        """Test prediction drift detection."""
        detector = DriftDetector()
        X_ref, _, y_pred_ref, y_pred_curr = sample_drift_data

        detector.set_reference(X_ref, y_pred_ref)

        drift_stats = detector.detect_prediction_drift(y_pred_curr)

        assert isinstance(drift_stats, dict)
        assert 'prediction_drift' in drift_stats
        assert 'mean_prediction_shift' in drift_stats
        assert 'prediction_ks_statistic' in drift_stats
        assert 'prediction_ks_p_value' in drift_stats

    def test_comprehensive_drift_assessment(self, sample_drift_data):
        """Test comprehensive drift assessment."""
        detector = DriftDetector()
        X_ref, X_curr, y_pred_ref, y_pred_curr = sample_drift_data

        detector.set_reference(X_ref, y_pred_ref)

        assessment = detector.comprehensive_drift_assessment(X_curr, y_pred_curr)

        assert isinstance(assessment, dict)
        assert 'feature_drift_ks' in assessment
        assert 'feature_drift_js' in assessment
        assert 'feature_drift_summary' in assessment
        assert 'prediction_drift' in assessment
        assert 'overall_drift_score' in assessment

        # Check drift score range
        assert 0 <= assessment['overall_drift_score'] <= 1

    def test_drift_detection_without_reference(self, sample_drift_data):
        """Test drift detection without setting reference data."""
        detector = DriftDetector()
        _, X_curr, _, y_pred_curr = sample_drift_data

        with pytest.raises(ValueError, match="Reference data not set"):
            detector.detect_feature_drift(X_curr)

        with pytest.raises(ValueError, match="Reference predictions not set"):
            detector.detect_prediction_drift(y_pred_curr)