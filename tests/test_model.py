"""Tests for the stability-weighted ensemble model."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from temporal_credit_degradation_detector.models.model import (
    StabilityWeightedEnsemble,
    CalibrationMonitor
)


class TestCalibrationMonitor:
    """Test cases for CalibrationMonitor class."""

    def test_init(self):
        """Test CalibrationMonitor initialization."""
        monitor = CalibrationMonitor(window_size=500)
        assert monitor.window_size == 500
        assert len(monitor.calibration_history) == 0
        assert len(monitor.predictions_buffer) == 0

    def test_update_insufficient_data(self):
        """Test update with insufficient data."""
        monitor = CalibrationMonitor(window_size=1000)

        y_prob = np.array([0.1, 0.3, 0.7])
        y_true = np.array([0, 0, 1])

        brier_score = monitor.update(y_prob, y_true)
        assert brier_score == 0.0  # Insufficient data

    def test_update_sufficient_data(self):
        """Test update with sufficient data."""
        monitor = CalibrationMonitor(window_size=200)

        # Generate sufficient data
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 150)
        y_prob = np.random.beta(2, 5, 150)

        brier_score = monitor.update(y_prob, y_true)

        assert isinstance(brier_score, float)
        assert 0 <= brier_score <= 1
        assert len(monitor.calibration_history) == 1

    def test_window_size_maintenance(self):
        """Test that window size is maintained."""
        monitor = CalibrationMonitor(window_size=100)

        # Add more data than window size
        for _ in range(5):
            y_prob = np.random.random(50)
            y_true = np.random.binomial(1, 0.5, 50)
            monitor.update(y_prob, y_true)

        assert len(monitor.predictions_buffer) <= 100

    def test_get_calibration_trend(self):
        """Test calibration trend calculation."""
        monitor = CalibrationMonitor(window_size=1000)

        # Simulate deteriorating calibration
        for i in range(15):
            # Gradually worsen calibration
            noise_level = 0.1 + i * 0.05
            y_true = np.random.binomial(1, 0.3, 100)
            y_prob = y_true + np.random.normal(0, noise_level, 100)
            y_prob = np.clip(y_prob, 0.01, 0.99)

            monitor.update(y_prob, y_true)

        trend = monitor.get_calibration_trend(recent_window=10)
        assert isinstance(trend, float)


class TestStabilityWeightedEnsemble:
    """Test cases for StabilityWeightedEnsemble class."""

    def test_init_default(self):
        """Test ensemble initialization with default parameters."""
        ensemble = StabilityWeightedEnsemble()

        assert ensemble.calibration_window == 1000
        assert ensemble.stability_alpha == 0.1
        assert ensemble.min_weight == 0.05
        assert ensemble.recalibration_threshold == 0.15
        assert ensemble.random_state == 42
        assert not ensemble.is_fitted_

    def test_init_custom_params(self):
        """Test ensemble initialization with custom parameters."""
        ensemble = StabilityWeightedEnsemble(
            stability_alpha=0.2,
            min_weight=0.1,
            calibration_window=500,
            random_state=123
        )

        assert ensemble.stability_alpha == 0.2
        assert ensemble.min_weight == 0.1
        assert ensemble.calibration_window == 500
        assert ensemble.random_state == 123

    def test_get_default_models(self):
        """Test default model creation."""
        ensemble = StabilityWeightedEnsemble()
        models = ensemble._get_default_models()

        assert len(models) == 4  # 2 LightGBM + 2 CatBoost
        assert all(hasattr(model, 'fit') for model in models)
        assert all(hasattr(model, 'predict') for model in models)

    def test_create_stability_features(self, sample_credit_data):
        """Test stability feature creation."""
        ensemble = StabilityWeightedEnsemble()
        X, _ = sample_credit_data

        X_stable = ensemble._create_stability_features(X)

        # Should have more or equal features
        assert X_stable.shape[1] >= X.shape[1]
        assert X_stable.shape[0] == X.shape[0]

        # Should have ratio and magnitude features
        ratio_cols = [col for col in X_stable.columns if 'ratio' in col]
        magnitude_cols = [col for col in X_stable.columns if 'magnitude' in col]

        assert len(ratio_cols) > 0
        assert len(magnitude_cols) > 0

    def test_fit(self, stability_ensemble, sample_temporal_data, assert_helpers):
        """Test ensemble fitting."""
        X_train, y_train = sample_temporal_data['train']

        # Fit the ensemble
        fitted_ensemble = stability_ensemble.fit(X_train, y_train)

        # Check that it returns self
        assert fitted_ensemble is stability_ensemble
        assert stability_ensemble.is_fitted_

        # Check that components are initialized
        assert len(stability_ensemble.models_) > 0
        assert len(stability_ensemble.calibrators_) > 0
        assert len(stability_ensemble.monitors_) > 0
        assert len(stability_ensemble.weights_) == len(stability_ensemble.models_)

        # Weights should sum to 1
        assert np.isclose(stability_ensemble.weights_.sum(), 1.0)

        # All weights should be positive
        assert np.all(stability_ensemble.weights_ > 0)

    def test_predict_proba(self, fitted_ensemble, sample_temporal_data, assert_helpers):
        """Test probability prediction."""
        X_test, _ = sample_temporal_data['test']

        y_prob = fitted_ensemble.predict_proba(X_test)

        # Check output format
        assert y_prob.shape == (len(X_test), 2)
        assert_helpers['assert_valid_probabilities'](y_prob[:, 1])

        # Probabilities should sum to 1
        assert np.allclose(y_prob.sum(axis=1), 1.0)

    def test_predict(self, fitted_ensemble, sample_temporal_data, assert_helpers):
        """Test class prediction."""
        X_test, _ = sample_temporal_data['test']

        y_pred = fitted_ensemble.predict(X_test)

        # Check output format
        assert len(y_pred) == len(X_test)
        assert_helpers['assert_valid_predictions'](y_pred)

    def test_predict_without_fit(self, stability_ensemble, sample_temporal_data):
        """Test prediction without fitting first."""
        X_test, _ = sample_temporal_data['test']

        with pytest.raises(ValueError, match="Model must be fitted"):
            stability_ensemble.predict_proba(X_test)

    def test_update_weights(self, fitted_ensemble, sample_temporal_data):
        """Test weight updating mechanism."""
        X_val, y_val = sample_temporal_data['val']

        # Get initial weights
        initial_weights = fitted_ensemble.weights_.copy()

        # Update weights
        stats = fitted_ensemble.update_weights(X_val, y_val, update_monitors=True)

        # Check return statistics
        assert isinstance(stats, dict)
        assert 'max_brier_score' in stats
        assert 'mean_auc' in stats
        assert 'weight_entropy' in stats
        assert 'needs_recalibration' in stats
        assert 'dominant_model' in stats

        # Weights should still sum to 1
        assert np.isclose(fitted_ensemble.weights_.sum(), 1.0)

        # All weights should be above minimum
        assert np.all(fitted_ensemble.weights_ >= fitted_ensemble.min_weight)

    def test_get_feature_importance(self, fitted_ensemble):
        """Test feature importance extraction."""
        importance = fitted_ensemble.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0

        # All importance scores should be non-negative
        assert all(score >= 0 for score in importance.values())

        # Importance scores should sum to approximately 1
        assert np.isclose(sum(importance.values()), 1.0, atol=1e-3)

    def test_save_load_model(self, fitted_ensemble, temp_dir):
        """Test model saving and loading."""
        model_path = Path(temp_dir) / "test_model.pkl"

        # Save model
        fitted_ensemble.save_model(str(model_path))
        assert model_path.exists()

        # Load model
        loaded_ensemble = StabilityWeightedEnsemble.load_model(str(model_path))

        # Check that loaded model has same state
        assert loaded_ensemble.is_fitted_
        assert len(loaded_ensemble.models_) == len(fitted_ensemble.models_)
        assert len(loaded_ensemble.calibrators_) == len(fitted_ensemble.calibrators_)
        assert np.array_equal(loaded_ensemble.weights_, fitted_ensemble.weights_)
        assert loaded_ensemble.feature_names_ == fitted_ensemble.feature_names_

    def test_save_without_fit(self, stability_ensemble, temp_dir):
        """Test saving unfitted model."""
        model_path = Path(temp_dir) / "unfitted_model.pkl"

        with pytest.raises(ValueError, match="Model must be fitted"):
            stability_ensemble.save_model(str(model_path))

    def test_reproducibility(self, sample_temporal_data):
        """Test that model training is reproducible."""
        X_train, y_train = sample_temporal_data['train']
        X_test, _ = sample_temporal_data['test']

        # Train two models with same random state
        ensemble1 = StabilityWeightedEnsemble(random_state=42)
        ensemble2 = StabilityWeightedEnsemble(random_state=42)

        ensemble1.fit(X_train, y_train)
        ensemble2.fit(X_train, y_train)

        # Predictions should be identical
        pred1 = ensemble1.predict_proba(X_test)
        pred2 = ensemble2.predict_proba(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_different_random_states(self, sample_temporal_data):
        """Test that different random states produce different results."""
        X_train, y_train = sample_temporal_data['train']
        X_test, _ = sample_temporal_data['test']

        # Train two models with different random states
        ensemble1 = StabilityWeightedEnsemble(random_state=42)
        ensemble2 = StabilityWeightedEnsemble(random_state=123)

        ensemble1.fit(X_train, y_train)
        ensemble2.fit(X_train, y_train)

        # Predictions should be different
        pred1 = ensemble1.predict_proba(X_test)[:, 1]
        pred2 = ensemble2.predict_proba(X_test)[:, 1]

        # Should not be identical (with high probability)
        assert not np.allclose(pred1, pred2, rtol=1e-3)

    def test_numpy_array_input(self, stability_ensemble, sample_temporal_data):
        """Test that model works with numpy arrays."""
        X_train, y_train = sample_temporal_data['train']
        X_test, _ = sample_temporal_data['test']

        # Convert to numpy arrays
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_test_np = X_test.values

        # Fit and predict
        stability_ensemble.fit(X_train_np, y_train_np)
        y_prob = stability_ensemble.predict_proba(X_test_np)

        assert y_prob.shape == (len(X_test_np), 2)

    def test_single_sample_prediction(self, fitted_ensemble, sample_temporal_data):
        """Test prediction on single sample."""
        X_test, _ = sample_temporal_data['test']

        # Take first sample
        X_single = X_test.iloc[:1]

        y_prob = fitted_ensemble.predict_proba(X_single)
        y_pred = fitted_ensemble.predict(X_single)

        assert y_prob.shape == (1, 2)
        assert len(y_pred) == 1

    def test_large_dataset_performance(self, stability_ensemble):
        """Test model performance on larger dataset."""
        # Generate larger synthetic dataset
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=5000,
            n_features=30,
            n_informative=20,
            n_classes=2,
            random_state=42
        )

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(30)])
        y_series = pd.Series(y)

        # Should handle larger datasets without issues
        stability_ensemble.fit(X_df, y_series)
        y_prob = stability_ensemble.predict_proba(X_df)

        assert y_prob.shape == (5000, 2)

    def test_extreme_weight_scenarios(self, fitted_ensemble, sample_temporal_data):
        """Test weight updates in extreme scenarios."""
        X_val, y_val = sample_temporal_data['val']

        # Simulate extreme performance degradation
        # Create very bad predictions (opposite of true labels)
        y_bad = 1 - y_val  # Flip labels

        # This should trigger significant weight updates
        stats = fitted_ensemble.update_weights(X_val, y_bad, update_monitors=True)

        # Should indicate need for recalibration
        assert stats['needs_recalibration'] == True
        assert stats['max_brier_score'] > fitted_ensemble.recalibration_threshold

    def test_weight_minimum_constraint(self, fitted_ensemble, sample_temporal_data):
        """Test that minimum weight constraint is enforced."""
        X_val, y_val = sample_temporal_data['val']

        # Force many weight updates to see if any weights go below minimum
        for _ in range(10):
            fitted_ensemble.update_weights(X_val, y_val, update_monitors=True)

        # All weights should be above minimum
        assert np.all(fitted_ensemble.weights_ >= fitted_ensemble.min_weight)

    def test_edge_case_constant_predictions(self, stability_ensemble):
        """Test model with constant prediction scenario."""
        # Create data where model might predict constant values
        X = pd.DataFrame({
            'feature_1': [1, 1, 1, 1, 1],
            'feature_2': [2, 2, 2, 2, 2]
        })
        y = pd.Series([0, 0, 0, 0, 0])

        # Should not crash even with constant target
        stability_ensemble.fit(X, y)
        y_prob = stability_ensemble.predict_proba(X)

        assert y_prob.shape == (5, 2)

    def test_edge_case_single_class(self, stability_ensemble):
        """Test model with single class in training data."""
        # Create data with only one class
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [6, 7, 8, 9, 10]
        })
        y = pd.Series([1, 1, 1, 1, 1])  # Only positive class

        # Should handle gracefully
        stability_ensemble.fit(X, y)
        y_prob = stability_ensemble.predict_proba(X)

        assert y_prob.shape == (5, 2)
        # All predictions should be high for positive class
        assert np.all(y_prob[:, 1] > 0.5)

    def test_feature_importance_without_fit(self, stability_ensemble):
        """Test feature importance extraction without fitting."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            stability_ensemble.get_feature_importance()