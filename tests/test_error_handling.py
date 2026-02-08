"""Error handling tests for temporal credit degradation detector.

These tests verify that the system handles various error conditions gracefully
and provides meaningful error messages.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor
from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator
from temporal_credit_degradation_detector.utils.config import Config


class TestDataLoaderErrors:
    """Test error handling in data loading scenarios."""

    def test_invalid_sample_size(self):
        """Test error handling for invalid sample sizes."""
        loader = DataLoader()

        # Negative sample size
        with pytest.raises(ValueError, match="sample_size must be positive"):
            loader.load_home_credit_data(sample_size=-100)

        # Zero sample size
        with pytest.raises(ValueError, match="sample_size must be positive"):
            loader.load_home_credit_data(sample_size=0)

    def test_empty_data_file_handling(self):
        """Test handling of empty data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty CSV file
            empty_file = Path(temp_dir) / "empty.csv"
            empty_file.write_text("")

            loader = DataLoader(data_path=temp_dir)

            # Should fall back to synthetic data
            X, y = loader.load_home_credit_data()
            assert len(X) > 0
            assert len(y) > 0

    def test_corrupted_data_file_handling(self):
        """Test handling of corrupted data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted CSV file
            corrupted_file = Path(temp_dir) / "home_credit.csv"
            corrupted_file.write_text("invalid,csv,content\nwith\ninconsistent\nnumber,of,columns,extra")

            loader = DataLoader(data_path=temp_dir)

            # Should fall back to synthetic data
            X, y = loader.load_home_credit_data()
            assert len(X) > 0
            assert len(y) > 0

    def test_missing_target_column_error(self):
        """Test error when target column is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file without TARGET column
            invalid_file = Path(temp_dir) / "home_credit.csv"
            df = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6]
            })
            df.to_csv(invalid_file, index=False)

            loader = DataLoader(data_path=temp_dir)

            # Should fall back to synthetic data when TARGET missing
            X, y = loader.load_home_credit_data()
            assert len(X) > 0
            assert len(y) > 0

    def test_temporal_splits_invalid_input(self):
        """Test temporal splits with invalid inputs."""
        loader = DataLoader()

        # Empty dataframes
        with pytest.raises(ValueError, match="X and y cannot be empty"):
            loader.create_temporal_splits(pd.DataFrame(), pd.Series(dtype=float))

        # Mismatched lengths
        X = pd.DataFrame({'feature': [1, 2, 3], 'APPLICATION_MONTH': [0, 1, 2]})
        y = pd.Series([0, 1])  # Different length

        with pytest.raises(ValueError, match="X and y must have same length"):
            loader.create_temporal_splits(X, y)

        # Missing time column
        X_no_time = pd.DataFrame({'feature': [1, 2, 3]})
        y_valid = pd.Series([0, 1, 1])

        with pytest.raises(KeyError, match="Time column.*not found"):
            loader.create_temporal_splits(X_no_time, y_valid, time_column='missing_column')

        # Invalid parameters
        X_valid = pd.DataFrame({'feature': [1, 2, 3], 'APPLICATION_MONTH': [0, 1, 2]})

        with pytest.raises(ValueError, match="must be >= 1"):
            loader.create_temporal_splits(X_valid, y_valid, train_months=0)

    def test_insufficient_data_for_splits(self):
        """Test temporal splits with insufficient data."""
        loader = DataLoader()

        # Not enough time periods
        X = pd.DataFrame({
            'feature': [1, 2],
            'APPLICATION_MONTH': [0, 0]  # Only one time period
        })
        y = pd.Series([0, 1])

        with pytest.raises(ValueError, match="Insufficient time periods"):
            loader.create_temporal_splits(X, y, train_months=5, val_months=2, test_months=2)


class TestPreprocessingErrors:
    """Test error handling in data preprocessing."""

    def test_invalid_configuration(self):
        """Test preprocessing with invalid configuration."""
        # Invalid discretization bins (will be caught during preprocessing)
        config = {'discretization_bins': 1}  # Too few bins

        preprocessor = CreditDataPreprocessor(config)

        # Should raise error during initialization of components
        X = pd.DataFrame({'feature': np.random.randn(100)})
        y = pd.Series(np.random.binomial(1, 0.3, 100))

        with pytest.raises(ValueError, match="discretization_bins must be >= 2"):
            preprocessor.fit_transform(X, y)

    def test_transform_before_fit(self):
        """Test transform called before fit."""
        preprocessor = CreditDataPreprocessor()
        X = pd.DataFrame({'feature': [1, 2, 3]})

        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(X)

    def test_empty_dataframe_handling(self):
        """Test preprocessing with empty dataframes."""
        preprocessor = CreditDataPreprocessor()

        # Empty dataframe
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        with pytest.raises((ValueError, IndexError)):
            preprocessor.fit_transform(X_empty, y_empty)

    def test_all_missing_values_column(self):
        """Test handling columns with all missing values."""
        preprocessor = CreditDataPreprocessor()

        X = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'another_good': [5, 4, 3, 2, 1]
        })
        y = pd.Series([0, 1, 0, 1, 0])

        # Should handle gracefully (impute or remove column)
        X_processed = preprocessor.fit_transform(X, y)

        # Should not crash and should handle missing column
        assert len(X_processed) == len(y)

    def test_single_class_target(self):
        """Test preprocessing with single class target."""
        preprocessor = CreditDataPreprocessor()

        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        })
        y = pd.Series([1, 1, 1, 1])  # All same class

        # Should complete but might issue warnings
        X_processed = preprocessor.fit_transform(X, y)
        assert len(X_processed) == len(y)

    def test_incompatible_transform_data(self):
        """Test transform with data incompatible with fitted preprocessor."""
        preprocessor = CreditDataPreprocessor()

        # Fit on original data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_train = pd.Series([0, 1, 0])

        preprocessor.fit_transform(X_train, y_train)

        # Try to transform data with different columns
        X_different = pd.DataFrame({
            'different_feature': [7, 8, 9]
        })

        # Should raise meaningful error
        with pytest.raises(KeyError):
            preprocessor.transform(X_different)

    def test_extreme_values_handling(self):
        """Test preprocessing with extreme values."""
        preprocessor = CreditDataPreprocessor()

        X = pd.DataFrame({
            'normal_feature': [1, 2, 3, 4, 5],
            'extreme_feature': [1e10, -1e10, np.inf, -np.inf, 1e15]
        })
        y = pd.Series([0, 1, 0, 1, 0])

        # Should handle extreme values without crashing
        try:
            X_processed = preprocessor.fit_transform(X, y)
            assert len(X_processed) == len(y)
            # Check no infinite values remain
            assert not np.isinf(X_processed.values).any()
        except (ValueError, OverflowError):
            # Acceptable to raise error for extreme values
            pass


class TestModelErrors:
    """Test error handling in model operations."""

    def test_invalid_model_parameters(self):
        """Test model initialization with invalid parameters."""
        # Invalid stability_alpha
        with pytest.raises(ValueError, match="stability_alpha must be in \\(0,1\\]"):
            model = StabilityWeightedEnsemble(stability_alpha=0)
            model._validate_config()

        with pytest.raises(ValueError, match="stability_alpha must be in \\(0,1\\]"):
            model = StabilityWeightedEnsemble(stability_alpha=1.5)
            model._validate_config()

        # Invalid min_weight
        with pytest.raises(ValueError, match="min_weight must be in \\[0,1\\)"):
            model = StabilityWeightedEnsemble(min_weight=-0.1)
            model._validate_config()

        with pytest.raises(ValueError, match="min_weight must be in \\[0,1\\)"):
            model = StabilityWeightedEnsemble(min_weight=1.5)
            model._validate_config()

        # Invalid calibration_window
        with pytest.raises(ValueError, match="calibration_window must be positive"):
            model = StabilityWeightedEnsemble(calibration_window=0)
            model._validate_config()

    def test_predict_before_fit(self):
        """Test prediction before model is fitted."""
        model = StabilityWeightedEnsemble()
        X = pd.DataFrame({'feature': [1, 2, 3]})

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict_proba(X)

    def test_fit_with_invalid_data(self):
        """Test fitting with invalid data."""
        model = StabilityWeightedEnsemble()

        # Empty data
        with pytest.raises((ValueError, IndexError)):
            model.fit(pd.DataFrame(), pd.Series(dtype=float))

        # Mismatched lengths
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([0, 1])  # Different length

        with pytest.raises(ValueError):
            model.fit(X, y)

        # Single sample
        X_single = pd.DataFrame({'feature': [1]})
        y_single = pd.Series([0])

        # Should either work or raise meaningful error
        try:
            model.fit(X_single, y_single)
        except ValueError as e:
            assert "sample" in str(e).lower() or "insufficient" in str(e).lower()

    def test_fit_with_single_class(self):
        """Test fitting with single class target."""
        model = StabilityWeightedEnsemble()

        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10]
        })
        y = pd.Series([1, 1, 1, 1, 1])  # All same class

        # Should raise meaningful error or handle gracefully
        with pytest.raises((ValueError, Exception)) as exc_info:
            model.fit(X, y)

        # Error message should indicate the issue
        assert any(word in str(exc_info.value).lower()
                  for word in ['class', 'label', 'target', 'binary'])

    def test_predict_with_wrong_features(self):
        """Test prediction with wrong number of features."""
        model = StabilityWeightedEnsemble()

        # Train on 3 features
        X_train = pd.DataFrame({
            'f1': [1, 2, 3, 4],
            'f2': [5, 6, 7, 8],
            'f3': [9, 10, 11, 12]
        })
        y_train = pd.Series([0, 1, 0, 1])

        model.fit(X_train, y_train)

        # Try to predict with 2 features
        X_pred = pd.DataFrame({
            'f1': [1, 2],
            'f2': [3, 4]
            # Missing f3
        })

        with pytest.raises((ValueError, KeyError)):
            model.predict_proba(X_pred)

    def test_model_save_load_errors(self):
        """Test model save/load error handling."""
        model = StabilityWeightedEnsemble()

        # Try to save unfitted model
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="not fitted"):
                model.save_model(temp_file.name)

        # Try to load non-existent file
        with pytest.raises(FileNotFoundError):
            StabilityWeightedEnsemble.load_model("/nonexistent/path.pkl")

        # Try to load corrupted file
        with tempfile.NamedTemporaryFile(mode='w') as temp_file:
            temp_file.write("corrupted content")
            temp_file.flush()

            with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
                StabilityWeightedEnsemble.load_model(temp_file.name)


class TestEvaluationErrors:
    """Test error handling in evaluation scenarios."""

    def test_metrics_with_invalid_inputs(self):
        """Test metrics calculation with invalid inputs."""
        evaluator = ModelEvaluator()

        # Mismatched lengths
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Different length

        with pytest.raises(ValueError, match="Mismatched lengths"):
            evaluator.calculate_metrics(y_true, y_pred)

        # Empty arrays
        with pytest.raises(ValueError, match="Empty input"):
            evaluator.calculate_metrics(np.array([]), np.array([]))

        # Invalid probability values
        y_true_valid = np.array([0, 1, 0, 1])
        y_pred_invalid = np.array([0.5, 1.5, -0.1, 0.8])  # Outside [0,1]

        with pytest.raises(ValueError, match="y_prob values must be in \\[0,1\\]"):
            evaluator.calculate_metrics(y_true_valid, y_pred_invalid)

        # NaN values
        y_pred_nan = np.array([0.5, np.nan, 0.8, 0.2])

        with pytest.raises(ValueError, match="NaN or infinite"):
            evaluator.calculate_metrics(y_true_valid, y_pred_nan)

        # Invalid target values
        y_true_invalid = np.array([0, 1, 2, 3])  # Not binary

        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            evaluator.calculate_metrics(y_true_invalid, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_single_class_evaluation(self):
        """Test evaluation with single class."""
        evaluator = ModelEvaluator()

        # All zeros
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # Should issue warning but still return metrics
        with pytest.warns(UserWarning):
            metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert 'auc_roc' in metrics
        # AUC should be undefined for single class, may return 0.5 or NaN

    def test_drift_detection_errors(self):
        """Test drift detection error handling."""
        evaluator = ModelEvaluator()

        # Try drift detection without reference
        X_current = pd.DataFrame({'feature': [1, 2, 3]})
        y_current = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="reference"):
            evaluator.detect_drift(X_current, y_current)

        # Set reference with mismatched features
        X_ref = pd.DataFrame({'different_feature': [4, 5, 6]})
        y_ref = pd.Series([1, 0, 1])

        evaluator.drift_detector.set_reference(X_ref, y_ref)

        # Try to detect drift with different features
        with pytest.raises(ValueError):
            evaluator.detect_drift(X_current, y_current)

    def test_business_metrics_edge_cases(self):
        """Test business metrics with edge cases."""
        evaluator = ModelEvaluator()

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred_perfect = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = evaluator.calculate_business_metrics(y_true, y_pred_perfect)
        assert isinstance(metrics, dict)

        # All same prediction
        y_pred_same = np.array([0.5, 0.5, 0.5, 0.5])

        metrics_same = evaluator.calculate_business_metrics(y_true, y_pred_same)
        assert isinstance(metrics_same, dict)


class TestSystemErrors:
    """Test system-level error handling."""

    def test_memory_constraints_simulation(self):
        """Test behavior under simulated memory constraints."""
        # Create large-ish dataset to stress test
        n_samples = 10000  # Reasonable size for testing

        X = pd.DataFrame(
            np.random.randn(n_samples, 50),
            columns=[f'feature_{i}' for i in range(50)]
        )
        X['APPLICATION_MONTH'] = np.random.randint(0, 24, n_samples)
        y = pd.Series(np.random.binomial(1, 0.1, n_samples))

        # Test preprocessing on larger dataset
        preprocessor = CreditDataPreprocessor()

        try:
            X_prep = preprocessor.fit_transform(X, y)
            assert len(X_prep) == len(y)

            # Test model training
            model = StabilityWeightedEnsemble(random_state=42)
            model.fit(X_prep, y)

            # Test predictions
            y_pred = model.predict_proba(X_prep)[:, 1]
            assert len(y_pred) == len(y)

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_concurrent_access_safety(self):
        """Test thread safety for model operations."""
        import threading

        model = StabilityWeightedEnsemble(random_state=42)

        # Train model
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.binomial(1, 0.3, 100))
        model.fit(X, y)

        # Test concurrent predictions
        results = {}
        errors = []

        def make_prediction(thread_id):
            try:
                X_test = pd.DataFrame(np.random.randn(10, 5))
                pred = model.predict_proba(X_test)
                results[thread_id] = pred
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_prediction, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        if errors:
            # If there are thread safety issues, they should be documented
            pytest.fail(f"Thread safety errors: {errors}")

        assert len(results) == 5  # All predictions should succeed

    def test_disk_space_handling(self):
        """Test handling when disk space is limited."""
        # This is hard to test without actually filling disk
        # Instead, test save to invalid path
        model = StabilityWeightedEnsemble()

        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.binomial(1, 0.4, 50))
        model.fit(X, y)

        # Try to save to invalid directory
        invalid_path = "/invalid/directory/that/does/not/exist/model.pkl"

        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            model.save_model(invalid_path)