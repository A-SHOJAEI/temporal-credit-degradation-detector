"""Integration tests for temporal credit degradation detector.

These tests verify that different components work together correctly
in realistic end-to-end scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor
from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble
from temporal_credit_degradation_detector.training.trainer import ModelTrainer
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator
from temporal_credit_degradation_detector.utils.config import Config


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline workflows."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline from data loading to model evaluation."""
        # Initialize components
        config = Config()
        loader = DataLoader()
        preprocessor = CreditDataPreprocessor(config.preprocessing)
        evaluator = ModelEvaluator()

        # Load data
        X, y = loader.load_home_credit_data(sample_size=500)

        # Create temporal splits
        splits = loader.create_temporal_splits(X, y, train_months=8, val_months=2, test_months=4)

        # Get training and test data
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']

        # Preprocess training data
        X_train_prep = preprocessor.fit_transform(X_train, y_train)
        X_test_prep = preprocessor.transform(X_test)

        # Train model
        model = StabilityWeightedEnsemble(
            stability_alpha=0.1,
            calibration_window=100,
            random_state=42
        )
        model.fit(X_train_prep, y_train)

        # Make predictions
        y_pred_proba = model.predict_proba(X_test_prep)[:, 1]

        # Evaluate
        metrics = evaluator.calculate_metrics(y_test, y_pred_proba)

        # Assertions
        assert len(X_train_prep) > 0
        assert len(X_test_prep) > 0
        assert len(y_pred_proba) == len(y_test)
        assert 'auc_roc' in metrics
        assert 0 <= metrics['auc_roc'] <= 1
        assert model.is_fitted_

    def test_preprocessing_model_compatibility(self):
        """Test that preprocessed data is compatible with model training."""
        # Load data
        loader = DataLoader()
        X, y = loader.load_lending_club_data(sample_size=300)

        # Different preprocessing configurations
        configs = [
            {'create_temporal_features': True, 'create_risk_features': True},
            {'create_temporal_features': False, 'create_risk_features': True},
            {'discretize_continuous': True, 'scale_features': True},
        ]

        for config_dict in configs:
            preprocessor = CreditDataPreprocessor(config_dict)
            X_prep = preprocessor.fit_transform(X, y)

            # Train model on preprocessed data
            model = StabilityWeightedEnsemble(random_state=42)
            model.fit(X_prep, y)

            # Make predictions
            y_pred = model.predict_proba(X_prep)

            # Assertions
            assert y_pred.shape[0] == len(y)
            assert y_pred.shape[1] == 2  # Binary classification
            assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

    def test_temporal_splits_with_preprocessing(self):
        """Test temporal splits work correctly with preprocessing pipeline."""
        loader = DataLoader()
        X, y = loader.load_home_credit_data(sample_size=800)

        # Create temporal splits
        splits = loader.create_temporal_splits(X, y, train_months=10, val_months=3, test_months=6)

        # Initialize preprocessor
        preprocessor = CreditDataPreprocessor()

        # Process each split
        processed_splits = {}
        for split_name, (X_split, y_split) in splits.items():
            if split_name == 'train':
                # Fit on training data
                X_processed = preprocessor.fit_transform(X_split, y_split)
            else:
                # Transform validation/test data
                X_processed = preprocessor.transform(X_split)

            processed_splits[split_name] = (X_processed, y_split)

        # Verify splits are properly processed
        for split_name, (X_proc, y_split) in processed_splits.items():
            assert len(X_proc) > 0, f"{split_name} split is empty"
            assert len(X_proc) == len(y_split), f"{split_name} split length mismatch"
            assert not X_proc.isna().all().any(), f"{split_name} has all-NaN columns"

        # Verify feature consistency across splits
        train_features = set(processed_splits['train'][0].columns)
        for split_name in ['val', 'test']:
            split_features = set(processed_splits[split_name][0].columns)
            assert train_features == split_features, f"Feature mismatch in {split_name}"

    def test_model_persistence_integration(self):
        """Test model saving/loading with preprocessor integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            preprocessor_path = Path(temp_dir) / "test_preprocessor.pkl"

            # Train original model
            loader = DataLoader()
            X, y = loader.load_home_credit_data(sample_size=400)

            preprocessor = CreditDataPreprocessor()
            X_prep = preprocessor.fit_transform(X, y)

            model = StabilityWeightedEnsemble(random_state=42)
            model.fit(X_prep, y)

            # Save model and preprocessor
            model.save_model(str(model_path))
            import joblib
            joblib.dump(preprocessor, preprocessor_path)

            # Create new data for testing
            X_new, y_new = loader.load_home_credit_data(sample_size=200)

            # Load model and preprocessor
            loaded_model = StabilityWeightedEnsemble.load_model(str(model_path))
            loaded_preprocessor = joblib.load(preprocessor_path)

            # Test prediction pipeline
            X_new_prep = loaded_preprocessor.transform(X_new)
            y_pred = loaded_model.predict_proba(X_new_prep)

            # Verify predictions
            assert y_pred.shape[0] == len(y_new)
            assert y_pred.shape[1] == 2
            assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

    def test_drift_detection_workflow(self):
        """Test drift detection in realistic scenario."""
        loader = DataLoader()
        evaluator = ModelEvaluator()

        # Generate reference and current data
        X_ref, y_ref = loader.load_home_credit_data(sample_size=300)
        X_current, y_current = loader.load_lending_club_data(sample_size=200)

        # Align features (simple approach - use common subset)
        common_features = ['FEATURE_22', 'FEATURE_23', 'FEATURE_24', 'FEATURE_25', 'FEATURE_26']

        # Create aligned datasets with common features
        X_ref_aligned = pd.DataFrame({
            feat: X_ref.iloc[:, i % X_ref.shape[1]] for i, feat in enumerate(common_features)
        })
        X_current_aligned = pd.DataFrame({
            feat: X_current.iloc[:, i % X_current.shape[1]] for i, feat in enumerate(common_features)
        })

        # Set reference data for drift detection
        evaluator.drift_detector.set_reference(X_ref_aligned, y_ref)

        # Detect drift
        drift_results = evaluator.detect_drift(X_current_aligned, y_current)

        # Verify drift detection results
        assert isinstance(drift_results, dict)
        assert 'feature_drift' in drift_results
        assert 'prediction_drift' in drift_results
        assert 'comprehensive_drift' in drift_results

        # Check that drift scores are reasonable
        for drift_type, scores in drift_results.items():
            if isinstance(scores, dict):
                for score in scores.values():
                    if isinstance(score, (int, float)):
                        assert not np.isnan(score), f"NaN drift score in {drift_type}"


class TestCrossValidationIntegration:
    """Test cross-validation with full pipeline integration."""

    def test_cross_validation_with_preprocessing(self):
        """Test cross-validation includes preprocessing in each fold."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'random_state': 42,
                'model_dir': temp_dir,
                'experiment_name': 'test_cv'
            }

            # Initialize trainer
            trainer = ModelTrainer(config)

            # Load data
            loader = DataLoader()
            X, y = loader.load_home_credit_data(sample_size=600)

            # Run cross-validation
            results = trainer.train_cross_validation(
                X, y,
                cv_folds=3,
                optimize_hp=False,  # Skip HP optimization for speed
                n_hp_trials=1
            )

            # Verify results
            assert 'cv_scores' in results
            assert 'avg_metrics' in results
            assert 'best_fold' in results
            assert len(results['cv_scores']) == 3  # 3 folds

            # Check that all folds have reasonable metrics
            for fold_metrics in results['cv_scores']:
                assert 'auc_roc' in fold_metrics
                assert 0 <= fold_metrics['auc_roc'] <= 1

            # Check model was saved
            assert trainer.model is not None
            assert trainer.model.is_fitted_

    def test_early_stopping_integration(self):
        """Test early stopping with model training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'random_state': 42,
                'model_dir': temp_dir,
                'experiment_name': 'test_early_stop'
            }

            trainer = ModelTrainer(config)
            loader = DataLoader()
            X, y = loader.load_home_credit_data(sample_size=400)

            # Split data
            splits = loader.create_temporal_splits(X, y, train_months=8, val_months=4, test_months=4)
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']

            # Train with early stopping
            results = trainer.train_with_early_stopping(
                X_train, y_train, X_val, y_val,
                max_updates=10,
                patience=3
            )

            # Verify early stopping results
            assert 'history' in results
            assert 'stopped_early' in results
            assert 'best_score' in results
            assert 'final_model' in results

            # Check that training history exists
            assert len(results['history']) > 0
            for epoch_metrics in results['history']:
                assert 'auc_roc' in epoch_metrics


class TestDataQualityIntegration:
    """Test integration with various data quality scenarios."""

    def test_high_missing_data_scenario(self):
        """Test pipeline with high missing data rates."""
        # Create data with high missing rates
        np.random.seed(42)
        n_samples, n_features = 300, 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Introduce high missing rates (30-70% missing)
        for col in X.columns:
            missing_rate = np.random.uniform(0.3, 0.7)
            missing_mask = np.random.random(n_samples) < missing_rate
            X.loc[missing_mask, col] = np.nan

        # Add temporal column and target
        X['APPLICATION_MONTH'] = np.random.randint(0, 12, n_samples)
        y = pd.Series(np.random.binomial(1, 0.15, n_samples))

        # Test preprocessing pipeline
        preprocessor = CreditDataPreprocessor()
        X_clean = X.drop(columns=['APPLICATION_MONTH'])  # Remove temporal for preprocessing
        X_prep = preprocessor.fit_transform(X_clean, y)

        # Verify no missing values after preprocessing
        assert not X_prep.isna().any().any(), "Missing values remain after preprocessing"

        # Test model training
        model = StabilityWeightedEnsemble(random_state=42)
        model.fit(X_prep, y)

        # Test predictions
        y_pred = model.predict_proba(X_prep)[:, 1]
        assert len(y_pred) == len(y)
        assert not np.isnan(y_pred).any()

    def test_class_imbalance_scenario(self):
        """Test pipeline with severe class imbalance."""
        loader = DataLoader()
        X, y = loader.load_home_credit_data(sample_size=500)

        # Create severe imbalance (2% positive class)
        n_positive = int(0.02 * len(y))
        y_imbalanced = pd.Series(np.zeros(len(y)))
        y_imbalanced.iloc[:n_positive] = 1
        y_imbalanced = y_imbalanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # Test preprocessing
        preprocessor = CreditDataPreprocessor()
        X_prep = preprocessor.fit_transform(X, y_imbalanced)

        # Test model training
        model = StabilityWeightedEnsemble(random_state=42)
        model.fit(X_prep, y_imbalanced)

        # Test predictions
        y_pred = model.predict_proba(X_prep)[:, 1]

        # Verify model handles imbalance reasonably
        assert len(y_pred) == len(y_imbalanced)
        assert not np.all(y_pred == y_pred[0])  # Not all predictions identical

        # Test evaluation
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_imbalanced, y_pred)

        # Should still produce reasonable metrics
        assert 'auc_roc' in metrics
        assert not np.isnan(metrics['auc_roc'])

    def test_small_dataset_scenario(self):
        """Test pipeline with very small dataset."""
        # Create minimal viable dataset
        n_samples = 50  # Very small
        n_features = 10

        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X['APPLICATION_MONTH'] = np.random.randint(0, 6, n_samples)  # 6 months
        y = pd.Series(np.random.binomial(1, 0.3, n_samples))

        # Test temporal splits with small data
        loader = DataLoader()
        splits = loader.create_temporal_splits(
            X, y,
            train_months=3,
            val_months=1,
            test_months=2
        )

        # Test preprocessing on small data
        X_train, y_train = splits['train']

        if len(X_train) > 5:  # Only proceed if we have minimal data
            preprocessor = CreditDataPreprocessor()
            X_prep = preprocessor.fit_transform(X_train, y_train)

            # Test model training on small data
            if len(X_prep) >= 10 and y_train.nunique() == 2:  # Need minimal samples and both classes
                model = StabilityWeightedEnsemble(
                    calibration_window=min(100, len(X_prep) // 2),  # Adjust for small data
                    random_state=42
                )
                model.fit(X_prep, y_train)

                # Verify model can make predictions
                y_pred = model.predict_proba(X_prep)
                assert y_pred.shape[0] == len(y_train)


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_preprocessing_model_feature_mismatch(self):
        """Test handling of feature mismatches between preprocessing and model."""
        loader = DataLoader()
        X, y = loader.load_home_credit_data(sample_size=200)

        # Train preprocessor and model on original data
        preprocessor = CreditDataPreprocessor()
        X_prep = preprocessor.fit_transform(X, y)

        model = StabilityWeightedEnsemble(random_state=42)
        model.fit(X_prep, y)

        # Create new data with different features
        X_new = pd.DataFrame({
            'different_feature_1': np.random.randn(100),
            'different_feature_2': np.random.randn(100),
            'APPLICATION_MONTH': np.random.randint(0, 12, 100)
        })

        # Test that preprocessing handles missing features gracefully
        # Note: This will likely fail without proper error handling
        # The test verifies that we get a meaningful error message
        with pytest.raises((KeyError, ValueError)) as exc_info:
            X_new_prep = preprocessor.transform(X_new)

        # Verify we get a meaningful error message
        assert "feature" in str(exc_info.value).lower() or "column" in str(exc_info.value).lower()

    def test_model_prediction_edge_cases(self):
        """Test model predictions with edge case inputs."""
        loader = DataLoader()
        X, y = loader.load_home_credit_data(sample_size=200)

        # Train normal model
        preprocessor = CreditDataPreprocessor()
        X_prep = preprocessor.fit_transform(X, y)

        model = StabilityWeightedEnsemble(random_state=42)
        model.fit(X_prep, y)

        # Test prediction on single sample
        X_single = X_prep.iloc[:1]
        y_pred_single = model.predict_proba(X_single)
        assert y_pred_single.shape == (1, 2)

        # Test prediction on empty data (should raise error)
        X_empty = X_prep.iloc[:0]

        if len(X_empty) == 0:
            # Empty prediction should either work (return empty) or raise meaningful error
            try:
                y_pred_empty = model.predict_proba(X_empty)
                assert len(y_pred_empty) == 0
            except ValueError:
                pass  # Acceptable to raise error for empty input

    def test_evaluation_with_edge_cases(self):
        """Test evaluation with edge case predictions."""
        evaluator = ModelEvaluator()

        # Test with perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred_perfect = np.array([0.0, 0.0, 1.0, 1.0])

        metrics_perfect = evaluator.calculate_metrics(y_true, y_pred_perfect)
        assert metrics_perfect['auc_roc'] == 1.0

        # Test with random predictions
        y_pred_random = np.array([0.5, 0.5, 0.5, 0.5])
        metrics_random = evaluator.calculate_metrics(y_true, y_pred_random)
        assert 'auc_roc' in metrics_random

        # Test with single class
        y_true_single = np.array([1, 1, 1, 1])
        y_pred_single_class = np.array([0.8, 0.7, 0.9, 0.6])

        # Should handle single class gracefully
        with pytest.warns(UserWarning):  # May warn about single class
            metrics_single = evaluator.calculate_metrics(y_true_single, y_pred_single_class)
        assert 'auc_roc' in metrics_single