"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init_default(self):
        """Test DataLoader initialization with default parameters."""
        loader = DataLoader()
        assert loader.data_path is None

    def test_init_with_path(self, temp_dir):
        """Test DataLoader initialization with custom path."""
        loader = DataLoader(temp_dir)
        assert str(loader.data_path) == temp_dir

    def test_load_home_credit_synthetic(self):
        """Test loading synthetic Home Credit data."""
        loader = DataLoader()
        X, y = loader.load_home_credit_data(sample_size=1000)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 1000
        assert len(y) == 1000
        assert y.name == 'TARGET'
        assert X.shape[1] > 20  # Should have multiple features

        # Check for temporal column
        assert 'APPLICATION_MONTH' in X.columns

        # Check target distribution (should be realistic for credit data)
        default_rate = y.mean()
        assert 0.05 <= default_rate <= 0.5, f"Default rate {default_rate} seems unrealistic"

    def test_load_lending_club_synthetic(self):
        """Test loading synthetic Lending Club data."""
        loader = DataLoader()
        X, y = loader.load_lending_club_data(sample_size=800)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 800
        assert len(y) == 800
        assert y.name == 'loan_status'

        # Check for temporal column
        assert 'ISSUE_MONTH' in X.columns

    def test_load_home_credit_from_file(self, mock_data_files):
        """Test loading Home Credit data from actual files."""
        loader = DataLoader(mock_data_files)
        X, y = loader.load_home_credit_data()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0

    def test_load_lending_club_from_file(self, mock_data_files):
        """Test loading Lending Club data from actual files."""
        loader = DataLoader(mock_data_files)
        X, y = loader.load_lending_club_data()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0

    def test_create_temporal_splits(self, sample_credit_data):
        """Test temporal splitting functionality."""
        loader = DataLoader()
        X, y = sample_credit_data

        splits = loader.create_temporal_splits(
            X, y,
            time_column='APPLICATION_MONTH',
            train_months=12,
            val_months=4,
            test_months=8
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']

        # Check that splits don't overlap and cover all data
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(X)

        # Check temporal ordering (train should have earlier months)
        # Note: temporal column is removed from splits
        assert 'APPLICATION_MONTH' not in X_train.columns
        assert 'APPLICATION_MONTH' not in X_val.columns
        assert 'APPLICATION_MONTH' not in X_test.columns

    def test_temporal_splits_invalid_column(self, sample_credit_data):
        """Test temporal splits with invalid time column."""
        loader = DataLoader()
        X, y = sample_credit_data

        with pytest.raises(ValueError, match="Time column 'invalid_col' not found"):
            loader.create_temporal_splits(X, y, time_column='invalid_col')

    def test_reproducibility(self):
        """Test that data generation is reproducible."""
        loader1 = DataLoader()
        loader2 = DataLoader()

        X1, y1 = loader1.load_home_credit_data(sample_size=500)
        X2, y2 = loader2.load_home_credit_data(sample_size=500)

        # Should be identical due to fixed random seed
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


class TestCreditDataPreprocessor:
    """Test cases for CreditDataPreprocessor class."""

    def test_init_default(self):
        """Test preprocessor initialization with default config."""
        preprocessor = CreditDataPreprocessor()
        assert not preprocessor.is_fitted
        assert len(preprocessor.numeric_features) == 0
        assert len(preprocessor.categorical_features) == 0

    def test_init_with_config(self):
        """Test preprocessor initialization with custom config."""
        config = {'scale_features': False, 'create_temporal_features': True}
        preprocessor = CreditDataPreprocessor(config)
        assert preprocessor.config == config

    def test_detect_feature_types(self, sample_credit_data):
        """Test automatic feature type detection."""
        preprocessor = CreditDataPreprocessor()
        X, _ = sample_credit_data

        preprocessor.detect_feature_types(X)

        assert len(preprocessor.numeric_features) > 0
        assert len(preprocessor.categorical_features) >= 0
        assert len(preprocessor.temporal_features) > 0

        # APPLICATION_MONTH should be detected as temporal
        assert 'APPLICATION_MONTH' in preprocessor.temporal_features

    def test_create_temporal_features(self, sample_credit_data):
        """Test temporal feature creation."""
        preprocessor = CreditDataPreprocessor()
        X, _ = sample_credit_data

        preprocessor.detect_feature_types(X)
        X_temporal = preprocessor.create_temporal_features(X)

        # Should have more columns than original
        assert X_temporal.shape[1] >= X.shape[1]

        # Should have quarter features
        quarter_cols = [col for col in X_temporal.columns if 'quarter' in col]
        assert len(quarter_cols) > 0

        # Should have trend features
        trend_cols = [col for col in X_temporal.columns if 'trend' in col]
        assert len(trend_cols) > 0

    def test_create_risk_features(self, sample_credit_data):
        """Test risk feature creation."""
        preprocessor = CreditDataPreprocessor()
        X, _ = sample_credit_data

        X_risk = preprocessor.create_risk_features(X)

        # Should have same or more columns
        assert X_risk.shape[1] >= X.shape[1]
        assert X_risk.shape[0] == X.shape[0]

    def test_handle_missing_values(self, preprocessor, sample_credit_data):
        """Test missing value handling."""
        X, y = sample_credit_data

        # Ensure we have missing values
        X_missing = X.copy()
        X_missing.iloc[0, 0] = np.nan
        X_missing.iloc[10, 5] = np.nan

        preprocessor.detect_feature_types(X_missing)
        X_imputed = preprocessor.handle_missing_values(X_missing, is_training=True)

        # Should have no missing values
        assert not X_imputed.isnull().any().any()
        assert X_imputed.shape == X_missing.shape

    def test_encode_categorical_features(self, preprocessor, sample_credit_data):
        """Test categorical feature encoding."""
        X, _ = sample_credit_data

        # Add some categorical features
        X_cat = X.copy()
        X_cat['category_1'] = np.random.choice(['A', 'B', 'C'], size=len(X))
        X_cat['category_2'] = np.random.choice(['X', 'Y'], size=len(X))

        preprocessor.detect_feature_types(X_cat)
        X_encoded = preprocessor.encode_categorical_features(X_cat, is_training=True)

        # Categorical columns should be numeric
        for col in preprocessor.categorical_features:
            if col in X_encoded.columns:
                assert pd.api.types.is_numeric_dtype(X_encoded[col])

    def test_scale_features(self, preprocessor, sample_credit_data):
        """Test feature scaling."""
        X, _ = sample_credit_data

        preprocessor.detect_feature_types(X)
        X_scaled = preprocessor.scale_features(X, is_training=True)

        # Should have same shape
        assert X_scaled.shape == X.shape

        # Numeric features should be scaled (approximately mean 0, std 1)
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            means = X_scaled[numeric_cols].mean()
            stds = X_scaled[numeric_cols].std()
            assert np.allclose(means, 0, atol=1e-10)
            assert np.allclose(stds, 1, atol=1e-10)

    def test_remove_redundant_features(self, preprocessor):
        """Test redundant feature removal."""
        # Create data with constant and duplicate features
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [1, 1, 1, 1, 1],  # Constant
            'feature_3': [1, 2, 3, 4, 5],  # Duplicate of feature_1
            'feature_4': [5, 4, 3, 2, 1]
        })

        original_shape = X.shape
        X_clean = preprocessor.remove_redundant_features(X, is_training=True)

        # Should have fewer features
        assert X_clean.shape[1] < original_shape[1]
        assert X_clean.shape[0] == original_shape[0]

    def test_fit_transform(self, preprocessor, sample_credit_data):
        """Test complete fit_transform pipeline."""
        X, y = sample_credit_data

        X_transformed = preprocessor.fit_transform(X, y)

        assert preprocessor.is_fitted
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]

        # Should have numeric data only (after encoding)
        assert X_transformed.select_dtypes(include=[np.number]).shape[1] == X_transformed.shape[1]

        # Should have no missing values
        assert not X_transformed.isnull().any().any()

    def test_transform_without_fit(self, preprocessor, sample_credit_data):
        """Test transform without fitting first."""
        X, _ = sample_credit_data

        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(X)

    def test_transform_consistency(self, fitted_preprocessor, sample_credit_data):
        """Test that transform produces consistent results."""
        X, _ = sample_credit_data

        # Transform twice
        X_transformed_1 = fitted_preprocessor.transform(X)
        X_transformed_2 = fitted_preprocessor.transform(X)

        # Should be identical
        pd.testing.assert_frame_equal(X_transformed_1, X_transformed_2)

    def test_get_feature_importance_weights(self, fitted_preprocessor, sample_credit_data):
        """Test feature importance weight calculation."""
        X, _ = sample_credit_data
        X_transformed = fitted_preprocessor.transform(X)

        weights = fitted_preprocessor.get_feature_importance_weights(X_transformed)

        assert isinstance(weights, dict)
        assert len(weights) == X_transformed.shape[1]

        # All weights should be positive
        assert all(w > 0 for w in weights.values())

        # Temporal features should have higher weights
        temporal_weights = [w for col, w in weights.items() if any(kw in col.lower() for kw in ['time', 'month', 'trend'])]
        regular_weights = [w for col, w in weights.items() if not any(kw in col.lower() for kw in ['time', 'month', 'trend', 'risk', 'ratio'])]

        if temporal_weights and regular_weights:
            assert max(temporal_weights) > max(regular_weights)

    def test_pipeline_with_missing_columns(self, fitted_preprocessor, sample_credit_data):
        """Test pipeline handles missing columns gracefully."""
        X, _ = sample_credit_data

        # Create data with missing column
        X_missing_col = X.drop(columns=[X.columns[0]])

        # Should not crash
        X_transformed = fitted_preprocessor.transform(X_missing_col)
        assert isinstance(X_transformed, pd.DataFrame)

    def test_pipeline_with_extra_columns(self, fitted_preprocessor, sample_credit_data):
        """Test pipeline handles extra columns gracefully."""
        X, _ = sample_credit_data

        # Add extra column
        X_extra = X.copy()
        X_extra['extra_column'] = np.random.random(len(X))

        # Should not crash
        X_transformed = fitted_preprocessor.transform(X_extra)
        assert isinstance(X_transformed, pd.DataFrame)

    def test_edge_case_single_sample(self, preprocessor):
        """Test preprocessor with single sample."""
        # Create single sample
        X = pd.DataFrame({
            'feature_1': [1.0],
            'feature_2': [2.0],
            'APPLICATION_MONTH': [5]
        })
        y = pd.Series([0])

        # Should not crash
        X_transformed = preprocessor.fit_transform(X, y)
        assert X_transformed.shape[0] == 1

    def test_edge_case_all_missing(self, preprocessor):
        """Test preprocessor with all missing values in a column."""
        X = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [np.nan, np.nan, np.nan],  # All missing
            'APPLICATION_MONTH': [1, 2, 3]
        })
        y = pd.Series([0, 1, 0])

        # Should not crash
        X_transformed = preprocessor.fit_transform(X, y)
        assert not X_transformed.isnull().any().any()

    def test_edge_case_constant_target(self, preprocessor):
        """Test preprocessor with constant target values."""
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4],
            'feature_2': [5, 6, 7, 8],
            'APPLICATION_MONTH': [1, 2, 3, 4]
        })
        y = pd.Series([1, 1, 1, 1])  # All same class

        # Should not crash
        X_transformed = preprocessor.fit_transform(X, y)
        assert X_transformed.shape[0] == 4