"""Test configuration and fixtures for temporal credit degradation detector tests."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.datasets import make_classification
import tempfile
import shutil
from pathlib import Path

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor
from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble
from temporal_credit_degradation_detector.utils.config import Config


@pytest.fixture
def sample_credit_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample credit data for testing."""
    # Generate base classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        class_sep=0.8,
        random_state=42
    )

    # Create realistic feature names
    feature_names = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'REGION_RATING_CLIENT', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
        'EXT_SOURCE_3', 'DAYS_REGISTRATION', 'HOUR_APPR_PROCESS_START',
        'REG_REGION_NOT_LIVE_REGION', 'ORGANIZATION_TYPE_encoded',
        'FLAG_DOCUMENT_3', 'FLAG_EMAIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE',
        'CREDIT_UTILIZATION'
    ]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Add temporal dimension
    df['APPLICATION_MONTH'] = np.random.choice(range(24), size=1000)

    # Add some missing values
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)

    target = pd.Series(y, name='TARGET')

    return df, target


@pytest.fixture
def sample_temporal_data() -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Create temporal split data for testing."""
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)
    df['APPLICATION_MONTH'] = np.repeat(range(20), 100)  # 20 months, 100 samples each

    target = pd.Series(y, name='TARGET')

    # Split into train/val/test based on time
    train_mask = df['APPLICATION_MONTH'] < 12
    val_mask = (df['APPLICATION_MONTH'] >= 12) & (df['APPLICATION_MONTH'] < 16)
    test_mask = df['APPLICATION_MONTH'] >= 16

    return {
        'train': (df[train_mask].drop(columns=['APPLICATION_MONTH']), target[train_mask]),
        'val': (df[val_mask].drop(columns=['APPLICATION_MONTH']), target[val_mask]),
        'test': (df[test_mask].drop(columns=['APPLICATION_MONTH']), target[test_mask])
    }


@pytest.fixture
def sample_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample predictions for testing."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 500)
    # Create realistic predictions (correlated with true labels)
    y_prob = np.random.beta(2, 5, 500)  # Skewed toward low values
    y_prob[y_true == 1] += 0.3  # Higher probabilities for actual positives
    y_prob = np.clip(y_prob, 0.01, 0.99)

    return y_true, y_prob


@pytest.fixture
def sample_drift_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create data showing concept drift for testing."""
    np.random.seed(42)

    # Reference data
    X_ref, y_ref_true = make_classification(
        n_samples=1000, n_features=10, n_informative=7, random_state=42
    )
    X_ref = pd.DataFrame(X_ref, columns=[f'feature_{i}' for i in range(10)])
    y_ref_prob = np.random.beta(2, 5, 1000)

    # Current data with drift
    X_curr, y_curr_true = make_classification(
        n_samples=1000, n_features=10, n_informative=7, random_state=123
    )
    X_curr = pd.DataFrame(X_curr, columns=[f'feature_{i}' for i in range(10)])

    # Add feature drift by shifting distributions
    for i in range(5):
        X_curr.iloc[:, i] += 0.5

    # Add prediction drift
    y_curr_prob = np.random.beta(3, 4, 1000)  # Different distribution

    return X_ref, X_curr, y_ref_prob, y_curr_prob


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config()


@pytest.fixture
def preprocessor(config: Config) -> CreditDataPreprocessor:
    """Create configured preprocessor."""
    return CreditDataPreprocessor(config.preprocessing.__dict__)


@pytest.fixture
def fitted_preprocessor(preprocessor: CreditDataPreprocessor, sample_credit_data) -> CreditDataPreprocessor:
    """Create fitted preprocessor."""
    X, y = sample_credit_data
    preprocessor.fit(X, y)
    return preprocessor


@pytest.fixture
def stability_ensemble(config: Config) -> StabilityWeightedEnsemble:
    """Create stability-weighted ensemble model."""
    return StabilityWeightedEnsemble(
        stability_alpha=config.model.stability_alpha,
        min_weight=config.model.min_weight,
        recalibration_threshold=config.model.recalibration_threshold,
        calibration_window=config.model.calibration_window,
        random_state=config.random_state
    )


@pytest.fixture
def fitted_ensemble(stability_ensemble: StabilityWeightedEnsemble, sample_temporal_data) -> StabilityWeightedEnsemble:
    """Create fitted stability-weighted ensemble."""
    X_train, y_train = sample_temporal_data['train']
    stability_ensemble.fit(X_train, y_train)
    return stability_ensemble


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def data_loader() -> DataLoader:
    """Create data loader for testing."""
    return DataLoader()


@pytest.fixture
def mock_data_files(temp_dir) -> Path:
    """Create mock data files for testing."""
    data_path = Path(temp_dir) / "data"
    data_path.mkdir()

    # Create mock Home Credit data
    home_credit_data = pd.DataFrame({
        'AMT_INCOME_TOTAL': np.random.normal(100000, 30000, 1000),
        'AMT_CREDIT': np.random.normal(500000, 150000, 1000),
        'DAYS_BIRTH': np.random.randint(-25000, -7000, 1000),
        'TARGET': np.random.binomial(1, 0.08, 1000)
    })
    home_credit_data.to_csv(data_path / "home_credit.csv", index=False)

    # Create mock Lending Club data
    lending_club_data = pd.DataFrame({
        'loan_amnt': np.random.normal(15000, 5000, 800),
        'annual_inc': np.random.normal(65000, 20000, 800),
        'dti': np.random.normal(15, 5, 800),
        'loan_status': np.random.binomial(1, 0.12, 800)
    })
    lending_club_data.to_csv(data_path / "lending_club.csv", index=False)

    return data_path


# Performance benchmarks for regression testing
PERFORMANCE_BENCHMARKS = {
    'min_auc_roc': 0.75,
    'max_brier_score': 0.20,
    'max_calibration_error': 0.10,
    'min_precision': 0.60,
    'min_recall': 0.50
}


@pytest.fixture
def performance_benchmarks() -> Dict[str, float]:
    """Performance benchmarks for regression testing."""
    return PERFORMANCE_BENCHMARKS


# Helper functions for tests

def assert_valid_probabilities(y_prob: np.ndarray) -> None:
    """Assert that predictions are valid probabilities."""
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "Probabilities must be between 0 and 1"
    assert not np.any(np.isnan(y_prob)), "Probabilities cannot contain NaN values"
    assert not np.any(np.isinf(y_prob)), "Probabilities cannot contain infinite values"


def assert_valid_predictions(y_pred: np.ndarray) -> None:
    """Assert that predictions are valid class labels."""
    unique_values = np.unique(y_pred)
    assert len(unique_values) <= 2, "Binary classifier should have at most 2 unique predictions"
    assert np.all(np.isin(unique_values, [0, 1])), "Predictions must be 0 or 1"


def assert_reproducible_results(func, *args, n_runs: int = 3, **kwargs) -> None:
    """Assert that function produces reproducible results."""
    results = []
    for _ in range(n_runs):
        result = func(*args, **kwargs)
        results.append(result)

    # Check if all results are the same (for deterministic functions)
    if isinstance(results[0], np.ndarray):
        for i in range(1, n_runs):
            np.testing.assert_array_equal(results[0], results[i],
                                        err_msg="Function should produce reproducible results")
    else:
        for i in range(1, n_runs):
            assert results[0] == results[i], "Function should produce reproducible results"


@pytest.fixture
def assert_helpers():
    """Provide assertion helper functions."""
    return {
        'assert_valid_probabilities': assert_valid_probabilities,
        'assert_valid_predictions': assert_valid_predictions,
        'assert_reproducible_results': assert_reproducible_results
    }