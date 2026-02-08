"""Data loading and preprocessing module for temporal credit degradation detection.

This module provides comprehensive data handling capabilities for credit risk modeling
with a focus on temporal analysis and concept drift detection. It includes:

- DataLoader: Handles data ingestion and temporal splitting
- CreditDataPreprocessor: Feature engineering and preprocessing pipeline

The module is designed to work with time-series credit data and supports
various preprocessing techniques including:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling and normalization
    - Temporal feature creation
    - Risk-based feature engineering

Example:
    Basic usage for loading and preprocessing credit data:

    >>> from temporal_credit_degradation_detector.data import DataLoader, CreditDataPreprocessor
    >>>
    >>> # Load data with temporal splits
    >>> loader = DataLoader()
    >>> data = loader.create_temporal_splits(df, time_column='date')
    >>>
    >>> # Preprocess features
    >>> preprocessor = CreditDataPreprocessor()
    >>> X_processed = preprocessor.fit_transform(data['train']['X'], data['train']['y'])
"""

from temporal_credit_degradation_detector.data.loader import DataLoader
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor

__all__ = ["DataLoader", "CreditDataPreprocessor"]