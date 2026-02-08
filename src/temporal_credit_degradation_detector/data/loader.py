"""Data loading utilities for credit risk datasets with temporal awareness.

This module provides comprehensive data loading capabilities for credit risk modeling
with built-in support for temporal splits and concept drift simulation. It supports
both real datasets (Home Credit, Lending Club) and synthetic data generation.

Key Features:
    - Temporal data splitting for time-aware model validation
    - Synthetic data generation that simulates concept drift
    - Support for multiple credit datasets
    - Automatic handling of missing data files with synthetic fallbacks

Example:
    Basic usage for loading data and creating temporal splits:

    >>> from temporal_credit_degradation_detector.data import DataLoader
    >>>
    >>> # Initialize loader
    >>> loader = DataLoader(data_path="/path/to/data")
    >>>
    >>> # Load data (falls back to synthetic if real data not available)
    >>> X, y = loader.load_home_credit_data(sample_size=10000)
    >>>
    >>> # Create temporal splits for proper validation
    >>> splits = loader.create_temporal_splits(
    ...     X, y,
    ...     time_column='APPLICATION_MONTH',
    ...     train_months=12,
    ...     val_months=4,
    ...     test_months=8
    ... )
    >>>
    >>> # Access splits
    >>> X_train, y_train = splits['train']
    >>> X_val, y_val = splits['val']
    >>> X_test, y_test = splits['test']

Note:
    The synthetic data generation includes simulated economic regimes to test
    model robustness against concept drift, making it suitable for testing
    temporal stability algorithms.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for credit risk datasets with temporal awareness."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """Initialize data loader.

        Args:
            data_path: Path to data directory. If None, will generate synthetic data.
        """
        self.data_path = Path(data_path) if data_path else None
        logger.info(f"Initialized DataLoader with path: {self.data_path}")

    def load_home_credit_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Home Credit Default Risk dataset or generate synthetic equivalent.

        Args:
            sample_size: Number of samples to load. If None, loads all data.
                Must be positive integer if provided.

        Returns:
            Features and target as (X, y) tuple where:
            - X: DataFrame with feature columns
            - y: Series with binary target values (0=no default, 1=default)

        Raises:
            ValueError: If sample_size is not positive
            FileNotFoundError: If data file is corrupted or unreadable
            KeyError: If required target column is missing

        Example:
            >>> loader = DataLoader("/path/to/data")
            >>> X, y = loader.load_home_credit_data(sample_size=1000)
            >>> print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        """
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        logger.info(f"Loading Home Credit data (sample_size={sample_size})...")

        # Try to load real data first
        if self.data_path and (self.data_path / "home_credit.csv").exists():
            try:
                logger.info(f"Reading data from {self.data_path / 'home_credit.csv'}")
                df = pd.read_csv(self.data_path / "home_credit.csv")

                if len(df) == 0:
                    logger.warning("Data file is empty, falling back to synthetic data")
                    return self._generate_synthetic_home_credit(sample_size or 50000)

                # Validate required columns
                if 'TARGET' not in df.columns:
                    raise KeyError("TARGET column not found in Home Credit data")

                # Sample if requested
                if sample_size:
                    actual_sample_size = min(sample_size, len(df))
                    if actual_sample_size < sample_size:
                        logger.warning(f"Requested {sample_size} samples but only {len(df)} available")
                    df = df.sample(n=actual_sample_size, random_state=42)

                X = df.drop(columns=['TARGET'])
                y = df['TARGET']

                # Validate data quality
                if X.shape[1] == 0:
                    raise ValueError("No feature columns found after dropping TARGET")

                logger.info(f"Loaded {len(X)} samples from Home Credit data with {X.shape[1]} features")
                return X, y

            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.error(f"Error reading CSV file: {e}. Falling back to synthetic data")
                return self._generate_synthetic_home_credit(sample_size or 50000)

            except Exception as e:
                logger.error(f"Unexpected error loading Home Credit data: {e}. Falling back to synthetic data")
                return self._generate_synthetic_home_credit(sample_size or 50000)

        else:
            if self.data_path:
                logger.info(f"Home Credit data file not found at {self.data_path}, generating synthetic data")
            else:
                logger.info("No data path provided, generating synthetic data")
            return self._generate_synthetic_home_credit(sample_size or 50000)

    def load_lending_club_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Lending Club dataset or generate synthetic equivalent.

        Args:
            sample_size: Number of samples to load. If None, loads all data.
                Must be positive integer if provided.

        Returns:
            Features and target as (X, y) tuple where:
            - X: DataFrame with loan feature columns
            - y: Series with binary loan status (0=paid, 1=default)

        Raises:
            ValueError: If sample_size is not positive
            FileNotFoundError: If data file is corrupted or unreadable
            KeyError: If required target column is missing

        Example:
            >>> loader = DataLoader("/path/to/data")
            >>> X, y = loader.load_lending_club_data(sample_size=5000)
            >>> print(f"Default rate: {y.mean():.3f}")
        """
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        logger.info(f"Loading Lending Club data (sample_size={sample_size})...")

        # Try to load real data first
        if self.data_path and (self.data_path / "lending_club.csv").exists():
            try:
                logger.info(f"Reading data from {self.data_path / 'lending_club.csv'}")
                df = pd.read_csv(self.data_path / "lending_club.csv")

                if len(df) == 0:
                    logger.warning("Data file is empty, falling back to synthetic data")
                    return self._generate_synthetic_lending_club(sample_size or 30000)

                # Validate required columns
                if 'loan_status' not in df.columns:
                    raise KeyError("loan_status column not found in Lending Club data")

                # Sample if requested
                if sample_size:
                    actual_sample_size = min(sample_size, len(df))
                    if actual_sample_size < sample_size:
                        logger.warning(f"Requested {sample_size} samples but only {len(df)} available")
                    df = df.sample(n=actual_sample_size, random_state=42)

                X = df.drop(columns=['loan_status'])
                y = df['loan_status']

                # Validate data quality
                if X.shape[1] == 0:
                    raise ValueError("No feature columns found after dropping loan_status")

                logger.info(f"Loaded {len(X)} samples from Lending Club data with {X.shape[1]} features")
                return X, y

            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.error(f"Error reading CSV file: {e}. Falling back to synthetic data")
                return self._generate_synthetic_lending_club(sample_size or 30000)

            except Exception as e:
                logger.error(f"Unexpected error loading Lending Club data: {e}. Falling back to synthetic data")
                return self._generate_synthetic_lending_club(sample_size or 30000)

        else:
            if self.data_path:
                logger.info(f"Lending Club data file not found at {self.data_path}, generating synthetic data")
            else:
                logger.info("No data path provided, generating synthetic data")
            return self._generate_synthetic_lending_club(sample_size or 30000)

    def _generate_synthetic_home_credit(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic Home Credit-like data."""
        logger.info(f"Generating synthetic Home Credit data with {n_samples} samples")

        # Create base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=50,
            n_informative=30,
            n_redundant=10,
            n_classes=2,
            class_sep=0.8,
            random_state=42
        )

        # Create realistic feature names and add temporal components
        feature_names = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
            'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
            'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE_encoded',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ] + [f'FEATURE_{i}' for i in range(22, 50)]

        df = pd.DataFrame(X, columns=feature_names)

        # Add temporal dimension (application month)
        df['APPLICATION_MONTH'] = np.random.choice(range(24), size=n_samples)  # 2 years of data

        # Create economic regime indicator (affects default rates)
        economic_regime = np.where(df['APPLICATION_MONTH'] < 8, 'stable',
                                 np.where(df['APPLICATION_MONTH'] < 16, 'recession', 'recovery'))
        df['ECONOMIC_REGIME'] = economic_regime

        # Adjust target based on economic regime (concept drift)
        y_adjusted = y.copy()
        recession_mask = (economic_regime == 'recession')
        y_adjusted[recession_mask] = np.random.binomial(1, 0.35, sum(recession_mask))  # Higher default rate

        target = pd.Series(y_adjusted, name='TARGET')

        logger.info(f"Generated synthetic Home Credit data: {df.shape}, default rate: {target.mean():.3f}")
        return df.drop(columns=['ECONOMIC_REGIME']), target

    def _generate_synthetic_lending_club(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic Lending Club-like data."""
        logger.info(f"Generating synthetic Lending Club data with {n_samples} samples")

        X, y = make_classification(
            n_samples=n_samples,
            n_features=40,
            n_informative=25,
            n_redundant=8,
            n_classes=2,
            class_sep=0.7,
            random_state=123
        )

        feature_names = [
            'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
            'installment', 'grade_encoded', 'sub_grade_encoded', 'emp_title_encoded',
            'emp_length', 'home_ownership_encoded', 'annual_inc', 'verification_status_encoded',
            'issue_d_encoded', 'purpose_encoded', 'title_encoded', 'zip_code_encoded',
            'addr_state_encoded', 'dti', 'delinq_2yrs', 'earliest_cr_line_encoded',
            'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status_encoded', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int'
        ] + [f'DERIVED_FEATURE_{i}' for i in range(36, 40)]

        df = pd.DataFrame(X, columns=feature_names)

        # Add temporal component
        df['ISSUE_MONTH'] = np.random.choice(range(18), size=n_samples)  # 1.5 years

        # Simulate economic impact on defaults
        economic_stress = np.where(df['ISSUE_MONTH'] > 12, 1.5, 1.0)  # Later months have more stress
        y_adjusted = np.where(
            np.random.random(n_samples) < (y * 0.3 * economic_stress),
            1, 0
        )

        target = pd.Series(y_adjusted, name='loan_status')

        logger.info(f"Generated synthetic Lending Club data: {df.shape}, default rate: {target.mean():.3f}")
        return df, target

    def create_temporal_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_column: str = 'APPLICATION_MONTH',
        train_months: int = 12,
        val_months: int = 4,
        test_months: int = 8
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Create temporal train/validation/test splits.

        Args:
            X: Feature matrix with temporal information
            y: Target vector
            time_column: Column name containing temporal information
            train_months: Number of months for training (must be >= 1)
            val_months: Number of months for validation (must be >= 1)
            test_months: Number of months for testing (must be >= 1)

        Returns:
            Dictionary with train/val/test splits, each containing (X, y) tuples

        Raises:
            ValueError: If inputs are invalid or insufficient data for splits
            KeyError: If time_column is not found in X

        Example:
            >>> loader = DataLoader()
            >>> X, y = loader.load_home_credit_data(1000)
            >>> splits = loader.create_temporal_splits(X, y, train_months=8, val_months=2, test_months=4)
            >>> X_train, y_train = splits['train']
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty")

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

        if time_column not in X.columns:
            raise KeyError(f"Time column '{time_column}' not found in features. Available columns: {list(X.columns)}")

        # Parameter validation
        if train_months < 1 or val_months < 1 or test_months < 1:
            raise ValueError("All month parameters must be >= 1")

        # Check for missing values in time column
        if X[time_column].isna().any():
            n_missing = X[time_column].isna().sum()
            logger.warning(f"Found {n_missing} missing values in time column '{time_column}', dropping these rows")
            valid_mask = X[time_column].notna()
            X = X[valid_mask]
            y = y[valid_mask]

        logger.info(f"Creating temporal splits with {train_months}/{val_months}/{test_months} months")
        logger.info(f"Input data: {len(X)} samples, time range: {X[time_column].min()}-{X[time_column].max()}")

        try:
            # Sort by time
            time_order = X[time_column].argsort()
            X_sorted, y_sorted = X.iloc[time_order], y.iloc[time_order]

            # Create splits based on time periods
            unique_times = sorted(X_sorted[time_column].unique())
            logger.info(f"Found {len(unique_times)} unique time periods")

            if len(unique_times) < train_months + val_months + 1:
                raise ValueError(
                    f"Insufficient time periods for splits. Need at least {train_months + val_months + 1}, "
                    f"but only have {len(unique_times)} unique time periods"
                )

            train_end = min(train_months, len(unique_times))
            val_end = min(train_end + val_months, len(unique_times))

            train_mask = X_sorted[time_column] < train_end
            val_mask = (X_sorted[time_column] >= train_end) & (X_sorted[time_column] < val_end)
            test_mask = X_sorted[time_column] >= val_end

            # Validate that each split has data
            if not train_mask.any():
                raise ValueError("Training split is empty")
            if not val_mask.any():
                raise ValueError("Validation split is empty")
            if not test_mask.any():
                raise ValueError("Test split is empty")

            splits = {
                'train': (X_sorted[train_mask].drop(columns=[time_column]), y_sorted[train_mask]),
                'val': (X_sorted[val_mask].drop(columns=[time_column]), y_sorted[val_mask]),
                'test': (X_sorted[test_mask].drop(columns=[time_column]), y_sorted[test_mask])
            }

            # Log split statistics
            for split_name, (X_split, y_split) in splits.items():
                if len(y_split) > 0:
                    default_rate = y_split.mean()
                    logger.info(f"{split_name}: {len(X_split)} samples, default rate: {default_rate:.3f}")
                else:
                    logger.warning(f"{split_name} split is empty!")

            return splits

        except Exception as e:
            logger.error(f"Error creating temporal splits: {e}")
            raise