"""Data preprocessing for credit risk modeling with temporal awareness.

This module provides comprehensive preprocessing capabilities for credit data, with
special attention to temporal features and concept drift handling. The preprocessing
pipeline is highly configurable and includes:

- Intelligent feature type detection
- Missing value imputation with configurable strategies
- Categorical encoding (ordinal and one-hot)
- Temporal and risk-based feature engineering
- Feature scaling and normalization
- Redundant feature removal
- Discretization of continuous variables

The preprocessor is designed to handle real-world credit data challenges including:
- High missing value rates
- Mixed data types (numerical, categorical, temporal)
- Class imbalance
- Feature drift over time

Example:
    Basic usage for preprocessing credit data:

    >>> from temporal_credit_degradation_detector.utils.config import Config
    >>> from temporal_credit_degradation_detector.data import CreditDataPreprocessor
    >>>
    >>> # Initialize with configuration
    >>> config = Config()
    >>> preprocessor = CreditDataPreprocessor(config.preprocessing)
    >>>
    >>> # Fit and transform training data
    >>> X_train_processed = preprocessor.fit_transform(X_train, y_train)
    >>>
    >>> # Transform test data using fitted preprocessor
    >>> X_test_processed = preprocessor.transform(X_test)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import OrdinalEncoder, OneHotEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.creation import MathFeatures
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import logging utilities with fallback
try:
    from temporal_credit_degradation_detector.utils.logging_utils import log_performance, StructuredLogger
    struct_logger = StructuredLogger(__name__)
except ImportError:
    # Fallback if logging utils not available
    def log_performance(**kwargs):
        def decorator(func):
            return func
        return decorator
    struct_logger = None


class CreditDataPreprocessor:
    """Comprehensive preprocessor for credit data with temporal drift handling.

    This class provides a complete preprocessing pipeline for credit risk data
    with support for temporal features, missing value handling, and feature engineering.

    Attributes:
        config: Preprocessing configuration object
        numeric_features: List of detected numeric feature names
        categorical_features: List of detected categorical feature names
        temporal_features: List of detected temporal feature names
        is_fitted: Whether the preprocessor has been fitted to data

    Example:
        >>> from temporal_credit_degradation_detector.utils.config import Config, PreprocessingConfig
        >>> config = PreprocessingConfig(discretization_bins=10, missing_value_fill=-999)
        >>> preprocessor = CreditDataPreprocessor(config)
        >>> X_processed = preprocessor.fit_transform(X_train, y_train)
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize preprocessor with configuration.

        Args:
            config: PreprocessingConfig object or dictionary with preprocessing parameters.
                If None, default configuration will be used.

        Raises:
            TypeError: If config is not a valid configuration object or dictionary
        """
        try:
            # Handle different config input types
            if config is None:
                from temporal_credit_degradation_detector.utils.config import PreprocessingConfig
                self.config = PreprocessingConfig()
                logger.info("Using default preprocessing configuration")
            elif hasattr(config, '__dict__'):  # Configuration object
                self.config = config
                logger.info("Using provided preprocessing configuration")
            elif isinstance(config, dict):  # Dictionary configuration (legacy support)
                from temporal_credit_degradation_detector.utils.config import PreprocessingConfig
                self.config = PreprocessingConfig(**config)
                logger.info("Converted dictionary to PreprocessingConfig")
            else:
                raise TypeError(f"Invalid config type: {type(config)}. Expected PreprocessingConfig, dict, or None")

            self.numeric_features: List[str] = []
            self.categorical_features: List[str] = []
            self.temporal_features: List[str] = []
            self.is_fitted: bool = False

            # Initialize preprocessing components
            self._init_preprocessors()
            logger.info(f"Initialized CreditDataPreprocessor with config: {type(self.config).__name__}")

        except Exception as e:
            logger.error(f"Error initializing CreditDataPreprocessor: {e}")
            raise

    def _init_preprocessors(self) -> None:
        """Initialize all preprocessing components using configuration values.

        Raises:
            ValueError: If configuration contains invalid values
        """
        try:
            # Imputers with configurable values
            self.numeric_imputer = ArbitraryNumberImputer(
                arbitrary_number=self.config.missing_value_fill,
                variables=None  # Will be set during fit
            )
            self.categorical_imputer = CategoricalImputer(
                imputation_method='missing',
                variables=None  # Will be set during fit
            )

            # Encoders
            self.ordinal_encoder = OrdinalEncoder(
                encoding_method='arbitrary',
                variables=None  # Will be set during fit
            )

            # Feature engineering (will be initialized in fit method)
            self.math_features = None

            # Discretizer for continuous features with configurable bins
            if self.config.discretization_bins < 2:
                raise ValueError(f"discretization_bins must be >= 2, got {self.config.discretization_bins}")

            self.discretizer = EqualFrequencyDiscretiser(
                variables=None,
                q=self.config.discretization_bins,
                return_object=False,
                return_boundaries=True
            )

            # Feature selection with configurable thresholds
            if not 0 <= self.config.constant_feature_threshold <= 1:
                raise ValueError(f"constant_feature_threshold must be between 0 and 1, got {self.config.constant_feature_threshold}")

            self.constant_dropper = DropConstantFeatures(tol=self.config.constant_feature_threshold)
            self.duplicate_dropper = DropDuplicateFeatures()

            # Scaler
            self.scaler = StandardScaler()

            logger.debug(f"Initialized preprocessors with missing_value_fill={self.config.missing_value_fill}, "
                        f"discretization_bins={self.config.discretization_bins}, "
                        f"constant_feature_threshold={self.config.constant_feature_threshold}")

        except Exception as e:
            logger.error(f"Error initializing preprocessors: {e}")
            raise ValueError(f"Invalid preprocessing configuration: {e}") from e

    def detect_feature_types(self, df: pd.DataFrame) -> None:
        """Automatically detect feature types.

        Args:
            df: Input dataframe
        """
        self.numeric_features = []
        self.categorical_features = []
        self.temporal_features = []

        for col in df.columns:
            if col.lower() in ['target', 'loan_status', 'default']:
                continue

            # Check for temporal features
            if any(keyword in col.lower() for keyword in ['month', 'date', 'time', 'day']):
                self.temporal_features.append(col)
            # Check for numeric features
            elif df[col].dtype in ['int64', 'float64']:
                # Skip if looks like encoded categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.05 or df[col].nunique() > 50:
                    self.numeric_features.append(col)
                else:
                    self.categorical_features.append(col)
            else:
                self.categorical_features.append(col)

        logger.info(f"Detected {len(self.numeric_features)} numeric, "
                   f"{len(self.categorical_features)} categorical, "
                   f"{len(self.temporal_features)} temporal features")

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal-aware features for drift detection using configurable parameters.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with additional temporal features

        Example:
            >>> preprocessor = CreditDataPreprocessor()
            >>> df_with_temporal = preprocessor.create_temporal_features(df)
        """
        try:
            df_temp = df.copy()
            features_created = 0

            # Create temporal bins and seasonal features
            if self.temporal_features and self.config.seasonal_features:
                for temp_col in self.temporal_features:
                    if temp_col in df_temp.columns and df_temp[temp_col].notna().any():
                        # Create quarter/season features
                        df_temp[f'{temp_col}_quarter'] = (df_temp[temp_col] // 3) + 1

                        # Create trend features (normalized temporal progression)
                        max_time = df_temp[temp_col].max()
                        if max_time > 0:
                            df_temp[f'{temp_col}_trend'] = df_temp[temp_col] / max_time
                        else:
                            df_temp[f'{temp_col}_trend'] = 0

                        # Create temporal bins based on lookback window
                        if self.config.temporal_lookback_months > 0:
                            df_temp[f'{temp_col}_period'] = df_temp[temp_col] // self.config.temporal_lookback_months

                        features_created += 3
                        logger.debug(f"Created temporal features for {temp_col}")

            # Create interaction features with time (configurable limit)
            if self.temporal_features and self.numeric_features:
                time_col = self.temporal_features[0] if self.temporal_features else None
                if time_col and time_col in df_temp.columns:
                    # Limit interactions to prevent feature explosion
                    max_interactions = min(len(self.numeric_features), self.config.max_temporal_interactions)

                    for num_col in self.numeric_features[:max_interactions]:
                        if num_col in df_temp.columns:
                            # Time-weighted feature
                            df_temp[f'{num_col}_time_weighted'] = (
                                df_temp[num_col] * df_temp[time_col]
                            )

                            # Relative change over time (if meaningful)
                            if df_temp[time_col].nunique() > 1:
                                df_temp[f'{num_col}_time_ratio'] = (
                                    df_temp[num_col] / (df_temp[time_col] + self.config.epsilon_division_safety)
                                )

                            features_created += 2

            logger.info(f"Created {features_created} temporal features. New shape: {df_temp.shape}")
            return df_temp

        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            logger.warning("Returning original dataframe due to temporal feature creation failure")
            return df.copy()

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific risk features using configurable thresholds.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with risk features

        Raises:
            ValueError: If required columns are missing for risk feature creation
        """
        try:
            df_risk = df.copy()
            features_created = 0

            # Credit utilization features with configurable threshold
            util_cols = [col for col in df_risk.columns if 'revol_util' in col.lower() or 'utilization' in col.lower()]
            if util_cols:
                util_col = util_cols[0]
                df_risk['credit_utilization_risk'] = np.where(
                    df_risk[util_col] > self.config.credit_util_high_threshold, 1, 0
                )
                # Create binned utilization features
                df_risk['credit_util_binned'] = pd.cut(
                    df_risk[util_col],
                    bins=self.config.credit_util_bins,
                    labels=False,
                    duplicates='drop'
                )
                features_created += 2
                logger.debug(f"Created credit utilization features with threshold {self.config.credit_util_high_threshold}")

            # Income-to-debt ratio features with configurable threshold
            income_cols = [col for col in df_risk.columns if 'income' in col.lower() or 'amt_income' in col.lower()]
            debt_cols = [col for col in df_risk.columns if 'credit' in col.lower() or 'loan' in col.lower() or 'amt_credit' in col.lower()]

            if income_cols and debt_cols:
                income_col = income_cols[0]
                debt_col = debt_cols[0]

                # Debt-to-income ratio
                df_risk['debt_to_income_ratio'] = (
                    df_risk[debt_col] / (df_risk[income_col] + self.config.epsilon_division_safety if hasattr(self.config, 'epsilon_division_safety') else 1e-8)
                )

                # Income sufficiency indicator with configurable threshold
                df_risk['income_credit_ratio_risk'] = np.where(
                    df_risk['debt_to_income_ratio'] > self.config.income_credit_ratio_threshold, 1, 0
                )
                features_created += 2
                logger.debug(f"Created income-debt ratio features")

            # Age-related features with configurable bins
            age_cols = [col for col in df_risk.columns if 'birth' in col.lower() or 'age' in col.lower()]
            if age_cols:
                age_col = age_cols[0]
                # Convert days to years if needed
                if 'days' in age_col.lower():
                    df_risk['age_years'] = abs(df_risk[age_col]) / 365.25
                else:
                    df_risk['age_years'] = df_risk[age_col]

                # Use configurable age bins
                try:
                    df_risk['age_risk_category'] = pd.cut(
                        df_risk['age_years'],
                        bins=self.config.age_bins,
                        labels=[f'age_group_{i}' for i in range(len(self.config.age_bins)-1)],
                        include_lowest=True
                    )
                    features_created += 2
                    logger.debug(f"Created age features with bins: {self.config.age_bins}")
                except Exception as e:
                    logger.warning(f"Could not create age categories with bins {self.config.age_bins}: {e}")

            # Payment behavior features if available
            payment_cols = [col for col in df_risk.columns if 'payment' in col.lower() or 'installment' in col.lower()]
            if payment_cols and income_cols:
                payment_col = payment_cols[0]
                income_col = income_cols[0]

                df_risk['payment_to_income_ratio'] = (
                    df_risk[payment_col] / (df_risk[income_col] + self.config.epsilon_division_safety if hasattr(self.config, 'epsilon_division_safety') else 1e-8)
                )
                df_risk['payment_burden_risk'] = np.where(
                    df_risk['payment_to_income_ratio'] > self.config.payment_ratio_threshold, 1, 0
                )
                features_created += 2

            logger.info(f"Created {features_created} risk features. New shape: {df_risk.shape}")
            return df_risk

        except Exception as e:
            logger.error(f"Error creating risk features: {e}")
            # Return original dataframe if feature creation fails
            logger.warning("Returning original dataframe due to risk feature creation failure")
            return df.copy()

    def handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input dataframe
            is_training: Whether this is training data (fit transformers)

        Returns:
            Dataframe with imputed values
        """
        df_imputed = df.copy()

        # Update imputer variables
        numeric_vars = [col for col in self.numeric_features if col in df_imputed.columns]
        categorical_vars = [col for col in self.categorical_features if col in df_imputed.columns]

        if numeric_vars:
            self.numeric_imputer.variables = numeric_vars
            if is_training:
                df_imputed = self.numeric_imputer.fit_transform(df_imputed)
            else:
                df_imputed = self.numeric_imputer.transform(df_imputed)

        if categorical_vars:
            # Cast detected categorical features to object dtype so feature_engine
            # CategoricalImputer accepts them (it rejects numeric dtypes)
            for col in categorical_vars:
                if df_imputed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    df_imputed[col] = df_imputed[col].astype(object)
            self.categorical_imputer.variables = categorical_vars
            if is_training:
                df_imputed = self.categorical_imputer.fit_transform(df_imputed)
            else:
                df_imputed = self.categorical_imputer.transform(df_imputed)

        logger.info(f"Handled missing values for {len(numeric_vars)} numeric and {len(categorical_vars)} categorical features")
        return df_imputed

    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            df: Input dataframe
            is_training: Whether this is training data

        Returns:
            Dataframe with encoded categorical features
        """
        df_encoded = df.copy()

        categorical_vars = [col for col in self.categorical_features if col in df_encoded.columns]
        if not categorical_vars:
            return df_encoded

        # Ensure categorical features are object dtype for feature_engine OrdinalEncoder
        for col in categorical_vars:
            if df_encoded[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                df_encoded[col] = df_encoded[col].astype(object)
        self.ordinal_encoder.variables = categorical_vars
        if is_training:
            df_encoded = self.ordinal_encoder.fit_transform(df_encoded)
        else:
            df_encoded = self.ordinal_encoder.transform(df_encoded)

        logger.info(f"Encoded {len(categorical_vars)} categorical features")
        return df_encoded

    def scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input dataframe
            is_training: Whether this is training data

        Returns:
            Dataframe with scaled features
        """
        df_scaled = df.copy()

        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df_scaled

        if is_training:
            df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        else:
            df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])

        logger.info(f"Scaled {len(numeric_cols)} numerical features")
        return df_scaled

    def remove_redundant_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Remove constant and duplicate features.

        Args:
            df: Input dataframe
            is_training: Whether this is training data

        Returns:
            Dataframe with redundant features removed
        """
        df_clean = df.copy()
        original_shape = df_clean.shape

        # Remove constant features
        if is_training:
            df_clean = self.constant_dropper.fit_transform(df_clean)
        else:
            df_clean = self.constant_dropper.transform(df_clean)

        # Remove duplicate features
        if is_training:
            df_clean = self.duplicate_dropper.fit_transform(df_clean)
        else:
            df_clean = self.duplicate_dropper.transform(df_clean)

        logger.info(f"Removed redundant features: {original_shape} -> {df_clean.shape}")
        return df_clean

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CreditDataPreprocessor':
        """Fit the preprocessor on training data.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on data with shape {X.shape}")

        # Detect feature types
        self.detect_feature_types(X)

        # Initialize math features with detected numeric columns
        if self.numeric_features:
            self.math_features = MathFeatures(
                variables=self.numeric_features[:3],  # Use first 3 numeric features to avoid explosion
                func=['sum', 'prod', 'mean', 'std'],
                new_variables_names=None
            )

        # Create temporal features for fitting
        X_with_temporal = self.create_temporal_features(X)

        # Fit all transformers
        # Update imputers with correct variables
        if self.numeric_features:
            self.numeric_imputer.variables = self.numeric_features
            self.numeric_imputer.fit(X_with_temporal)

        if self.categorical_features:
            self.categorical_imputer.variables = self.categorical_features
            self.categorical_imputer.fit(X_with_temporal)

        # Fit feature engineering transformers
        if self.math_features is not None:
            self.math_features.fit(X_with_temporal)

        if self.numeric_features:
            self.discretizer.variables = self.numeric_features
            self.discretizer.fit(X_with_temporal)

        # Fit encoder and scaler on numeric features
        if self.categorical_features:
            self.ordinal_encoder.variables = self.categorical_features
            self.ordinal_encoder.fit(X_with_temporal)

        if self.numeric_features:
            self.scaler.fit(X_with_temporal[self.numeric_features])

        # Fit feature selection components
        self.constant_dropper.fit(X_with_temporal)
        self.duplicate_dropper.fit(X_with_temporal)

        # Store original column names
        self.original_columns = X.columns.tolist()

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self

    @log_performance(log_memory=True, memory_threshold_mb=200.0)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.info(f"Transforming data with shape {X.shape}")
        df_transformed = X.copy()

        # Create temporal features
        df_transformed = self.create_temporal_features(df_transformed)

        # Create risk features
        df_transformed = self.create_risk_features(df_transformed)

        # Align columns to match what was seen during fit_transform
        if hasattr(self, '_fitted_columns_after_feature_eng'):
            fitted_cols = self._fitted_columns_after_feature_eng
            # Add missing columns with 0
            for col in fitted_cols:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0
            # Keep only fitted columns in the same order
            df_transformed = df_transformed[fitted_cols]

        # Update feature types after feature creation
        self.detect_feature_types(df_transformed)

        # Handle missing values
        df_transformed = self.handle_missing_values(df_transformed, is_training=False)

        # Encode categorical features
        df_transformed = self.encode_categorical_features(df_transformed, is_training=False)

        # Remove redundant features
        df_transformed = self.remove_redundant_features(df_transformed, is_training=False)

        # Scale features
        df_transformed = self.scale_features(df_transformed, is_training=False)

        logger.info(f"Transformation complete. Final shape: {df_transformed.shape}")
        return df_transformed

    @log_performance(log_memory=True, memory_threshold_mb=300.0)
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data in one step.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Transformed feature matrix
        """
        logger.info(f"Fitting and transforming data with shape {X.shape}")
        df_transformed = X.copy()

        # Detect feature types first
        self.detect_feature_types(df_transformed)

        # Create temporal features
        df_transformed = self.create_temporal_features(df_transformed)

        # Create risk features
        df_transformed = self.create_risk_features(df_transformed)

        # Update feature types after feature creation
        self.detect_feature_types(df_transformed)

        # Save columns after feature engineering for alignment during transform()
        self._fitted_columns_after_feature_eng = df_transformed.columns.tolist()

        # Handle missing values
        df_transformed = self.handle_missing_values(df_transformed, is_training=True)

        # Encode categorical features
        df_transformed = self.encode_categorical_features(df_transformed, is_training=True)

        # Remove redundant features
        df_transformed = self.remove_redundant_features(df_transformed, is_training=True)

        # Scale features
        df_transformed = self.scale_features(df_transformed, is_training=True)

        self.is_fitted = True
        logger.info(f"Fit-transform complete. Final shape: {df_transformed.shape}")
        return df_transformed

    def get_feature_importance_weights(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance weights for stability scoring.

        Args:
            X: Transformed feature matrix

        Returns:
            Dictionary mapping feature names to importance weights
        """
        weights = {}

        # Higher weight for temporal features (important for drift detection)
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['time', 'month', 'trend', 'quarter']):
                weights[col] = 1.5
            elif any(keyword in col.lower() for keyword in ['risk', 'ratio', 'utilization']):
                weights[col] = 1.3
            else:
                weights[col] = 1.0

        logger.info(f"Generated feature importance weights for {len(weights)} features")
        return weights