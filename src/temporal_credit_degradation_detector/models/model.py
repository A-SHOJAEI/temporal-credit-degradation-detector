"""Stability-weighted ensemble model for temporal credit risk prediction."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as cb
from scipy.special import softmax
import joblib
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


class CalibrationMonitor:
    """Monitors model calibration quality over time."""

    def __init__(self, window_size: int = 1000, min_buffer_size: int = 100):
        """Initialize calibration monitor.

        Args:
            window_size: Size of rolling window for calibration assessment.
            min_buffer_size: Minimum buffer size for reliable calibration estimates.
        """
        self.window_size = window_size
        self.min_buffer_size = min_buffer_size
        self.calibration_history: List[float] = []
        self.predictions_buffer: List[Tuple[float, int]] = []  # (prob, actual)

    def update(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """Update calibration monitor with new predictions.

        Args:
            y_prob: Predicted probabilities
            y_true: True labels

        Returns:
            Current Brier score (calibration metric)
        """
        # Add new predictions to buffer
        for prob, true in zip(y_prob, y_true):
            self.predictions_buffer.append((prob, true))

        # Maintain window size
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer = self.predictions_buffer[-self.window_size:]

        # Calculate current calibration (use configurable minimum buffer size)
        if len(self.predictions_buffer) >= self.min_buffer_size:
            probs, trues = zip(*self.predictions_buffer)
            brier_score = brier_score_loss(trues, probs)
            self.calibration_history.append(brier_score)
            return brier_score
        else:
            return 0.0

    def get_calibration_trend(self, recent_window: int = 10) -> float:
        """Get recent calibration trend.

        Args:
            recent_window: Number of recent measurements to consider

        Returns:
            Calibration trend (negative means deteriorating)
        """
        if len(self.calibration_history) < recent_window:
            return 0.0

        recent_scores = self.calibration_history[-recent_window:]
        # Calculate slope (negative slope means deteriorating calibration)
        x = np.arange(len(recent_scores))
        trend = np.polyfit(x, recent_scores, 1)[0]
        return -trend  # Negative because lower Brier score is better


class StabilityWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Novel stability-weighted ensemble for graceful model degradation.

    This ensemble automatically reweights base models based on their recent
    calibration quality, enabling graceful degradation during economic shifts
    rather than catastrophic failure.
    """

    def __init__(
        self,
        base_models: Optional[List[Any]] = None,
        calibration_window: Optional[int] = None,
        stability_alpha: Optional[float] = None,
        min_weight: Optional[float] = None,
        recalibration_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
        config: Optional[Any] = None
    ):
        """Initialize stability-weighted ensemble.

        Args:
            base_models: List of base models. If None, uses default models based on config.
            calibration_window: Size of rolling window for calibration monitoring.
                If None, uses config value.
            stability_alpha: Learning rate for weight updates. If None, uses config value.
            min_weight: Minimum weight for any model. If None, uses config value.
            recalibration_threshold: Brier score threshold for triggering recalibration.
                If None, uses config value.
            random_state: Random seed for reproducibility. If None, uses config value.
            config: ModelConfig object. If None, default configuration will be used.

        Example:
            >>> from temporal_credit_degradation_detector.utils.config import ModelConfig
            >>> config = ModelConfig(stability_alpha=0.05, calibration_window=500)
            >>> ensemble = StabilityWeightedEnsemble(config=config)
        """
        # Handle configuration
        if config is None:
            try:
                from temporal_credit_degradation_detector.utils.config import ModelConfig
                self.config = ModelConfig()
                logger.info("Using default ModelConfig")
            except ImportError:
                # Fallback to hardcoded defaults if config not available
                self.config = None
                logger.warning("ModelConfig not available, using hardcoded defaults")
        else:
            self.config = config

        # Set parameters with fallbacks to config or defaults
        if self.config:
            self.calibration_window = calibration_window if calibration_window is not None else self.config.calibration_window
            self.stability_alpha = stability_alpha if stability_alpha is not None else self.config.stability_alpha
            self.min_weight = min_weight if min_weight is not None else self.config.min_weight
            self.recalibration_threshold = recalibration_threshold if recalibration_threshold is not None else self.config.recalibration_threshold
            self.random_state = random_state if random_state is not None else self.config.random_state
        else:
            # Hardcoded fallbacks
            self.calibration_window = calibration_window if calibration_window is not None else 1000
            self.stability_alpha = stability_alpha if stability_alpha is not None else 0.1
            self.min_weight = min_weight if min_weight is not None else 0.05
            self.recalibration_threshold = recalibration_threshold if recalibration_threshold is not None else 0.15
            self.random_state = random_state if random_state is not None else 42

        self.base_models = base_models

        # Initialize components
        self.models_: List[Any] = []
        self.calibrators_: List[CalibratedClassifierCV] = []
        self.monitors_: List[CalibrationMonitor] = []
        self.weights_: np.ndarray = np.array([])
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

        # Validate configuration
        self._validate_config()

        logger.info(f"Initialized StabilityWeightedEnsemble with alpha={self.stability_alpha}, "
                   f"window={self.calibration_window}, min_weight={self.min_weight}")

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.calibration_window <= 0:
            raise ValueError(f"calibration_window must be positive, got {self.calibration_window}")

        if not 0 < self.stability_alpha <= 1:
            raise ValueError(f"stability_alpha must be in (0,1], got {self.stability_alpha}")

        if not 0 <= self.min_weight < 1:
            raise ValueError(f"min_weight must be in [0,1), got {self.min_weight}")

        if not 0 < self.recalibration_threshold < 1:
            raise ValueError(f"recalibration_threshold must be in (0,1), got {self.recalibration_threshold}")

        logger.debug("Configuration validation passed")

    def _get_default_models(self) -> List[Any]:
        """Get default base models using configuration parameters.

        Returns:
            List of configured base models

        Raises:
            ImportError: If required model libraries are not available
        """
        if self.base_models is not None:
            return self.base_models

        try:
            # Get configuration parameters or use hardcoded defaults
            if self.config:
                n_estimators = self.config.n_estimators
                max_depth = self.config.max_depth
                learning_rate = self.config.learning_rate
                min_samples_split = self.config.min_samples_split
                min_samples_leaf = self.config.min_samples_leaf
            else:
                # Fallback to hardcoded values
                n_estimators = 100
                max_depth = 6
                learning_rate = 0.1
                min_samples_split = 20
                min_samples_leaf = 10

            models = []

            # LightGBM models with different configurations
            models.extend([
                lgb.LGBMClassifier(
                    n_estimators=n_estimators * 2,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_samples=min_samples_leaf,
                    random_state=self.random_state,
                    verbose=-1
                ),
                lgb.LGBMClassifier(
                    n_estimators=n_estimators * 3,
                    max_depth=max_depth - 2,
                    learning_rate=learning_rate * 0.5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.5,
                    reg_lambda=0.5,
                    min_child_samples=min_samples_leaf,
                    random_state=self.random_state + 1 if self.random_state is not None else None,
                    verbose=-1
                )
            ])

            # CatBoost models with different configurations
            models.extend([
                cb.CatBoostClassifier(
                    iterations=n_estimators * 2,
                    depth=max_depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=10,
                    random_state=self.random_state,
                    verbose=False
                ),
                cb.CatBoostClassifier(
                    iterations=n_estimators * 3,
                    depth=max_depth - 2,
                    learning_rate=learning_rate * 0.5,
                    l2_leaf_reg=50,
                    random_state=self.random_state + 1 if self.random_state is not None else None,
                    verbose=False
                )
            ])

            # Limit the number of models if configured
            if self.config and hasattr(self.config, 'n_base_models'):
                models = models[:self.config.n_base_models]

            logger.debug(f"Created {len(models)} default base models with n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, learning_rate={learning_rate}")

            return models

        except Exception as e:
            logger.error(f"Error creating default models: {e}")
            raise ImportError(f"Could not create default models: {e}") from e

    def _create_stability_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features that help detect instability.

        Args:
            X: Input feature matrix

        Returns:
            Feature matrix with stability indicators
        """
        X_stability = X.copy()

        # Feature interaction ratios (detect when relationships break down)
        numeric_cols = X_stability.select_dtypes(include=[np.number]).columns[:10]  # Limit for performance

        if len(numeric_cols) >= 2:
            # Create ratio features
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:i+3]:  # Limit combinations
                    ratio_name = f'{col1}_{col2}_ratio'
                    X_stability[ratio_name] = (
                        X_stability[col1] / (X_stability[col2] + 1e-8)
                    )

        # Feature magnitude indicators (detect distribution shifts)
        for col in numeric_cols[:5]:
            X_stability[f'{col}_magnitude'] = np.abs(X_stability[col])
            X_stability[f'{col}_squared'] = X_stability[col] ** 2

        logger.debug(f"Created stability features: {X.shape} -> {X_stability.shape}")
        return X_stability

    @log_performance(log_memory=True, memory_threshold_mb=500.0)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'StabilityWeightedEnsemble':
        """Fit the stability-weighted ensemble.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting StabilityWeightedEnsemble on {X.shape} samples")

        # Log data information using structured logger
        if struct_logger:
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            y_dist = y_series.value_counts().to_dict()
            struct_logger.log_data_info(X.shape, y_dist, data_source="training")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, pd.Series):
            y = y.values

        self.feature_names_ = X.columns.tolist()

        # Create stability-aware features
        X_enhanced = self._create_stability_features(X)

        # Initialize base models
        self.models_ = self._get_default_models()
        self.calibrators_ = []
        self.monitors_ = []

        # Fit each base model with calibration
        for i, model in enumerate(self.models_):
            logger.info(f"Fitting base model {i+1}/{len(self.models_)}")

            # Fit model
            model.fit(X_enhanced, y)

            # Create calibrated version
            calibrator = CalibratedClassifierCV(
                model,
                method='isotonic',
                cv=3
            )
            calibrator.fit(X_enhanced, y)
            self.calibrators_.append(calibrator)

            # Initialize calibration monitor
            monitor = CalibrationMonitor(self.calibration_window)
            self.monitors_.append(monitor)

        # Initialize equal weights
        self.weights_ = np.ones(len(self.models_)) / len(self.models_)

        # Perform initial calibration assessment
        self._assess_initial_calibration(X_enhanced, y)

        self.is_fitted_ = True
        logger.info("StabilityWeightedEnsemble fitted successfully")
        return self

    def _assess_initial_calibration(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Assess initial calibration quality of each model.

        Args:
            X: Feature matrix
            y: Target vector
        """
        kfold = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            for i, calibrator in enumerate(self.calibrators_):
                y_prob = calibrator.predict_proba(X_val)[:, 1]
                self.monitors_[i].update(y_prob, y_val)

        logger.info("Initial calibration assessment completed")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities with stability weighting.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        # Create stability features
        X_enhanced = self._create_stability_features(X)

        # Get predictions from each model
        predictions_list = []
        for calibrator in self.calibrators_:
            pred = calibrator.predict_proba(X_enhanced)
            predictions_list.append(pred)

        predictions = np.array(predictions_list)  # Shape: (n_models, n_samples, n_classes)

        # Apply stability weights
        weighted_pred = np.zeros_like(predictions[0])
        for i, weight in enumerate(self.weights_):
            weighted_pred += weight * predictions[i]

        return weighted_pred

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def update_weights(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        update_monitors: bool = True
    ) -> Dict[str, float]:
        """Update model weights based on recent performance.

        Args:
            X: Recent feature matrix
            y: Recent true labels
            update_monitors: Whether to update calibration monitors

        Returns:
            Dictionary with weight update statistics
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before weight updates")

        # Convert inputs
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        if isinstance(y, pd.Series):
            y = y.values

        X_enhanced = self._create_stability_features(X)

        # Calculate performance metrics for each model
        calibration_scores = []
        auc_scores = []

        for i, calibrator in enumerate(self.calibrators_):
            y_prob = calibrator.predict_proba(X_enhanced)[:, 1]

            # Update monitor if requested
            if update_monitors:
                brier_score = self.monitors_[i].update(y_prob, y)
            else:
                brier_score = brier_score_loss(y, y_prob)

            calibration_scores.append(brier_score)

            # Calculate AUC
            try:
                auc = roc_auc_score(y, y_prob)
            except ValueError:
                auc = 0.5  # Default for edge cases
            auc_scores.append(auc)

        # Calculate stability scores (lower Brier + higher AUC = higher stability)
        calibration_scores = np.array(calibration_scores)
        auc_scores = np.array(auc_scores)

        # Stability score: combine calibration and discrimination
        # Invert Brier score so higher is better
        inverted_brier = 1.0 / (calibration_scores + 1e-8)
        stability_scores = 0.6 * auc_scores + 0.4 * inverted_brier

        # Update weights using exponential moving average
        new_weights = softmax(stability_scores)
        self.weights_ = (
            (1 - self.stability_alpha) * self.weights_ +
            self.stability_alpha * new_weights
        )

        # Apply minimum weight constraint
        self.weights_ = np.maximum(self.weights_, self.min_weight)
        self.weights_ = self.weights_ / self.weights_.sum()  # Renormalize

        # Check for recalibration need
        max_brier = max(calibration_scores)
        needs_recalibration = max_brier > self.recalibration_threshold

        stats = {
            'max_brier_score': float(max_brier),
            'mean_auc': float(np.mean(auc_scores)),
            'weight_entropy': float(-np.sum(self.weights_ * np.log(self.weights_ + 1e-8))),
            'needs_recalibration': needs_recalibration,
            'dominant_model': int(np.argmax(self.weights_))
        }

        logger.info(f"Updated weights: {self.weights_}, stats: {stats}")
        return stats

    def get_feature_importance(self) -> Dict[str, float]:
        """Get weighted feature importance across all models.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_dict = {}

        for i, model in enumerate(self.models_):
            weight = self.weights_[i]

            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.feature_names_

                # Handle enhanced features
                if len(importances) > len(self.feature_names_):
                    # Extended with stability features
                    extended_features = [f'stability_feature_{j}' for j in range(len(importances) - len(self.feature_names_))]
                    feature_names = self.feature_names_ + extended_features

                for feat, imp in zip(feature_names, importances):
                    if feat not in importance_dict:
                        importance_dict[feat] = 0.0
                    importance_dict[feat] += weight * imp

        # Normalize importance scores
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}

        return importance_dict

    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'models': self.models_,
            'calibrators': self.calibrators_,
            'weights': self.weights_,
            'feature_names': self.feature_names_,
            'monitors': self.monitors_,
            'config': {
                'calibration_window': self.calibration_window,
                'stability_alpha': self.stability_alpha,
                'min_weight': self.min_weight,
                'recalibration_threshold': self.recalibration_threshold,
                'random_state': self.random_state
            }
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'StabilityWeightedEnsemble':
        """Load a fitted model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded StabilityWeightedEnsemble instance
        """
        model_data = joblib.load(filepath)
        config = model_data['config']

        # Create instance
        instance = cls(
            calibration_window=config['calibration_window'],
            stability_alpha=config['stability_alpha'],
            min_weight=config['min_weight'],
            recalibration_threshold=config['recalibration_threshold'],
            random_state=config['random_state']
        )

        # Restore state
        instance.models_ = model_data['models']
        instance.calibrators_ = model_data['calibrators']
        instance.weights_ = model_data['weights']
        instance.feature_names_ = model_data['feature_names']
        instance.monitors_ = model_data['monitors']
        instance.is_fitted_ = True

        logger.info(f"Model loaded from {filepath}")
        return instance