"""Custom model components for temporal credit risk prediction.

This module contains reusable components that enhance the stability-weighted
ensemble with advanced drift detection and adaptive calibration mechanisms.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TemporalDriftDetector:
    """Detects concept drift in temporal credit data using statistical tests.

    This component monitors feature distributions and prediction patterns over time,
    triggering alerts when significant drift is detected. Uses both KS tests for
    feature drift and Jensen-Shannon divergence for prediction drift.
    """

    def __init__(
        self,
        window_size: int = 1000,
        significance_level: float = 0.05,
        drift_threshold: float = 0.1
    ):
        """Initialize drift detector.

        Args:
            window_size: Size of sliding window for drift detection
            significance_level: P-value threshold for statistical significance
            drift_threshold: Threshold for JS divergence drift detection
        """
        self.window_size = window_size
        self.significance_level = significance_level
        self.drift_threshold = drift_threshold

        self.reference_features: Optional[np.ndarray] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.drift_history: List[Dict[str, float]] = []

    def fit_reference(
        self,
        X: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ) -> 'TemporalDriftDetector':
        """Fit reference distributions for drift detection.

        Args:
            X: Reference feature matrix
            predictions: Reference prediction probabilities

        Returns:
            Self for method chaining
        """
        self.reference_features = X[-self.window_size:].copy()
        if predictions is not None:
            self.reference_predictions = predictions[-self.window_size:].copy()

        logger.info(
            f"Reference distributions fitted with {len(self.reference_features)} samples"
        )
        return self

    def detect_feature_drift(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Detect feature drift using Kolmogorov-Smirnov test.

        Args:
            X_new: New feature matrix to test for drift

        Returns:
            Dictionary containing drift detection results
        """
        if self.reference_features is None:
            raise ValueError("Must fit reference distributions first")

        n_features = X_new.shape[1]
        drift_detected = []
        p_values = []

        for i in range(n_features):
            ref_values = self.reference_features[:, i]
            new_values = X_new[:, i]

            # KS test for distribution shift
            ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
            p_values.append(p_value)

            if p_value < self.significance_level:
                drift_detected.append(i)

        drift_ratio = len(drift_detected) / n_features

        result = {
            'drift_detected': len(drift_detected) > 0,
            'drift_ratio': drift_ratio,
            'drifted_features': drift_detected,
            'mean_p_value': np.mean(p_values),
            'min_p_value': np.min(p_values)
        }

        self.drift_history.append(result)
        return result

    def detect_prediction_drift(self, predictions_new: np.ndarray) -> Dict[str, Any]:
        """Detect prediction drift using Jensen-Shannon divergence.

        Args:
            predictions_new: New prediction probabilities

        Returns:
            Dictionary containing prediction drift results
        """
        if self.reference_predictions is None:
            raise ValueError("Must fit reference predictions first")

        # Create histogram distributions
        bins = np.linspace(0, 1, 21)
        ref_hist, _ = np.histogram(self.reference_predictions, bins=bins, density=True)
        new_hist, _ = np.histogram(predictions_new, bins=bins, density=True)

        # Add small epsilon to avoid division by zero
        ref_hist = ref_hist + 1e-10
        new_hist = new_hist + 1e-10

        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, new_hist)

        result = {
            'drift_detected': js_div > self.drift_threshold,
            'js_divergence': float(js_div),
            'prediction_shift': float(
                np.mean(predictions_new) - np.mean(self.reference_predictions)
            )
        }

        return result


class AdaptiveCalibrator(BaseEstimator, TransformerMixin):
    """Adaptive calibration component that recalibrates based on drift signals.

    This component uses isotonic regression for non-parametric calibration and
    can be retrained dynamically when drift is detected to maintain calibration
    quality over time.
    """

    def __init__(self, calibration_method: str = 'isotonic', min_samples: int = 100):
        """Initialize adaptive calibrator.

        Args:
            calibration_method: Calibration method ('isotonic' or 'sigmoid')
            min_samples: Minimum samples required for calibration
        """
        self.calibration_method = calibration_method
        self.min_samples = min_samples
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'AdaptiveCalibrator':
        """Fit calibrator on prediction probabilities.

        Args:
            y_prob: Predicted probabilities (uncalibrated)
            y_true: True binary labels

        Returns:
            Self for method chaining
        """
        if len(y_prob) < self.min_samples:
            logger.warning(
                f"Insufficient samples for calibration: {len(y_prob)} < {self.min_samples}"
            )
            return self

        if self.calibration_method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
            self.is_fitted = True
            logger.info(f"Calibrator fitted on {len(y_prob)} samples")
        else:
            raise ValueError(f"Unsupported calibration method: {self.calibration_method}")

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to prediction probabilities.

        Args:
            y_prob: Uncalibrated prediction probabilities

        Returns:
            Calibrated prediction probabilities
        """
        if not self.is_fitted or self.calibrator is None:
            logger.warning("Calibrator not fitted, returning uncalibrated probabilities")
            return y_prob

        calibrated = self.calibrator.predict(y_prob)
        return np.clip(calibrated, 0, 1)

    def recalibrate(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'AdaptiveCalibrator':
        """Recalibrate with new data (alias for fit for clarity).

        Args:
            y_prob: New predicted probabilities
            y_true: New true labels

        Returns:
            Self for method chaining
        """
        return self.fit(y_prob, y_true)


class StabilityScorer:
    """Calculates stability scores for ensemble model weighting.

    This component computes composite stability scores based on calibration quality,
    prediction consistency, and temporal performance trends.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3):
        """Initialize stability scorer.

        Args:
            alpha: Weight for calibration component
            beta: Weight for consistency component
            gamma: Weight for trend component
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError("Stability score weights must sum to 1.0")

    def calculate_calibration_score(self, brier_score: float) -> float:
        """Calculate calibration component of stability score.

        Args:
            brier_score: Brier score (lower is better)

        Returns:
            Calibration score in [0, 1] (higher is better)
        """
        # Transform Brier score to [0, 1] where higher is better
        # Brier score ranges from 0 (perfect) to 1 (worst)
        return 1.0 - np.clip(brier_score, 0, 1)

    def calculate_consistency_score(
        self,
        predictions: np.ndarray,
        previous_predictions: Optional[np.ndarray] = None
    ) -> float:
        """Calculate prediction consistency score.

        Args:
            predictions: Current predictions
            previous_predictions: Previous predictions for same samples

        Returns:
            Consistency score in [0, 1]
        """
        if previous_predictions is None or len(previous_predictions) == 0:
            return 1.0  # No baseline to compare

        # Calculate prediction variance as measure of consistency
        pred_std = np.std(predictions)
        consistency = 1.0 / (1.0 + pred_std)
        return float(np.clip(consistency, 0, 1))

    def calculate_trend_score(self, performance_history: List[float]) -> float:
        """Calculate performance trend component.

        Args:
            performance_history: Recent performance metrics (higher is better)

        Returns:
            Trend score in [0, 1]
        """
        if len(performance_history) < 2:
            return 1.0

        # Calculate trend using linear regression slope
        x = np.arange(len(performance_history))
        slope = np.polyfit(x, performance_history, 1)[0]

        # Transform slope to [0, 1] score
        # Positive slope → score > 0.5, negative slope → score < 0.5
        trend_score = 0.5 + np.tanh(slope * 10) * 0.5
        return float(np.clip(trend_score, 0, 1))

    def calculate_stability_score(
        self,
        brier_score: float,
        predictions: np.ndarray,
        performance_history: List[float],
        previous_predictions: Optional[np.ndarray] = None
    ) -> float:
        """Calculate composite stability score.

        Args:
            brier_score: Current Brier score
            predictions: Current predictions
            performance_history: Recent performance history
            previous_predictions: Previous predictions for consistency

        Returns:
            Composite stability score in [0, 1]
        """
        calib_score = self.calculate_calibration_score(brier_score)
        consist_score = self.calculate_consistency_score(predictions, previous_predictions)
        trend_score = self.calculate_trend_score(performance_history)

        stability = (
            self.alpha * calib_score +
            self.beta * consist_score +
            self.gamma * trend_score
        )

        return float(np.clip(stability, 0, 1))
