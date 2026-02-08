"""Comprehensive evaluation metrics including drift detection capabilities."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftDetector:
    """Advanced drift detection for credit risk models."""

    def __init__(self, window_size: int = 1000, significance_level: float = 0.05):
        """Initialize drift detector.

        Args:
            window_size: Size of reference window for comparison
            significance_level: Statistical significance threshold
        """
        self.window_size = window_size
        self.significance_level = significance_level
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_targets: Optional[np.ndarray] = None

    def set_reference(
        self,
        X_ref: pd.DataFrame,
        y_pred_ref: np.ndarray,
        y_true_ref: Optional[np.ndarray] = None
    ) -> None:
        """Set reference data for drift detection.

        Args:
            X_ref: Reference feature matrix
            y_pred_ref: Reference predictions
            y_true_ref: Reference true labels (optional)
        """
        self.reference_data = X_ref.copy()
        self.reference_predictions = y_pred_ref.copy()
        if y_true_ref is not None:
            self.reference_targets = y_true_ref.copy()

        logger.info(f"Reference data set: {len(X_ref)} samples")

    def detect_feature_drift(
        self,
        X_current: pd.DataFrame,
        method: str = 'ks_test'
    ) -> Dict[str, Dict[str, float]]:
        """Detect feature-level drift using statistical tests.

        Args:
            X_current: Current feature matrix
            method: Statistical test method ('ks_test', 'js_divergence', 'chi2_test')

        Returns:
            Dictionary with drift statistics for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        drift_results = {}
        common_features = list(set(self.reference_data.columns) & set(X_current.columns))

        for feature in common_features:
            ref_values = self.reference_data[feature].values
            current_values = X_current[feature].values

            drift_stats = {}

            if method == 'ks_test':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_values, current_values)
                drift_stats['statistic'] = float(statistic)
                drift_stats['p_value'] = float(p_value)
                drift_stats['is_drift'] = p_value < self.significance_level

            elif method == 'js_divergence':
                # Jensen-Shannon divergence
                # Create histograms
                min_val = min(ref_values.min(), current_values.min())
                max_val = max(ref_values.max(), current_values.max())
                bins = np.linspace(min_val, max_val, 50)

                hist_ref, _ = np.histogram(ref_values, bins=bins, density=True)
                hist_current, _ = np.histogram(current_values, bins=bins, density=True)

                # Normalize to probabilities
                hist_ref = hist_ref / hist_ref.sum()
                hist_current = hist_current / hist_current.sum()

                js_div = jensenshannon(hist_ref, hist_current)
                drift_stats['js_divergence'] = float(js_div)
                drift_stats['is_drift'] = js_div > 0.1  # Threshold for JS divergence

            elif method == 'chi2_test':
                # Chi-square test (for categorical features)
                try:
                    # Discretize continuous features
                    ref_discretized = pd.cut(ref_values, bins=10, duplicates='drop')
                    current_discretized = pd.cut(current_values, bins=10, duplicates='drop')

                    ref_counts = ref_discretized.value_counts()
                    current_counts = current_discretized.value_counts()

                    # Align categories
                    all_cats = set(ref_counts.index) | set(current_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
                    current_aligned = [current_counts.get(cat, 0) for cat in all_cats]

                    statistic, p_value = stats.chisquare(current_aligned, ref_aligned)
                    drift_stats['statistic'] = float(statistic)
                    drift_stats['p_value'] = float(p_value)
                    drift_stats['is_drift'] = p_value < self.significance_level

                except Exception:
                    drift_stats['statistic'] = np.nan
                    drift_stats['p_value'] = np.nan
                    drift_stats['is_drift'] = False

            drift_results[feature] = drift_stats

        logger.info(f"Feature drift detection complete: {sum(r['is_drift'] for r in drift_results.values())} features drifted")

        return drift_results

    def detect_prediction_drift(
        self,
        y_pred_current: np.ndarray,
        y_true_current: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Detect drift in model predictions.

        Args:
            y_pred_current: Current predictions
            y_true_current: Current true labels (optional)

        Returns:
            Dictionary with prediction drift statistics
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set. Call set_reference() first.")

        drift_stats = {}

        # Distribution shift in predictions
        ks_stat, ks_p = stats.ks_2samp(self.reference_predictions, y_pred_current)
        drift_stats['prediction_ks_statistic'] = float(ks_stat)
        drift_stats['prediction_ks_p_value'] = float(ks_p)
        drift_stats['prediction_drift'] = ks_p < self.significance_level

        # Mean prediction shift
        ref_mean = np.mean(self.reference_predictions)
        current_mean = np.mean(y_pred_current)
        drift_stats['mean_prediction_shift'] = float(current_mean - ref_mean)

        # Performance drift (if true labels available)
        if y_true_current is not None and self.reference_targets is not None:
            try:
                ref_auc = roc_auc_score(self.reference_targets, self.reference_predictions)
                current_auc = roc_auc_score(y_true_current, y_pred_current)
                drift_stats['auc_degradation'] = float(ref_auc - current_auc)
                drift_stats['significant_auc_drop'] = (ref_auc - current_auc) > 0.05

            except Exception:
                drift_stats['auc_degradation'] = np.nan
                drift_stats['significant_auc_drop'] = False

        logger.info(f"Prediction drift detection complete: {drift_stats}")

        return drift_stats

    def comprehensive_drift_assessment(
        self,
        X_current: pd.DataFrame,
        y_pred_current: np.ndarray,
        y_true_current: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Comprehensive drift assessment combining multiple methods.

        Args:
            X_current: Current feature matrix
            y_pred_current: Current predictions
            y_true_current: Current true labels (optional)

        Returns:
            Comprehensive drift assessment results
        """
        results = {}

        # Feature drift assessment
        feature_drift_ks = self.detect_feature_drift(X_current, method='ks_test')
        feature_drift_js = self.detect_feature_drift(X_current, method='js_divergence')

        results['feature_drift_ks'] = feature_drift_ks
        results['feature_drift_js'] = feature_drift_js

        # Summarize feature drift
        ks_drift_count = sum(1 for stats in feature_drift_ks.values() if stats['is_drift'])
        js_drift_count = sum(1 for stats in feature_drift_js.values() if stats['is_drift'])

        results['feature_drift_summary'] = {
            'total_features': len(feature_drift_ks),
            'ks_drift_count': ks_drift_count,
            'js_drift_count': js_drift_count,
            'ks_drift_ratio': ks_drift_count / len(feature_drift_ks) if feature_drift_ks else 0,
            'js_drift_ratio': js_drift_count / len(feature_drift_js) if feature_drift_js else 0
        }

        # Prediction drift assessment
        prediction_drift = self.detect_prediction_drift(y_pred_current, y_true_current)
        results['prediction_drift'] = prediction_drift

        # Overall drift score
        feature_drift_score = max(
            results['feature_drift_summary']['ks_drift_ratio'],
            results['feature_drift_summary']['js_drift_ratio']
        )
        prediction_drift_score = 1.0 if prediction_drift['prediction_drift'] else 0.0

        results['overall_drift_score'] = 0.7 * feature_drift_score + 0.3 * prediction_drift_score

        logger.info(f"Comprehensive drift assessment complete. Overall score: {results['overall_drift_score']:.3f}")

        return results


class ModelEvaluator:
    """Comprehensive model evaluation with temporal awareness."""

    def __init__(self, calibration_bins: int = 10):
        """Initialize model evaluator.

        Args:
            calibration_bins: Number of bins for calibration analysis
        """
        self.calibration_bins = calibration_bins
        self.drift_detector = DriftDetector()

    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        y_pred: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics with robust error handling.

        Args:
            y_true: True labels (binary: 0 or 1)
            y_prob: Predicted probabilities (must be between 0 and 1)
            y_pred: Predicted labels (optional, will be derived from y_prob)
            sample_weight: Sample weights (optional, must be positive)

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If inputs are invalid or incompatible
            TypeError: If inputs have wrong type
        """
        # Input validation and conversion
        try:
            y_true = np.asarray(y_true).flatten()
            y_prob = np.asarray(y_prob).flatten()
        except (ValueError, TypeError) as e:
            raise TypeError(f"Cannot convert inputs to numpy arrays: {e}")

        # Validate array lengths
        if len(y_true) != len(y_prob):
            raise ValueError(f"Mismatched lengths: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")

        if len(y_true) == 0:
            raise ValueError("Empty input arrays provided")

        # Validate y_true values
        unique_y_true = np.unique(y_true)
        if not np.all(np.isin(unique_y_true, [0, 1])):
            raise ValueError(f"y_true must contain only 0 and 1, found: {unique_y_true}")

        if len(unique_y_true) == 1:
            logger.warning(f"y_true contains only one class: {unique_y_true[0]}. Some metrics may be unreliable.")

        # Validate y_prob values
        if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
            raise ValueError("y_prob contains NaN or infinite values")

        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            prob_min, prob_max = y_prob.min(), y_prob.max()
            raise ValueError(f"y_prob values must be in [0,1], found range [{prob_min:.3f}, {prob_max:.3f}]")

        # Handle y_pred
        if y_pred is None:
            y_pred = (y_prob > 0.5).astype(int)
        else:
            try:
                y_pred = np.asarray(y_pred).flatten()
                if len(y_pred) != len(y_true):
                    raise ValueError(f"y_pred length ({len(y_pred)}) doesn't match y_true ({len(y_true)})")
                if not np.all(np.isin(y_pred, [0, 1])):
                    raise ValueError("y_pred must contain only 0 and 1")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid y_pred provided: {e}. Deriving from y_prob.")
                y_pred = (y_prob > 0.5).astype(int)

        # Handle sample weights
        if sample_weight is not None:
            try:
                sample_weight = np.asarray(sample_weight).flatten()
                if len(sample_weight) != len(y_true):
                    raise ValueError(f"sample_weight length ({len(sample_weight)}) doesn't match y_true ({len(y_true)})")
                if np.any(sample_weight < 0):
                    raise ValueError("sample_weight must be non-negative")
                if np.all(sample_weight == 0):
                    raise ValueError("All sample weights are zero")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid sample_weight: {e}. Ignoring sample weights.")
                sample_weight = None

        logger.debug(f"Validated inputs: {len(y_true)} samples, "
                    f"{len(unique_y_true)} classes, "
                    f"prob range [{y_prob.min():.3f}, {y_prob.max():.3f}]")

        metrics = {}

        try:
            # Classification metrics
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob, sample_weight=sample_weight)

            # Threshold-based metrics
            metrics['precision'] = precision_score(y_true, y_pred, sample_weight=sample_weight)
            metrics['recall'] = recall_score(y_true, y_pred, sample_weight=sample_weight)
            metrics['f1_score'] = f1_score(y_true, y_pred, sample_weight=sample_weight)

            # Calibration metrics
            metrics['brier_score'] = brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
            metrics['log_loss'] = log_loss(y_true, y_prob, sample_weight=sample_weight)

            # Custom calibration error
            calibration_error = self.calculate_calibration_error(y_true, y_prob)
            metrics['calibration_error'] = calibration_error

            # Confusion matrix derived metrics
            cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Business metrics for credit risk
            metrics.update(self.calculate_business_metrics(y_true, y_prob))

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            logger.error(f"Input shapes - y_true: {y_true.shape if hasattr(y_true, 'shape') else 'N/A'}, "
                        f"y_prob: {y_prob.shape if hasattr(y_prob, 'shape') else 'N/A'}")

            # Return safe fallback metrics based on input validation
            try:
                # Try to calculate basic AUC if possible
                if len(np.unique(y_true)) == 2 and len(y_prob) > 0:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['auc_roc'] = 0.5
            except:
                metrics['auc_roc'] = 0.5

            # Safe fallback values
            metrics['brier_score'] = 1.0
            metrics['calibration_error'] = 1.0
            metrics['accuracy'] = (y_true == np.round(y_prob)).mean() if len(y_true) > 0 else 0.0
            metrics['log_loss'] = 1.0

            logger.warning("Using fallback metrics due to calculation errors")

        return metrics

    def calculate_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error (ECE).

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            Expected Calibration Error
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=self.calibration_bins
            )

            bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper, fraction_pos, mean_pred in zip(
                bin_lowers, bin_uppers, fraction_of_positives, mean_predicted_value
            ):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return float(ece)

        except Exception:
            return 1.0

    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        default_loss: float = 1.0,
        profit_margin: float = 0.15
    ) -> Dict[str, float]:
        """Calculate business-relevant metrics for credit risk.

        Args:
            y_true: True labels (1 = default, 0 = no default)
            y_prob: Predicted default probabilities
            default_loss: Loss when a default occurs
            profit_margin: Profit margin on successful loans

        Returns:
            Dictionary of business metrics
        """
        business_metrics = {}

        try:
            # Calculate profit at different thresholds
            thresholds = np.linspace(0.1, 0.9, 9)
            profits = []

            for threshold in thresholds:
                y_pred = (y_prob > threshold).astype(int)

                # True positives: correctly identified defaults (avoided losses)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                # False positives: rejected good loans (missed profits)
                fp = np.sum((y_pred == 1) & (y_true == 0))
                # True negatives: approved good loans (earned profits)
                tn = np.sum((y_pred == 0) & (y_true == 0))
                # False negatives: approved bad loans (incurred losses)
                fn = np.sum((y_pred == 0) & (y_true == 1))

                # Calculate profit
                profit = (tn * profit_margin) - (fn * default_loss)
                profits.append(profit)

            # Best threshold and profit
            best_idx = np.argmax(profits)
            business_metrics['optimal_threshold'] = float(thresholds[best_idx])
            business_metrics['max_profit'] = float(profits[best_idx])

            # Profit at 0.5 threshold
            y_pred_05 = (y_prob > 0.5).astype(int)
            tp_05 = np.sum((y_pred_05 == 1) & (y_true == 1))
            fp_05 = np.sum((y_pred_05 == 1) & (y_true == 0))
            tn_05 = np.sum((y_pred_05 == 0) & (y_true == 0))
            fn_05 = np.sum((y_pred_05 == 0) & (y_true == 1))

            profit_05 = (tn_05 * profit_margin) - (fn_05 * default_loss)
            business_metrics['profit_at_05_threshold'] = float(profit_05)

            # Risk-adjusted metrics
            business_metrics['approval_rate_optimal'] = float(1 - thresholds[best_idx])
            business_metrics['default_rate_if_all_approved'] = float(np.mean(y_true))

        except Exception as e:
            logger.warning(f"Error calculating business metrics: {e}")
            business_metrics['optimal_threshold'] = 0.5
            business_metrics['max_profit'] = 0.0

        return business_metrics

    def evaluate_temporal_stability(
        self,
        predictions_by_time: Dict[str, Tuple[np.ndarray, np.ndarray]],
        time_periods: List[str]
    ) -> Dict[str, Any]:
        """Evaluate model stability across time periods.

        Args:
            predictions_by_time: Dictionary mapping time periods to (y_true, y_prob) tuples
            time_periods: List of time period names

        Returns:
            Temporal stability analysis results
        """
        stability_results = {}

        # Calculate metrics for each time period
        metrics_by_time = {}
        for period in time_periods:
            if period in predictions_by_time:
                y_true, y_prob = predictions_by_time[period]
                metrics_by_time[period] = self.calculate_metrics(y_true, y_prob)

        stability_results['metrics_by_time'] = metrics_by_time

        # Calculate stability statistics
        if len(metrics_by_time) > 1:
            metric_names = list(next(iter(metrics_by_time.values())).keys())
            stability_stats = {}

            for metric in metric_names:
                values = [metrics_by_time[period][metric] for period in time_periods
                         if period in metrics_by_time]
                stability_stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
                }

            stability_results['stability_stats'] = stability_stats

            # Overall stability score (lower is more stable)
            auc_cv = stability_stats.get('auc_roc', {}).get('cv', float('inf'))
            brier_cv = stability_stats.get('brier_score', {}).get('cv', float('inf'))
            stability_results['overall_stability_score'] = float(
                0.5 * min(auc_cv, 1.0) + 0.5 * min(brier_cv, 1.0)
            )

        logger.info(f"Temporal stability evaluation complete for {len(time_periods)} periods")

        return stability_results

    def create_drift_report(
        self,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        y_pred_reference: np.ndarray,
        y_pred_current: np.ndarray,
        y_true_reference: Optional[np.ndarray] = None,
        y_true_current: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Create comprehensive drift detection report.

        Args:
            X_reference: Reference feature data
            X_current: Current feature data
            y_pred_reference: Reference predictions
            y_pred_current: Current predictions
            y_true_reference: Reference true labels (optional)
            y_true_current: Current true labels (optional)

        Returns:
            Comprehensive drift report
        """
        # Set reference data
        self.drift_detector.set_reference(X_reference, y_pred_reference, y_true_reference)

        # Perform comprehensive drift assessment
        drift_results = self.drift_detector.comprehensive_drift_assessment(
            X_current, y_pred_current, y_true_current
        )

        # Calculate performance comparison if true labels available
        if y_true_reference is not None and y_true_current is not None:
            ref_metrics = self.calculate_metrics(y_true_reference, y_pred_reference)
            current_metrics = self.calculate_metrics(y_true_current, y_pred_current)

            performance_comparison = {}
            for metric in ref_metrics:
                if metric in current_metrics:
                    performance_comparison[f'{metric}_reference'] = ref_metrics[metric]
                    performance_comparison[f'{metric}_current'] = current_metrics[metric]
                    performance_comparison[f'{metric}_degradation'] = (
                        ref_metrics[metric] - current_metrics[metric]
                    )

            drift_results['performance_comparison'] = performance_comparison

        # Generate recommendations
        recommendations = self._generate_drift_recommendations(drift_results)
        drift_results['recommendations'] = recommendations

        return drift_results

    def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on drift analysis.

        Args:
            drift_results: Results from drift analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        overall_drift_score = drift_results.get('overall_drift_score', 0)
        feature_drift_ratio = drift_results.get('feature_drift_summary', {}).get('ks_drift_ratio', 0)
        prediction_drift = drift_results.get('prediction_drift', {}).get('prediction_drift', False)

        if overall_drift_score > 0.3:
            recommendations.append("HIGH DRIFT DETECTED: Consider immediate model retraining")

        if feature_drift_ratio > 0.2:
            recommendations.append("Significant feature drift detected: Review data pipeline and feature engineering")

        if prediction_drift:
            recommendations.append("Prediction distribution has shifted: Model recalibration recommended")

        performance_comparison = drift_results.get('performance_comparison', {})
        if performance_comparison:
            auc_degradation = performance_comparison.get('auc_roc_degradation', 0)
            if auc_degradation > 0.05:
                recommendations.append("Model performance has significantly degraded: Urgent retraining required")

        if not recommendations:
            recommendations.append("No significant drift detected: Continue monitoring")

        return recommendations