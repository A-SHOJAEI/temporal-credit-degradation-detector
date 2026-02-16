"""Training pipeline with MLflow integration and early stopping."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
import mlflow
import mlflow.sklearn
import optuna
try:
    from optuna.integration import OptunaSearchCV
except ImportError:
    # Fallback if optuna-integration is not installed
    OptunaSearchCV = None
import joblib
import warnings

from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble
from temporal_credit_degradation_detector.data.preprocessing import CreditDataPreprocessor
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility for training."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if early stopping criteria is met.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        self.early_stop = self.counter >= self.patience
        return self.early_stop


class ModelTrainer:
    """Comprehensive training pipeline with MLflow tracking."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer.

        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.experiment_name = config.get('experiment_name', 'temporal_credit_model')
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)

        # Initialize MLflow
        self._setup_mlflow()

        # Initialize components
        self.preprocessor: Optional[CreditDataPreprocessor] = None
        self.model: Optional[StabilityWeightedEnsemble] = None
        self.evaluator = ModelEvaluator()

        logger.info(f"Initialized ModelTrainer with experiment: {self.experiment_name}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without tracking.")

    def prepare_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for training.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Preprocessed training and validation data
        """
        logger.info(f"Preparing data: train {X_train.shape}, val {X_val.shape if X_val is not None else 'None'}")

        # Initialize and fit preprocessor
        preprocessor_config = self.config.get('preprocessing', {})
        self.preprocessor = CreditDataPreprocessor(preprocessor_config)

        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train, y_train)

        # Transform validation data if provided
        X_val_processed = None
        if X_val is not None:
            X_val_processed = self.preprocessor.transform(X_val)

        logger.info(f"Data preprocessing complete: train {X_train_processed.shape}")

        return X_train_processed, y_train, X_val_processed, y_val

    def _objective_function(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Validation AUC score
        """
        # Suggest hyperparameters using configurable ranges
        training_config = self.config.get('training', {})

        params = {
            'stability_alpha': trial.suggest_float(
                'stability_alpha',
                training_config.get('stability_alpha_min', 0.01),
                training_config.get('stability_alpha_max', 0.3)
            ),
            'min_weight': trial.suggest_float(
                'min_weight',
                training_config.get('min_weight_min', 0.01),
                training_config.get('min_weight_max', 0.2)
            ),
            'recalibration_threshold': trial.suggest_float(
                'recalibration_threshold',
                0.1, 0.25  # Keep this range fixed as it's calibration-specific
            ),
            'calibration_window': trial.suggest_int(
                'calibration_window',
                training_config.get('calibration_window_min', 500),
                training_config.get('calibration_window_max', 2000),
                step=250
            ),
            'random_state': self.config.get('random_state', 42)
        }

        # Create and train model
        model = StabilityWeightedEnsemble(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)

        return auc_score

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        # Create study
        study = optuna.create_study(direction='maximize')

        # Run optimization
        study.optimize(
            lambda trial: self._objective_function(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials
        )

        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")

        # Log to MLflow
        try:
            with mlflow.start_run(nested=True):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_auc", study.best_value)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        return best_params

    def train_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        optimize_hp: bool = True,
        n_hp_trials: int = 30
    ) -> Dict[str, Any]:
        """Train model with cross-validation and comprehensive error handling.

        Args:
            X: Features dataframe
            y: Target series (binary labels)
            cv_folds: Number of CV folds (must be >= 2)
            optimize_hp: Whether to optimize hyperparameters
            n_hp_trials: Number of hyperparameter optimization trials (must be >= 1)

        Returns:
            Training results and metrics

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If training fails completely
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty")

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

        if n_hp_trials < 1:
            raise ValueError(f"n_hp_trials must be >= 1, got {n_hp_trials}")

        if len(X) < cv_folds:
            raise ValueError(f"Not enough samples ({len(X)}) for {cv_folds} folds")

        unique_labels = y.nunique()
        if unique_labels < 2:
            raise ValueError(f"Need at least 2 classes for classification, found {unique_labels}")

        logger.info(f"Starting {cv_folds}-fold cross-validation training on {len(X)} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        # Initialize cross-validation with error handling
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.get('random_state', 42))
        except Exception as e:
            logger.error(f"Error initializing StratifiedKFold: {e}")
            raise RuntimeError(f"Could not initialize cross-validation: {e}") from e

        cv_scores = []
        cv_predictions = []
        fold_models = []
        failed_folds = []

        try:
            with mlflow.start_run():
                # Log configuration with error handling
                try:
                    mlflow.log_params(self.config)
                except Exception as e:
                    logger.warning(f"Could not log MLflow params: {e}")

                best_params = None

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    logger.info(f"Training fold {fold + 1}/{cv_folds}")

                    try:
                        # Split data
                        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

                        # Validate fold splits
                        if len(X_fold_train) == 0 or len(X_fold_val) == 0:
                            raise ValueError(f"Empty split in fold {fold + 1}")

                        if y_fold_train.nunique() < 2:
                            raise ValueError(f"Fold {fold + 1} training set has only one class")

                        # Prepare data with error handling
                        try:
                            X_fold_train_prep, y_fold_train, X_fold_val_prep, y_fold_val = self.prepare_data(
                                X_fold_train, y_fold_train, X_fold_val, y_fold_val
                            )
                        except Exception as e:
                            logger.error(f"Data preparation failed for fold {fold + 1}: {e}")
                            raise

                        # Optimize hyperparameters on first fold
                        if fold == 0 and optimize_hp and best_params is None:
                            try:
                                logger.info("Optimizing hyperparameters on first fold")
                                best_params = self.optimize_hyperparameters(
                                    X_fold_train_prep, y_fold_train,
                                    X_fold_val_prep, y_fold_val,
                                    n_trials=n_hp_trials
                                )
                                logger.info(f"Best hyperparameters: {best_params}")
                            except Exception as e:
                                logger.warning(f"Hyperparameter optimization failed: {e}. Using default params.")
                                best_params = self.config.get('model_params', {})
                        elif best_params is None:
                            best_params = self.config.get('model_params', {})

                        # Train model with error handling
                        try:
                            model = StabilityWeightedEnsemble(
                                **best_params,
                                random_state=self.config.get('random_state', 42)
                            )
                            model.fit(X_fold_train_prep, y_fold_train)
                        except Exception as e:
                            logger.error(f"Model training failed for fold {fold + 1}: {e}")
                            raise

                        # Evaluate fold with error handling
                        try:
                            y_pred_proba = model.predict_proba(X_fold_val_prep)[:, 1]
                            fold_metrics = self.evaluator.calculate_metrics(y_fold_val, y_pred_proba)
                        except Exception as e:
                            logger.error(f"Model evaluation failed for fold {fold + 1}: {e}")
                            raise

                        # Store results
                        cv_scores.append(fold_metrics)
                        cv_predictions.extend(list(zip(val_idx, y_pred_proba, y_fold_val)))
                        fold_models.append(model)

                        # Log fold results with error handling
                        try:
                            for metric, value in fold_metrics.items():
                                mlflow.log_metric(f"fold_{fold}_{metric}", value)
                        except Exception as e:
                            logger.warning(f"Could not log MLflow metrics for fold {fold + 1}: {e}")

                        logger.info(f"Fold {fold + 1} AUC: {fold_metrics.get('auc_roc', 0):.4f}")

                    except Exception as e:
                        logger.error(f"Fold {fold + 1} failed completely: {e}")
                        failed_folds.append(fold + 1)

                        # If too many folds fail, abort
                        if len(failed_folds) > cv_folds // 2:
                            raise RuntimeError(f"More than half the folds failed: {failed_folds}") from e

                        continue  # Try next fold

                # Validate we have at least some successful folds
                if len(cv_scores) == 0:
                    raise RuntimeError("All cross-validation folds failed")

                if failed_folds:
                    logger.warning(f"Failed folds: {failed_folds}. Continuing with {len(cv_scores)} successful folds.")

                # Calculate average metrics with error handling
                try:
                    avg_metrics = {}
                    for metric in cv_scores[0].keys():
                        metric_values = [score[metric] for score in cv_scores if metric in score]
                        if metric_values:  # Only calculate if we have values
                            avg_metrics[f"cv_{metric}_mean"] = np.mean(metric_values)
                            avg_metrics[f"cv_{metric}_std"] = np.std(metric_values)

                    # Log average metrics
                    try:
                        for metric, value in avg_metrics.items():
                            mlflow.log_metric(metric, value)
                    except Exception as e:
                        logger.warning(f"Could not log average metrics to MLflow: {e}")

                except Exception as e:
                    logger.error(f"Error calculating average metrics: {e}")
                    avg_metrics = {}

                # Select best model (based on validation AUC) with error handling
                try:
                    auc_scores = [score.get('auc_roc', 0) for score in cv_scores]
                    if not auc_scores:
                        raise ValueError("No AUC scores available for model selection")

                    best_fold = np.argmax(auc_scores)
                    if best_fold >= len(fold_models):
                        raise IndexError(f"Best fold index {best_fold} exceeds available models {len(fold_models)}")

                    self.model = fold_models[best_fold]
                    logger.info(f"Selected model from fold {best_fold + 1} with AUC: {auc_scores[best_fold]:.4f}")

                except Exception as e:
                    logger.error(f"Error selecting best model: {e}")
                    if fold_models:
                        logger.warning("Using first available model as fallback")
                        self.model = fold_models[0]
                        best_fold = 0
                    else:
                        raise RuntimeError("No trained models available") from e

                # Save model with error handling
                try:
                    model_path = self.model_dir / f"best_model_cv.pkl"
                    self.model.save_model(str(model_path))
                    try:
                        mlflow.log_artifact(str(model_path))
                    except Exception as e:
                        logger.warning(f"Could not log model artifact: {e}")
                    logger.info(f"Model saved to {model_path}")

                except Exception as e:
                    logger.error(f"Error saving model: {e}")
                    # Don't fail the entire process for save errors
                    logger.warning("Continuing without saving model")

                # Save preprocessor with error handling
                try:
                    if self.preprocessor:
                        preprocessor_path = self.model_dir / "preprocessor.pkl"
                        joblib.dump(self.preprocessor, preprocessor_path)
                        try:
                            mlflow.log_artifact(str(preprocessor_path))
                        except Exception as e:
                            logger.warning(f"Could not log preprocessor artifact: {e}")
                        logger.info(f"Preprocessor saved to {preprocessor_path}")
                    else:
                        logger.warning("No preprocessor to save")

                except Exception as e:
                    logger.error(f"Error saving preprocessor: {e}")
                    logger.warning("Continuing without saving preprocessor")

                logger.info(f"Cross-validation complete. Best fold: {best_fold + 1}")

                results = {
                    'cv_scores': cv_scores,
                    'avg_metrics': avg_metrics,
                    'best_fold': best_fold,
                    'predictions': cv_predictions,
                    'model': self.model,
                    'failed_folds': failed_folds
                }

                return results

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise RuntimeError(f"Cross-validation training failed: {e}") from e

    def train_with_early_stopping(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        max_updates: int = 50,
        patience: int = 5
    ) -> Dict[str, Any]:
        """Train with early stopping based on validation performance.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            max_updates: Maximum number of weight updates
            patience: Early stopping patience

        Returns:
            Training results
        """
        logger.info("Training with early stopping")

        # Prepare data
        X_train_prep, y_train, X_val_prep, y_val = self.prepare_data(
            X_train, y_train, X_val, y_val
        )

        # Initialize model
        model_params = self.config.get('model_params', {})
        self.model = StabilityWeightedEnsemble(
            **model_params,
            random_state=self.config.get('random_state', 42)
        )

        # Fit initial model
        self.model.fit(X_train_prep, y_train)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, mode='max')

        training_history = []

        with mlflow.start_run():
            mlflow.log_params(self.config)

            for update in range(max_updates):
                # Evaluate current model
                y_pred_proba = self.model.predict_proba(X_val_prep)[:, 1]
                val_metrics = self.evaluator.calculate_metrics(y_val, y_pred_proba)

                # Log metrics
                for metric, value in val_metrics.items():
                    mlflow.log_metric(metric, value, step=update)

                training_history.append(val_metrics)

                # Check early stopping
                val_auc = val_metrics.get('auc_roc', 0)
                if early_stopping(val_auc):
                    logger.info(f"Early stopping triggered at update {update}")
                    break

                # Update model weights
                weight_stats = self.model.update_weights(X_val_prep, y_val, update_monitors=True)

                # Log weight statistics
                for stat_name, stat_value in weight_stats.items():
                    mlflow.log_metric(f"weight_{stat_name}", stat_value, step=update)

                if update % 10 == 0:
                    logger.info(f"Update {update}: Val AUC = {val_auc:.4f}")

            # Save final model
            model_path = self.model_dir / "final_model.pkl"
            self.model.save_model(str(model_path))
            mlflow.log_artifact(str(model_path))

            # Save preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            joblib.dump(self.preprocessor, preprocessor_path)
            mlflow.log_artifact(str(preprocessor_path))

            results = {
                'training_history': training_history,
                'final_metrics': val_metrics,
                'total_updates': update + 1,
                'model': self.model
            }

            logger.info(f"Training complete after {update + 1} updates")

            return results

    def evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """Evaluate trained model on test set.

        Args:
            X_test: Test features
            y_test: Test targets
            save_predictions: Whether to save test predictions

        Returns:
            Test evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating model on test set: {X_test.shape}")

        # Preprocess test data
        X_test_prep = self.preprocessor.transform(X_test)

        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test_prep)[:, 1]
        y_pred = self.model.predict(X_test_prep)

        # Calculate comprehensive metrics
        test_metrics = self.evaluator.calculate_metrics(y_test, y_pred_proba, y_pred)

        # Additional analysis
        feature_importance = self.model.get_feature_importance()

        # Log to MLflow
        try:
            with mlflow.start_run():
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric}", value)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        # Save predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'actual': y_test.values,
                'predicted_proba': y_pred_proba,
                'predicted_class': y_pred
            })
            pred_path = self.model_dir / "test_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            logger.info(f"Predictions saved to {pred_path}")

        results = {
            'metrics': test_metrics,
            'predictions': y_pred_proba,
            'feature_importance': feature_importance,
            'model_weights': self.model.weights_
        }

        logger.info(f"Test evaluation complete. AUC: {test_metrics.get('auc_roc', 0):.4f}")

        return results

    def save_training_artifacts(self, results: Dict[str, Any], suffix: str = "") -> None:
        """Save training artifacts to disk.

        Args:
            results: Training results dictionary
            suffix: Optional suffix for file names
        """
        suffix = f"_{suffix}" if suffix else ""

        # Save training history if available
        if 'training_history' in results:
            history_path = self.model_dir / f"training_history{suffix}.json"
            import json
            with open(history_path, 'w') as f:
                json.dump(results['training_history'], f, indent=2)

        # Save metrics
        if 'avg_metrics' in results:
            metrics_path = self.model_dir / f"cv_metrics{suffix}.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump(results['avg_metrics'], f, indent=2)

        logger.info(f"Training artifacts saved to {self.model_dir}")

    def load_model(self, model_path: str, preprocessor_path: str) -> None:
        """Load trained model and preprocessor.

        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model = StabilityWeightedEnsemble.load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Model and preprocessor loaded from {model_path}, {preprocessor_path}")