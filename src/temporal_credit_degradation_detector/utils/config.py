"""Configuration management for temporal credit degradation detection."""

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the stability-weighted ensemble model."""

    # Stability weighting parameters
    stability_alpha: float = 0.1
    min_weight: float = 0.05
    recalibration_threshold: float = 0.15
    calibration_window: int = 1000
    random_state: int = 42

    # Base model parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_samples_split: int = 20
    min_samples_leaf: int = 10

    # Feature importance and stability
    stability_feature_limit: int = 10
    feature_subset_size: int = 5
    epsilon_division_safety: float = 1e-8

    # Calibration and metrics
    min_brier_buffer_size: int = 100
    js_divergence_bins: int = 50
    js_divergence_threshold: float = 0.1

    # Ensemble parameters
    n_base_models: int = 3
    stability_window_size: int = 1000


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    # General preprocessing flags
    handle_missing: bool = True
    create_temporal_features: bool = True
    create_risk_features: bool = True
    scale_features: bool = True
    remove_redundant: bool = True
    discretize_continuous: bool = False
    feature_selection: bool = False
    max_features: Optional[int] = None

    # Missing value imputation
    missing_value_fill: float = -999
    categorical_missing_fill: str = 'unknown'

    # Feature engineering thresholds
    discretization_bins: int = 5
    constant_feature_threshold: float = 0.98
    correlation_threshold: float = 0.95

    # Age binning (for demographic features)
    age_bins: list = field(default_factory=lambda: [0, 25, 45, 65, 100])

    # Credit utilization thresholds
    credit_util_high_threshold: float = 50.0
    credit_util_bins: int = 10

    # Risk feature engineering
    income_credit_ratio_threshold: float = 2.0
    payment_ratio_threshold: float = 0.3

    # Temporal features
    temporal_lookback_months: int = 12
    seasonal_features: bool = True

    # Numerical stability
    epsilon_division_safety: float = 1e-8

    # Feature interaction limits
    max_temporal_interactions: int = 5


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Cross-validation and training
    cv_folds: int = 5
    max_updates: int = 50
    patience: int = 5
    optimize_hyperparameters: bool = True
    n_hp_trials: int = 30
    early_stopping: bool = True
    save_checkpoints: bool = True
    use_sample_weights: bool = False

    # Hyperparameter optimization ranges
    stability_alpha_min: float = 0.01
    stability_alpha_max: float = 0.3
    calibration_window_min: int = 500
    calibration_window_max: int = 2000
    min_weight_min: float = 0.01
    min_weight_max: float = 0.2

    # Training monitoring
    log_training_progress: bool = True
    validation_frequency: int = 1
    early_stopping_metric: str = 'auc_roc'
    early_stopping_tolerance: float = 1e-4


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Calibration and statistical testing
    calibration_bins: int = 10
    significance_level: float = 0.05
    drift_window_size: int = 1000
    temporal_analysis: bool = True
    business_metrics: bool = True
    save_predictions: bool = True

    # Business metrics configuration
    default_loss: float = 1.0
    profit_margin: float = 0.15
    profit_threshold_min: float = 0.1
    profit_threshold_max: float = 0.9
    profit_threshold_steps: int = 9

    # Drift detection thresholds
    js_divergence_threshold: float = 0.1
    psi_threshold: float = 0.25
    kolmogorov_smirnov_threshold: float = 0.05

    # Logging and reporting
    log_interval_batches: int = 10
    detailed_metrics: bool = True
    save_distribution_plots: bool = False


@dataclass
class DataConfig:
    """Configuration for data handling."""

    data_path: Optional[str] = None
    home_credit_samples: int = 50000
    lending_club_samples: int = 30000
    train_months: int = 12
    val_months: int = 4
    test_months: int = 8
    temporal_column: str = 'APPLICATION_MONTH'
    target_column: str = 'TARGET'
    sample_weights_column: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    experiment_name: str = 'temporal_credit_degradation'
    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    log_artifacts: bool = True
    log_model: bool = True


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # General settings
    random_state: int = 42
    log_level: str = 'INFO'
    output_dir: str = 'outputs'
    model_dir: str = 'models'

    # Logging configuration
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format: str = '%Y-%m-%d %H:%M:%S'
    enable_debug_logging: bool = False
    log_performance_metrics: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure output directories exist
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.model_dir).mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Validate configuration
        self._validate_config()

    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration with file and console handlers.

        Raises:
            OSError: If log file cannot be created
        """
        try:
            log_level = getattr(logging, self.log_level.upper(), logging.INFO)

            # Clear existing handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Create formatters
            detailed_formatter = logging.Formatter(
                self.log_format,
                datefmt=self.log_date_format
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(detailed_formatter)

            # Configure root logger
            root_logger.setLevel(log_level)
            root_logger.addHandler(console_handler)

            # File handler (optional)
            if self.log_to_file:
                try:
                    log_dir = Path(self.output_dir) / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)

                    log_file = self.log_file_path or str(log_dir / "temporal_credit_detector.log")

                    file_handler = logging.FileHandler(log_file)
                    file_handler.setLevel(logging.DEBUG if self.enable_debug_logging else log_level)
                    file_handler.setFormatter(detailed_formatter)
                    root_logger.addHandler(file_handler)

                    logger.info(f"Logging to file: {log_file}")

                except Exception as e:
                    logger.warning(f"Could not setup file logging: {e}. Using console only.")

            # Debug logging for development
            if self.enable_debug_logging:
                root_logger.setLevel(logging.DEBUG)
                logger.debug("Debug logging enabled")

            # Performance logging
            if self.log_performance_metrics:
                perf_logger = logging.getLogger('performance')
                perf_logger.setLevel(logging.INFO)
                logger.debug("Performance logging enabled")

            # Module-specific logging levels
            logging.getLogger('optuna').setLevel(logging.WARNING)  # Reduce optuna verbosity
            logging.getLogger('lightgbm').setLevel(logging.WARNING)  # Reduce LightGBM verbosity
            logging.getLogger('catboost').setLevel(logging.ERROR)  # Reduce CatBoost verbosity

            logger.info(f"Logging configured at {self.log_level} level")

        except Exception as e:
            # Fallback to basic configuration if anything goes wrong
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.error(f"Error setting up advanced logging, using basic config: {e}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        if not (0 < self.model.stability_alpha < 1):
            raise ValueError("stability_alpha must be between 0 and 1")

        if not (0 < self.model.min_weight < 1):
            raise ValueError("min_weight must be between 0 and 1")

        if self.model.calibration_window < 100:
            raise ValueError("calibration_window must be at least 100")

        # Training validation
        if self.training.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

        if self.training.patience < 1:
            raise ValueError("patience must be at least 1")

        # Data validation
        if self.data.train_months < 1:
            raise ValueError("train_months must be at least 1")

        if self.data.val_months < 1:
            raise ValueError("val_months must be at least 1")

        if self.data.test_months < 1:
            raise ValueError("test_months must be at least 1")

        logger.info("Configuration validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}

        # Convert dataclass fields to dict
        for field_name in ['model', 'preprocessing', 'training', 'evaluation', 'data', 'experiment']:
            field_value = getattr(self, field_name)
            if hasattr(field_value, '__dict__'):
                config_dict[field_name] = field_value.__dict__
            else:
                config_dict[field_name] = field_value

        # Add general settings
        config_dict.update({
            'random_state': self.random_state,
            'log_level': self.log_level,
            'output_dir': self.output_dir,
            'model_dir': self.model_dir
        })

        return config_dict

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Path to save configuration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        # Extract sub-configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))

        # Extract general settings
        general_settings = {
            'random_state': config_dict.get('random_state', 42),
            'log_level': config_dict.get('log_level', 'INFO'),
            'output_dir': config_dict.get('output_dir', 'outputs'),
            'model_dir': config_dict.get('model_dir', 'models')
        }

        return cls(
            model=model_config,
            preprocessing=preprocessing_config,
            training=training_config,
            evaluation=evaluation_config,
            data=data_config,
            experiment=experiment_config,
            **general_settings
        )

    def update_from_dict(self, update_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary.

        Args:
            update_dict: Dictionary with updates
        """
        for section, updates in update_dict.items():
            if hasattr(self, section) and isinstance(updates, dict):
                section_obj = getattr(self, section)
                if hasattr(section_obj, '__dict__'):
                    for key, value in updates.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                        else:
                            logger.warning(f"Unknown parameter {key} in section {section}")
            elif hasattr(self, section):
                setattr(self, section, updates)
            else:
                logger.warning(f"Unknown configuration section: {section}")

        # Re-validate after updates
        self._validate_config()
        logger.info("Configuration updated successfully")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """Load configuration from file with optional overrides.

    Args:
        config_path: Path to configuration file. If None, uses default config.
        overrides: Dictionary of configuration overrides

    Returns:
        Configuration instance
    """
    # Start with default configuration
    config_dict = {}

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}. Using defaults.")

    # Apply overrides
    if overrides:
        _deep_update(config_dict, overrides)
        logger.info("Configuration overrides applied")

    # Create configuration object
    config = Config.from_dict(config_dict)

    return config


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Returns:
        Dictionary of configuration overrides from environment
    """
    env_overrides = {}

    # Model parameters
    if 'MODEL_STABILITY_ALPHA' in os.environ:
        env_overrides.setdefault('model', {})['stability_alpha'] = float(os.environ['MODEL_STABILITY_ALPHA'])

    if 'MODEL_MIN_WEIGHT' in os.environ:
        env_overrides.setdefault('model', {})['min_weight'] = float(os.environ['MODEL_MIN_WEIGHT'])

    # Training parameters
    if 'TRAINING_CV_FOLDS' in os.environ:
        env_overrides.setdefault('training', {})['cv_folds'] = int(os.environ['TRAINING_CV_FOLDS'])

    if 'TRAINING_HP_TRIALS' in os.environ:
        env_overrides.setdefault('training', {})['n_hp_trials'] = int(os.environ['TRAINING_HP_TRIALS'])

    # Data parameters
    if 'DATA_PATH' in os.environ:
        env_overrides.setdefault('data', {})['data_path'] = os.environ['DATA_PATH']

    # Experiment parameters
    if 'EXPERIMENT_NAME' in os.environ:
        env_overrides.setdefault('experiment', {})['experiment_name'] = os.environ['EXPERIMENT_NAME']

    if 'MLFLOW_TRACKING_URI' in os.environ:
        env_overrides.setdefault('experiment', {})['mlflow_tracking_uri'] = os.environ['MLFLOW_TRACKING_URI']

    # General parameters
    if 'RANDOM_STATE' in os.environ:
        env_overrides['random_state'] = int(os.environ['RANDOM_STATE'])

    if 'LOG_LEVEL' in os.environ:
        env_overrides['log_level'] = os.environ['LOG_LEVEL']

    if env_overrides:
        logger.info(f"Loaded {len(env_overrides)} configuration overrides from environment")

    return env_overrides


def create_config_template(output_path: Union[str, Path]) -> None:
    """Create a configuration template file.

    Args:
        output_path: Path to save the template
    """
    # Create default configuration
    config = Config()

    # Add comments to the template
    template_dict = config.to_dict()

    # Save template
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Temporal Credit Degradation Detector Configuration\n")
        f.write("# This is a template file showing all available configuration options\n\n")
        yaml.dump(template_dict, f, default_flow_style=False, indent=2)

    logger.info(f"Configuration template saved to {output_path}")


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Deep update of nested dictionaries.

    Args:
        base_dict: Base dictionary to update
        update_dict: Updates to apply
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value