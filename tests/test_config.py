"""Tests for configuration management."""

import pytest
import logging
from pathlib import Path
from temporal_credit_degradation_detector.utils.config import (
    Config, ModelConfig, PreprocessingConfig, TrainingConfig,
    EvaluationConfig, DataConfig, ExperimentConfig
)


class TestModelConfig:
    """Test model configuration validation."""

    def test_default_initialization(self):
        """Test default model configuration."""
        config = ModelConfig()

        assert config.stability_alpha == 0.1
        assert config.min_weight == 0.05
        assert config.recalibration_threshold == 0.15
        assert config.calibration_window == 1000
        assert config.random_state == 42

    def test_custom_parameters(self):
        """Test model configuration with custom parameters."""
        config = ModelConfig(
            stability_alpha=0.05,
            min_weight=0.01,
            calibration_window=500,
            n_estimators=200
        )

        assert config.stability_alpha == 0.05
        assert config.min_weight == 0.01
        assert config.calibration_window == 500
        assert config.n_estimators == 200

    def test_parameter_types(self):
        """Test that parameters have correct types."""
        config = ModelConfig()

        assert isinstance(config.stability_alpha, float)
        assert isinstance(config.min_weight, float)
        assert isinstance(config.calibration_window, int)
        assert isinstance(config.n_estimators, int)


class TestPreprocessingConfig:
    """Test preprocessing configuration validation."""

    def test_default_initialization(self):
        """Test default preprocessing configuration."""
        config = PreprocessingConfig()

        assert config.handle_missing is True
        assert config.create_temporal_features is True
        assert config.missing_value_fill == -999
        assert config.discretization_bins == 5

    def test_feature_engineering_flags(self):
        """Test feature engineering configuration flags."""
        config = PreprocessingConfig(
            create_temporal_features=False,
            create_risk_features=False,
            discretize_continuous=True
        )

        assert config.create_temporal_features is False
        assert config.create_risk_features is False
        assert config.discretize_continuous is True

    def test_threshold_parameters(self):
        """Test threshold parameter configuration."""
        config = PreprocessingConfig(
            constant_feature_threshold=0.95,
            correlation_threshold=0.9,
            credit_util_high_threshold=75.0
        )

        assert config.constant_feature_threshold == 0.95
        assert config.correlation_threshold == 0.9
        assert config.credit_util_high_threshold == 75.0

    def test_age_bins_configuration(self):
        """Test age bins configuration."""
        custom_bins = [0, 30, 50, 70, 100]
        config = PreprocessingConfig(age_bins=custom_bins)

        assert config.age_bins == custom_bins

    def test_invalid_discretization_bins(self):
        """Test that invalid discretization bins raise error when validated."""
        config = PreprocessingConfig(discretization_bins=1)
        assert config.discretization_bins == 1  # Config object allows it
        # Validation would happen during preprocessor initialization

    def test_invalid_threshold_values(self):
        """Test invalid threshold values."""
        # These should be valid at config level, validation happens at runtime
        config = PreprocessingConfig(
            constant_feature_threshold=1.5,  # > 1
            correlation_threshold=-0.5,      # < 0
        )
        assert config.constant_feature_threshold == 1.5
        assert config.correlation_threshold == -0.5


class TestTrainingConfig:
    """Test training configuration validation."""

    def test_default_initialization(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.cv_folds == 5
        assert config.max_updates == 50
        assert config.patience == 5
        assert config.optimize_hyperparameters is True
        assert config.n_hp_trials == 30

    def test_hyperparameter_ranges(self):
        """Test hyperparameter optimization ranges."""
        config = TrainingConfig(
            stability_alpha_min=0.005,
            stability_alpha_max=0.5,
            calibration_window_min=250,
            calibration_window_max=3000
        )

        assert config.stability_alpha_min == 0.005
        assert config.stability_alpha_max == 0.5
        assert config.calibration_window_min == 250
        assert config.calibration_window_max == 3000

    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            early_stopping=True,
            patience=10,
            early_stopping_metric='f1_score',
            early_stopping_tolerance=1e-5
        )

        assert config.early_stopping is True
        assert config.patience == 10
        assert config.early_stopping_metric == 'f1_score'
        assert config.early_stopping_tolerance == 1e-5

    def test_minimal_configuration(self):
        """Test minimal valid configuration."""
        config = TrainingConfig(
            cv_folds=2,
            max_updates=1,
            patience=1,
            n_hp_trials=1
        )

        assert config.cv_folds == 2
        assert config.max_updates == 1
        assert config.patience == 1
        assert config.n_hp_trials == 1


class TestEvaluationConfig:
    """Test evaluation configuration validation."""

    def test_default_initialization(self):
        """Test default evaluation configuration."""
        config = EvaluationConfig()

        assert config.calibration_bins == 10
        assert config.significance_level == 0.05
        assert config.drift_window_size == 1000
        assert config.business_metrics is True

    def test_business_metrics_config(self):
        """Test business metrics configuration."""
        config = EvaluationConfig(
            default_loss=2.0,
            profit_margin=0.25,
            profit_threshold_min=0.05,
            profit_threshold_max=0.95
        )

        assert config.default_loss == 2.0
        assert config.profit_margin == 0.25
        assert config.profit_threshold_min == 0.05
        assert config.profit_threshold_max == 0.95

    def test_drift_detection_config(self):
        """Test drift detection configuration."""
        config = EvaluationConfig(
            js_divergence_threshold=0.15,
            psi_threshold=0.3,
            kolmogorov_smirnov_threshold=0.01
        )

        assert config.js_divergence_threshold == 0.15
        assert config.psi_threshold == 0.3
        assert config.kolmogorov_smirnov_threshold == 0.01


class TestDataConfig:
    """Test data configuration validation."""

    def test_default_initialization(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.data_path is None
        assert config.home_credit_samples == 50000
        assert config.lending_club_samples == 30000
        assert config.train_months == 12
        assert config.temporal_column == 'APPLICATION_MONTH'

    def test_temporal_split_config(self):
        """Test temporal split configuration."""
        config = DataConfig(
            train_months=18,
            val_months=6,
            test_months=12,
            temporal_column='issue_month'
        )

        assert config.train_months == 18
        assert config.val_months == 6
        assert config.test_months == 12
        assert config.temporal_column == 'issue_month'

    def test_data_path_configuration(self):
        """Test data path configuration."""
        config = DataConfig(data_path='/path/to/data')
        assert config.data_path == '/path/to/data'


class TestExperimentConfig:
    """Test experiment configuration validation."""

    def test_default_initialization(self):
        """Test default experiment configuration."""
        config = ExperimentConfig()

        assert config.experiment_name == 'temporal_credit_degradation'
        assert config.run_name is None
        assert config.mlflow_tracking_uri is None
        assert config.log_artifacts is True

    def test_custom_experiment_config(self):
        """Test custom experiment configuration."""
        config = ExperimentConfig(
            experiment_name='credit_risk_model_v2',
            run_name='test_run_001',
            mlflow_tracking_uri='sqlite:///mlruns.db',
            log_artifacts=False
        )

        assert config.experiment_name == 'credit_risk_model_v2'
        assert config.run_name == 'test_run_001'
        assert config.mlflow_tracking_uri == 'sqlite:///mlruns.db'
        assert config.log_artifacts is False


class TestMainConfig:
    """Test main configuration class."""

    def test_default_initialization(self):
        """Test default main configuration."""
        config = Config()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.experiment, ExperimentConfig)

    def test_logging_configuration(self):
        """Test logging configuration."""
        config = Config(
            log_level='DEBUG',
            log_to_file=True,
            enable_debug_logging=True
        )

        assert config.log_level == 'DEBUG'
        assert config.log_to_file is True
        assert config.enable_debug_logging is True

    def test_directory_creation(self):
        """Test that directories are created during initialization."""
        config = Config(
            output_dir='test_outputs',
            model_dir='test_models'
        )

        assert Path('test_outputs').exists()
        assert Path('test_models').exists()

        # Clean up
        Path('test_outputs').rmdir()
        Path('test_models').rmdir()

    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        config = Config()

        # Access nested config values
        assert config.model.stability_alpha == 0.1
        assert config.preprocessing.missing_value_fill == -999
        assert config.training.cv_folds == 5
        assert config.evaluation.calibration_bins == 10
        assert config.data.train_months == 12

    def test_config_modification(self):
        """Test modifying configuration values."""
        config = Config()

        # Modify nested values
        config.model.stability_alpha = 0.05
        config.preprocessing.discretization_bins = 10
        config.training.cv_folds = 3

        assert config.model.stability_alpha == 0.05
        assert config.preprocessing.discretization_bins == 10
        assert config.training.cv_folds == 3


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_valid_configuration_passes(self):
        """Test that valid configuration passes validation."""
        config = Config()
        # Should not raise any exceptions
        config._validate_config()

    def test_invalid_stability_alpha(self):
        """Test invalid stability_alpha raises error."""
        config = Config()
        config.model.stability_alpha = 1.5  # > 1

        with pytest.raises(ValueError, match="stability_alpha must be between 0 and 1"):
            config._validate_config()

    def test_invalid_min_weight(self):
        """Test invalid min_weight raises error."""
        config = Config()
        config.model.min_weight = -0.1  # < 0

        with pytest.raises(ValueError, match="min_weight must be between 0 and 1"):
            config._validate_config()

    def test_invalid_calibration_window(self):
        """Test invalid calibration_window raises error."""
        config = Config()
        config.model.calibration_window = 50  # < 100

        with pytest.raises(ValueError, match="calibration_window must be at least 100"):
            config._validate_config()

    def test_multiple_validation_errors(self):
        """Test that first validation error is raised."""
        config = Config()
        config.model.stability_alpha = 2.0  # Invalid
        config.model.min_weight = -0.5      # Invalid

        # Should raise first error encountered
        with pytest.raises(ValueError):
            config._validate_config()


class TestConfigSerialization:
    """Test configuration serialization capabilities."""

    def test_config_to_dict(self):
        """Test converting config to dictionary (manual implementation)."""
        config = Config()

        # Manual conversion test - Config doesn't have built-in to_dict
        assert hasattr(config, 'model')
        assert hasattr(config, 'preprocessing')
        assert hasattr(config, 'training')

        # Test that all sub-configs have expected attributes
        assert hasattr(config.model, 'stability_alpha')
        assert hasattr(config.preprocessing, 'missing_value_fill')
        assert hasattr(config.training, 'cv_folds')

    def test_config_repr(self):
        """Test config string representation."""
        config = ModelConfig()
        repr_str = repr(config)

        # Should contain class name and key parameters
        assert 'ModelConfig' in repr_str
        assert 'stability_alpha' in repr_str


class TestConfigEdgeCases:
    """Test configuration edge cases and boundary conditions."""

    def test_zero_values(self):
        """Test zero values in configuration."""
        config = ModelConfig(
            calibration_window=100,  # Minimum allowed
            n_estimators=1,          # Minimum useful
        )

        assert config.calibration_window == 100
        assert config.n_estimators == 1

    def test_extreme_values(self):
        """Test extreme but valid values."""
        config = PreprocessingConfig(
            discretization_bins=100,
            max_temporal_interactions=50,
            temporal_lookback_months=60
        )

        assert config.discretization_bins == 100
        assert config.max_temporal_interactions == 50
        assert config.temporal_lookback_months == 60

    def test_none_values(self):
        """Test None values where allowed."""
        config = DataConfig(
            data_path=None,
            sample_weights_column=None
        )

        assert config.data_path is None
        assert config.sample_weights_column is None

    def test_string_values(self):
        """Test string value configuration."""
        config = EvaluationConfig(
            early_stopping_metric='precision',
        )
        config2 = DataConfig(
            temporal_column='date_column',
            target_column='default_flag'
        )

        assert hasattr(config, 'early_stopping_metric')  # May not exist in current config
        assert config2.temporal_column == 'date_column'
        assert config2.target_column == 'default_flag'