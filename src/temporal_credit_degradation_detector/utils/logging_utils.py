"""Advanced logging utilities for temporal credit degradation detection.

This module provides performance monitoring, structured logging, and debugging utilities
that enhance the observability of the machine learning pipeline.
"""

import logging
import time
import functools
import psutil
import os
from typing import Any, Callable, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger('performance')


class PerformanceMonitor:
    """Monitor performance metrics for ML operations."""

    def __init__(self, log_memory: bool = True, log_time: bool = True):
        """Initialize performance monitor.

        Args:
            log_memory: Whether to log memory usage
            log_time: Whether to log execution time
        """
        self.log_memory = log_memory
        self.log_time = log_time
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB.

        Returns:
            Dictionary with memory usage metrics
        """
        try:
            memory_info = self.process.memory_info()
            return {
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': self.process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {}


def log_performance(
    func_name: Optional[str] = None,
    log_args: bool = False,
    log_memory: bool = True,
    log_time: bool = True,
    memory_threshold_mb: float = 100.0
) -> Callable:
    """Decorator to log function performance metrics.

    Args:
        func_name: Custom name for logging (uses function name if None)
        log_args: Whether to log function arguments
        log_memory: Whether to log memory usage
        log_time: Whether to log execution time
        memory_threshold_mb: Log warning if memory usage exceeds this threshold

    Returns:
        Decorated function

    Example:
        >>> @log_performance(log_memory=True, memory_threshold_mb=500.0)
        ... def train_model(X, y):
        ...     # Training code here
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"
            monitor = PerformanceMonitor(log_memory=log_memory, log_time=log_time)

            # Log function start
            start_memory = monitor.get_memory_usage() if log_memory else {}
            start_time = time.time() if log_time else None

            if log_args and (args or kwargs):
                arg_info = f" with args: {len(args)} positional, {len(kwargs)} keyword"
            else:
                arg_info = ""

            perf_logger.info(f"Starting {name}{arg_info}")

            if start_memory:
                perf_logger.debug(f"{name} start memory: {start_memory['memory_rss_mb']:.1f} MB "
                                f"({start_memory['memory_percent']:.1f}%)")

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log completion
                end_time = time.time() if log_time else None
                end_memory = monitor.get_memory_usage() if log_memory else {}

                # Calculate metrics
                if log_time and start_time is not None:
                    duration = end_time - start_time
                    perf_logger.info(f"{name} completed in {duration:.2f} seconds")

                if end_memory:
                    memory_mb = end_memory['memory_rss_mb']
                    perf_logger.debug(f"{name} end memory: {memory_mb:.1f} MB")

                    # Memory warning
                    if memory_mb > memory_threshold_mb:
                        perf_logger.warning(f"{name} high memory usage: {memory_mb:.1f} MB "
                                          f"(threshold: {memory_threshold_mb:.1f} MB)")

                    # Memory delta
                    if start_memory:
                        memory_delta = memory_mb - start_memory['memory_rss_mb']
                        if abs(memory_delta) > 10:  # Only log if significant change
                            perf_logger.debug(f"{name} memory delta: {memory_delta:+.1f} MB")

                return result

            except Exception as e:
                perf_logger.error(f"{name} failed after {time.time() - start_time:.2f}s: {e}")
                raise

        return wrapper
    return decorator


class StructuredLogger:
    """Structured logging for ML experiments."""

    def __init__(self, logger_name: str):
        """Initialize structured logger.

        Args:
            logger_name: Name of the logger
        """
        self.logger = logging.getLogger(logger_name)

    def log_data_info(
        self,
        X_shape: tuple,
        y_distribution: Dict[Any, int],
        missing_data: Dict[str, float] = None,
        data_source: str = "unknown"
    ) -> None:
        """Log structured data information.

        Args:
            X_shape: Shape of feature matrix
            y_distribution: Distribution of target classes
            missing_data: Missing data percentages by column
            data_source: Source of the data
        """
        self.logger.info(f"Data loaded from {data_source}: {X_shape[0]} samples, {X_shape[1]} features")
        self.logger.info(f"Target distribution: {y_distribution}")

        if missing_data:
            high_missing = {col: pct for col, pct in missing_data.items() if pct > 0.1}
            if high_missing:
                self.logger.warning(f"Columns with >10% missing data: {high_missing}")

    def log_model_config(self, config: Dict[str, Any], model_type: str) -> None:
        """Log model configuration.

        Args:
            config: Model configuration dictionary
            model_type: Type of model being configured
        """
        self.logger.info(f"{model_type} configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_training_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best_score: float,
        patience_counter: int = None
    ) -> None:
        """Log training progress.

        Args:
            epoch: Current epoch/iteration
            metrics: Current metrics
            best_score: Best score so far
            patience_counter: Early stopping patience counter
        """
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metric_str} (best: {best_score:.4f})")

        if patience_counter is not None:
            self.logger.debug(f"Early stopping patience: {patience_counter}")

    def log_evaluation_results(
        self,
        metrics: Dict[str, float],
        dataset_name: str = "test",
        model_name: str = "model"
    ) -> None:
        """Log evaluation results.

        Args:
            metrics: Evaluation metrics
            dataset_name: Name of the evaluated dataset
            model_name: Name of the model
        """
        self.logger.info(f"{model_name} evaluation on {dataset_name}:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        # Highlight key metrics
        key_metrics = ['auc_roc', 'f1_score', 'precision', 'recall']
        key_values = {k: v for k, v in metrics.items() if k in key_metrics}
        if key_values:
            summary = ", ".join([f"{k}: {v:.3f}" for k, v in key_values.items()])
            self.logger.info(f"Key metrics - {summary}")


def setup_debug_logging(output_dir: str = "debug_logs") -> None:
    """Setup comprehensive debug logging for troubleshooting.

    Args:
        output_dir: Directory for debug log files
    """
    debug_dir = Path(output_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create detailed debug formatter
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )

    # Debug file handler
    debug_file = debug_dir / "debug.log"
    debug_handler = logging.FileHandler(debug_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(debug_formatter)

    # Error file handler
    error_file = debug_dir / "errors.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(debug_formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(error_handler)
    root_logger.setLevel(logging.DEBUG)

    logger.info(f"Debug logging enabled. Files: {debug_file}, {error_file}")


def log_system_info() -> None:
    """Log system information for reproducibility."""
    try:
        import sys
        import platform
        import numpy as np
        import pandas as pd
        import sklearn

        system_logger = logging.getLogger('system')

        system_logger.info("System Information:")
        system_logger.info(f"  Platform: {platform.platform()}")
        system_logger.info(f"  Python: {sys.version}")
        system_logger.info(f"  NumPy: {np.__version__}")
        system_logger.info(f"  Pandas: {pd.__version__}")
        system_logger.info(f"  Scikit-learn: {sklearn.__version__}")

        # Memory info
        memory = psutil.virtual_memory()
        system_logger.info(f"  Total RAM: {memory.total / 1024**3:.1f} GB")
        system_logger.info(f"  Available RAM: {memory.available / 1024**3:.1f} GB")

        # CPU info
        system_logger.info(f"  CPU cores: {psutil.cpu_count()}")

    except Exception as e:
        logger.warning(f"Could not log system info: {e}")