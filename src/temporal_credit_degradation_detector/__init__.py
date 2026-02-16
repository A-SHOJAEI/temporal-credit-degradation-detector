"""Temporal Credit Degradation Detector

A production-grade system that detects concept drift in credit risk models
by analyzing how feature importance and prediction reliability degrade over time.
"""

__version__ = "0.1.0"
__author__ = "AI Research Team"

from temporal_credit_degradation_detector.models.model import StabilityWeightedEnsemble
from temporal_credit_degradation_detector.training.trainer import ModelTrainer
from temporal_credit_degradation_detector.evaluation.metrics import ModelEvaluator

__all__ = [
    "StabilityWeightedEnsemble",
    "ModelTrainer",
    "ModelEvaluator",
]