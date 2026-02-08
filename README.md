# Temporal Credit Degradation Detector

A production-grade system that detects concept drift in credit risk models by analyzing how feature importance and prediction reliability degrade over time across different economic regimes. Unlike standard drift detection, this implements a novel **stability-weighted ensemble** that automatically reweights base models based on their recent calibration quality, enabling graceful model degradation rather than catastrophic failure during economic shifts.

## Key Innovation

The **Stability-Weighted Ensemble** continuously monitors model calibration and adapts weights dynamically, providing:
- Graceful degradation during economic regime changes
- Real-time drift detection and adaptation
- Superior performance stability across temporal periods
- Business-aware metrics for credit risk applications

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/A-SHOJAEI/temporal-credit-degradation-detector.git
cd temporal-credit-degradation-detector

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from temporal_credit_degradation_detector import StabilityWeightedEnsemble, DataLoader, ModelTrainer

# Load data
loader = DataLoader()
X, y = loader.load_home_credit_data(sample_size=10000)
splits = loader.create_temporal_splits(X, y)

# Train model
trainer = ModelTrainer({'experiment_name': 'credit_model'})
results = trainer.train_cross_validation(
    splits['train'][0], splits['train'][1],
    cv_folds=5, optimize_hp=True
)

# Evaluate
test_results = trainer.evaluate_on_test(
    splits['test'][0], splits['test'][1]
)

print(f"AUC-ROC: {test_results['metrics']['auc_roc']:.4f}")
print(f"Brier Score: {test_results['metrics']['brier_score']:.4f}")
```

### Command Line Interface

```bash
# Train model
python scripts/train.py --config configs/default.yaml --cv-folds 5 --optimize-hp

# Evaluate model
python scripts/evaluate.py \
    --model-path models/best_model_cv.pkl \
    --preprocessor-path models/preprocessor.pkl \
    --evaluation-type comprehensive
```

## Architecture

The system consists of four main components:

1. **Stability-Weighted Ensemble**: Novel ensemble method combining LightGBM and CatBoost with adaptive weight management based on recent calibration quality
2. **Temporal Preprocessor**: Creates time-aware features for drift detection across economic regimes
3. **Drift Monitor**: Real-time detection of feature and prediction drift using statistical tests (KS, JS divergence)
4. **Business Evaluator**: Credit-specific metrics including calibration analysis and business-aware thresholds

## Results

Evaluated on synthetic Home Credit data (10,000 samples) with temporal drift simulation using a stability-weighted LightGBM + CatBoost ensemble:

| Metric | Value |
|--------|-------|
| AUC-ROC (Test) | 0.8396 |
| AUC-ROC (CV Mean) | 0.6813 |
| Brier Score | 0.2129 |
| Calibration Error | 0.1974 |
| Precision | 0.8876 |
| Recall | 0.3628 |

**Model**: Stability-weighted LightGBM + CatBoost ensemble with 5-fold temporal cross-validation and hyperparameter optimization.

**Dataset**: Synthetic Home Credit data (10K samples) with simulated temporal drift across economic regimes.

## Technical Highlights

- **Adaptive Weight Management**: Models are continuously reweighted based on calibration quality
- **Comprehensive Drift Detection**: Statistical tests (KS, JS divergence) detect feature and prediction drift
- **Temporal Feature Engineering**: Time-aware features capture economic regime changes
- **Production-Ready Pipeline**: MLflow tracking, automated testing, configuration management

## Project Structure

```
temporal-credit-degradation-detector/
├── src/temporal_credit_degradation_detector/
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Stability-weighted ensemble implementation
│   ├── training/             # Training pipeline with MLflow
│   ├── evaluation/           # Comprehensive evaluation metrics
│   └── utils/                # Configuration and utilities
├── tests/                    # Comprehensive test suite
├── scripts/                  # Training and evaluation scripts
├── notebooks/                # Exploration and analysis notebooks
└── configs/                  # Configuration files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
