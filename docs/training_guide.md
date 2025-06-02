# Model Training Guide

This guide explains how to use the enhanced training pipeline with improved features, train/test splitting, and model selection.

## Overview of Improvements

### 1. Enhanced Feature Engineering
- **Technical Indicators**: Advanced indicators like Bollinger Bands, MACD, Stochastic Oscillator, ADX, CCI
- **Volume Indicators**: OBV, Chaikin A/D, Money Flow Index, Volume Rate of Change
- **Volatility Features**: Multiple volatility measures (Parkinson, Garman-Klass, Historical)
- **Pattern Recognition**: Candlestick pattern detection (Doji, Hammer, Engulfing, etc.)
- **Market Microstructure**: Bid-ask spread estimates, illiquidity measures, price impact
- **Regime Indicators**: Bull/bear market detection, market drawdown features
- **Cross-sectional Features**: Relative strength vs market/sector

### 2. Robust Train/Test Splitting
- **Temporal Gap**: Prevents lookahead bias with configurable gap days
- **Event-Aware Splitting**: Avoids splitting during major market events
- **Walk-Forward Analysis**: Multiple train/test splits for robust evaluation
- **Expanding Window**: Growing training set over time
- **Purged Cross-Validation**: Time series aware CV with lookahead prevention

### 3. Advanced Model Selection
- **Multiple Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, SVM, etc.
- **Hyperparameter Optimization**: Grid search, random search, or Optuna
- **Ensemble Methods**: Voting classifiers combining multiple models
- **Probability Calibration**: Isotonic/sigmoid calibration for better probabilities
- **Feature Selection**: Statistical methods to select most predictive features

## Running the Training Pipeline

### Basic Training
```bash
# Basic training with default settings
python train_models.py

# This uses:
# - Event-aware train/test split
# - XGBoost, LightGBM, and Random Forest
# - Random hyperparameter search
```

### Enhanced Features
```bash
# Use enhanced feature set (100+ features)
python train_models.py --use_enhanced_features

# Note: This requires TA-Lib installation
# On Mac: brew install ta-lib
# Then: pip install ta-lib
```

### Different Split Methods
```bash
# Simple temporal split
python train_models.py --split_method temporal --test_size 0.2

# Walk-forward analysis (5 splits)
python train_models.py --split_method walk_forward

# Expanding window
python train_models.py --split_method expanding_window
```

### Model Selection
```bash
# Train specific models
python train_models.py --models xgboost lightgbm catboost

# Train ensemble
python train_models.py --models xgboost lightgbm random_forest ensemble

# Train all available models
python train_models.py --models xgboost lightgbm catboost random_forest gradient_boosting extra_trees logistic_regression svm
```

### Hyperparameter Optimization
```bash
# Grid search (slower but thorough)
python train_models.py --optimization grid

# Random search with 100 trials
python train_models.py --optimization random --n_trials 100

# Optuna optimization (Bayesian)
python train_models.py --optimization optuna --n_trials 50

# No optimization (use defaults)
python train_models.py --optimization none
```

### Feature Selection
```bash
# Select top 50 features using F-statistic
python train_models.py --feature_selection f_classif --n_features 50

# Select features using mutual information
python train_models.py --feature_selection mutual_info --n_features 30
```

### Advanced Configuration
```bash
# Full example with all options
python train_models.py \
    --use_enhanced_features \
    --split_method walk_forward \
    --models xgboost lightgbm catboost \
    --optimization optuna \
    --n_trials 100 \
    --gap_days 5 \
    --feature_selection mutual_info \
    --n_features 40
```

## Running Feature Analysis

To understand which features are most important:

```bash
# Run feature engineering EDA
cd notebooks
python feature_engineering_eda.py
```

This will generate:
- Feature distribution plots
- Correlation analysis
- Feature importance rankings
- Predictive power analysis
- Reports in `reports/` directory

## Output Files

After training, you'll find:

```
models/
├── xgboost_20240115_143022.pkl      # Trained models
├── lightgbm_20240115_143022.pkl
├── best_model_20240115_143022.pkl   # Best performing model
├── preprocessing_20240115_143022.pkl # Scalers and feature selectors
├── results_20240115_143022.pkl      # Training results
├── model_comparison.png             # Performance visualization
├── training_report.md               # Detailed report
└── final_training_report.md         # Summary across all splits

reports/
├── feature_distributions.png
├── correlation_matrix.png
├── feature_importance_comparison.png
├── feature_predictive_power.png
└── feature_engineering_report.md

data/
└── feature_recommendations.json     # Recommended features
```

## Interpreting Results

### Model Performance Metrics
- **ROC AUC**: Primary metric (0.5 = random, 1.0 = perfect)
- **Accuracy**: Overall correct predictions
- **Precision**: When predicting outperformance, how often correct
- **Recall**: Of actual outperformers, how many identified
- **F1 Score**: Harmonic mean of precision and recall

### Feature Importance
- **Random Forest Importance**: Based on information gain
- **Mutual Information**: Non-linear relationships
- **F-statistic**: Linear relationships
- **Consensus Features**: Appear in top rankings across methods

### Train/Test Split Validation
- Check for lookahead bias warnings
- Verify gap days between train and test
- Note any major events in test period

## Best Practices

1. **Start Simple**: Begin with basic features and temporal split
2. **Add Complexity**: Gradually add enhanced features and advanced splits
3. **Monitor Overfitting**: Compare train vs test performance
4. **Use Multiple Splits**: Walk-forward or expanding window for robustness
5. **Feature Selection**: Too many features can hurt performance
6. **Ensemble Methods**: Often outperform individual models

## Troubleshooting

### TA-Lib Installation Issues
```bash
# Mac
brew install ta-lib
pip install ta-lib

# Linux
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib

# Windows
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

### Memory Issues
- Use feature selection to reduce dimensions
- Train on subset of models
- Use smaller hyperparameter search space

### Performance Issues
- Use `--optimization none` for faster training
- Reduce `--n_trials` for optimization
- Use fewer models in ensemble

## Next Steps

1. **Production Deployment**: Use the API to serve predictions
2. **Model Monitoring**: Track performance over time
3. **Feature Engineering**: Add domain-specific features
4. **Alternative Data**: Incorporate news, sentiment, or options data
5. **Deep Learning**: Experiment with LSTM/Transformer models 