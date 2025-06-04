# Feature Engineering Analysis Report
Date: 2025-06-02 23:01
Total features analyzed: 84
Total samples: 1802275

## 1. Top Features by Importance
### Random Forest Top 15:
- Beta_20d: 0.0188
- Volatility_50d: 0.0185
- Returns_5d_SP500: 0.0176
- Volatility_20d: 0.0175
- Volatility_10d: 0.0167
- Volume_MA_20: 0.0167
- Returns_1d_SP500: 0.0162
- Volatility_5d: 0.0160
- BB_Width_50d: 0.0158
- Volume_Ratio: 0.0158
- Returns_1d_lag3: 0.0157
- Returns_1d_lag2: 0.0155
- Returns_50d: 0.0154
- Log_Return_50d: 0.0154
- Returns_1d_lag1: 0.0153

### Consensus Features (appearing in all importance metrics):

## 2. Highly Correlated Features to Consider Removing
- Open and High (r = 1.000)
- Open and Low (r = 1.000)
- Open and Close (r = 1.000)
- Open and VWAP (r = 0.999)
- Open and Price_MA_5d (r = 1.000)
- Open and Price_MA_10d (r = 1.000)
- Open and Price_Std_10d (r = 0.819)
- Open and Price_MA_20d (r = 0.999)
- Open and Price_Std_20d (r = 0.838)
- Open and Price_MA_50d (r = 0.997)

## 3. Feature Engineering Recommendations
### Features to Add:
- **Market Microstructure**: Bid-ask spread, order flow imbalance
- **Cross-sectional**: Stock vs sector performance, relative strength
- **Alternative Data**: Sentiment scores, news volume
- **Macroeconomic**: Interest rates, VIX, dollar index
- **Options-based**: Implied volatility, put-call ratio

### Feature Transformations:
- Apply log transformation to skewed features
- Create interaction terms for top features
- Add polynomial features for non-linear relationships
- Implement regime-based features (bull/bear market indicators)

# Feature Engineering Report

## Overview
This report details the feature engineering process used in the stock prediction pipeline, referencing the code in `train_models.py` and related modules. The process includes both basic and enhanced feature engineering, with a focus on creating a robust and informative feature set for model training.

---

### 1. Basic Feature Engineering

**Module Referenced:** `src.data.feature_engineering.FeatureGenerator`

- **Process:**
  - The script loads raw stock and S&P 500 data using `StockDataLoader.prepare_training_data()`.
  - Basic features are generated using `FeatureGenerator.prepare_features()`, which likely includes:
    - Price-based features (open, close, high, low, volume)
    - Returns and log-returns
    - Rolling statistics (mean, std, min, max, etc.)
    - Technical indicators (moving averages, RSI, MACD, etc.)
  - These features are designed to capture short-term and long-term price trends, volatility, and momentum.

**Diagram:**
```
Raw Data (Stock, S&P 500)
      │
      ▼
Basic Feature Engineering (FeatureGenerator)
      │
      ▼
Feature DataFrame (price, returns, rolling stats, technical indicators)
```

---

### 2. Enhanced Feature Engineering

**Module Referenced:** `src.data.enhanced_features.EnhancedFeatureGenerator`

- **Process:**
  - If `--use_enhanced_features` is set, additional features are generated using `EnhancedFeatureGenerator.generate_all_features()`.
  - Enhanced features may include:
    - Cross-asset and macroeconomic features
    - Event-based features (earnings, splits, news)
    - Advanced technical indicators
    - Feature interactions and polynomial features
    - Lagged features and target encoding
  - Feature groups are printed for transparency.
  - Feature importance-based selection can be applied to reduce dimensionality (`--feature_importance_selection`).

**Diagram:**
```
Feature DataFrame (basic)
      │
      ▼
Enhanced Feature Engineering (EnhancedFeatureGenerator)
      │
      ▼
Enhanced Feature DataFrame (macro, event, advanced, interactions)
```

---

### 3. Feature Selection

- **Options:**
  - Statistical selection (`--feature_selection`): ANOVA F-test, mutual information
  - Advanced selection (`--advanced_feature_selection`): importance, correlation, RFE, SHAP
  - Feature importance selection can be used to keep only the most predictive features (`--max_features`).
- **Implementation:**
  - Selection is performed after feature generation, using the optimizer's `select_features()` method.
  - Only selected features (plus `Symbol` and `Label`) are kept for training.

---

### 4. Saving and Reproducibility

- **Feature cache:** Enhanced features can be cached (`--use_cache`, `--cache_dir`).
- **Final features:** Saved in the models directory (e.g., `models/feature_names_*.pkl`).
- **How to run:**
  - Run `python train_models.py --use_enhanced_features` with desired options.
  - Features are generated and saved automatically.

---

### 5. Summary Table

| Feature Type         | Description                                 | Module/Class                  |
|---------------------|---------------------------------------------|-------------------------------|
| Price/Volume        | Raw OHLCV data                              | FeatureGenerator              |
| Rolling Stats       | Rolling mean, std, min, max, etc.           | FeatureGenerator              |
| Technical Indicators| MA, RSI, MACD, etc.                         | FeatureGenerator              |
| Macro/Event         | Market indices, earnings, splits, news      | EnhancedFeatureGenerator      |
| Interactions        | Feature crosses, polynomial features         | EnhancedFeatureGenerator      |
| Lagged/Encoded      | Lagged values, target encoding              | EnhancedFeatureGenerator      |

---

## In-Depth Explanations

- **Why these features?**
  - Price and volume capture market microstructure.
  - Rolling stats and technicals capture trends and volatility.
  - Macro and event features add context and help model regime changes.
  - Feature selection ensures only the most predictive features are used, improving generalization and reducing overfitting.

- **References in Code:**
  - `load_and_prepare_data()` in `train_models.py` orchestrates the entire process.
  - Feature generation and selection are modular and configurable via command-line arguments.

---

## How to Reproduce

1. Install requirements: `pip install -r requirements.txt`
2. Run: `python train_models.py --use_enhanced_features --feature_importance_selection --max_features 100`
3. Features and reports will be saved in the `models/` and `reports/` directories.

---

## Diagrams

![Feature Engineering Pipeline](reports/feature_engineering_pipeline.png)

(You can generate diagrams using tools like draw.io or matplotlib and save them in the `reports/` directory.)

---

# End of Feature Engineering Report
