# StocksPredicter

## Project Overview
This project predicts whether a given S&P 500 stock will outperform the S&P 500 index over the next 5 trading days using historical OHLCV data. It features a modular, end-to-end machine learning pipeline for feature generation, model training, evaluation, and a simple REST API for inference.

## Features
- **Feature Engineering:** Generates technical and advanced financial features from OHLCV data (per stock).
- **Robust Train/Test Split:** Multiple strategies (temporal, event-aware, walk-forward, expanding window) to avoid lookahead bias and account for market events.
- **Model Training:** Supports XGBoost, LightGBM, CatBoost, Random Forest, and more. Includes hyperparameter optimization (random, grid, Optuna).
- **Feature Selection:** F-statistic, mutual information, and advanced methods.
- **Pipeline Optimization:** Memory optimization, checkpointing, benchmarking, and optional GPU acceleration.
- **API:** FastAPI-based REST API for predictions (stock + date → prediction + certainty).
- **MLOps Plan:** (see `MLOps_plan.txt`) for Docker, experiment tracking, and pipeline orchestration.

## Directory Structure
- `src/` — Core modules (data, features, models, utils)
- `api/` — FastAPI app for model serving
- `notebooks/` — EDA, feature analysis, and workflow notebooks
- `reports/` — Model training and feature analysis reports
- `checkpoints/` — Model checkpoints and experiment runs
- `cache/` — Feature cache for faster computation
- `tests/` — Unit and integration tests

## Setup Instructions
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   For enhanced features:
   - TA-Lib (required for some technical indicators)
     - Windows: Download and install from [TA-Lib website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
     - Mac: `brew install ta-lib`
     - Linux: `sudo apt-get install ta-lib`
   - Then: `pip install ta-lib`

2. **Run Model Training:**
   ```sh
   python train_models.py --use_enhanced_features --models xgboost lightgbm catboost --optimization optuna --n_trials 50
   ```
   See `docs/training_guide.md` for more options.

3. **Run the API:**
   ```sh
   cd api
   uvicorn app:app --reload
   ```
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

4. **Run Notebooks:**
   - See `notebooks/` for EDA and feature analysis.

## API Usage
- **POST /predict**: `{ "symbol": "AAPL", "date": "YYYY-MM-DD" }` → prediction, certainty, features
- **GET /health**: Health check

## Documentation

The `docs/` directory contains guides and references to help you use, understand, and extend the StocksPredicter project.

**Model Results:** For detailed model results and analysis, see the `reports/` directory. Key contents include:

- **model_training_report.html** — Full training report with model comparison (XGBoost vs LightGBM), performance metrics (AUC, F1, accuracy, precision, recall), feature engineering summary, and recommendations. LightGBM achieved slightly higher AUC/accuracy, while XGBoost had higher recall.
- **model_comparison_chart.png** — Visual comparison of model performance.
- **feature_importance_comparison.png** — Feature importance plots for top models.
- **feature_selection_report.md** — Details on feature selection methods and top features used.
- **feature_data/** — JSON files summarizing features, splits, and class balance for each experiment.
- **optimization/benchmarks/** — Benchmark reports on training time and memory usage.
- **optimization/feature_selection/** — Additional feature selection results and plots.

The reports provide:
- Performance metrics for each model and split method
- Feature selection outcomes and top features
- Benchmarking of training time and memory
- Visualizations of feature importance and model results

See `model_training_report.html` for a comprehensive summary and recommendations.

### Contents

- **training_guide.md** — Step-by-step instructions for model training, hyperparameter tuning, and evaluation.
- **MLOps_Plan.md** — Outline of the MLOps strategy, including Dockerization, experiment tracking, and pipeline orchestration.
- **feature_engineering_overview.md** — Overview of feature engineering strategies and rationale.
- **model_selection_report.md** — Analysis and comparison of different model choices.
- **train_test_split_report.md** — Details on train/test split strategies and their impact.

Refer to these documents for in-depth explanations, usage examples, and best practices.

