# Model Selection and Training Report

## Overview
This report describes the model selection, training, and optimization process used in the stock prediction pipeline, referencing `train_models.py` and related modules. It covers the models tried, optimization strategies, evaluation, and how the best model was chosen.

---

### 1. Models Tried

- **Default Models:**
  - XGBoost (`xgboost`)
  - LightGBM (`lightgbm`)
  - Random Forest (`random_forest`)
- **Optional:**
  - Ensemble (stacking/blending of base models)
- **Reference:**
  - Models are specified via `--models` argument.
  - Implemented in `src.models.advanced_train.AdvancedModelTrainer`.

---

### 2. Training and Optimization

- **Hyperparameter Optimization:**
  - Options: grid search, random search, Optuna, or none (`--optimization`)
  - Number of trials: `--n_trials` (default 50)
  - Early stopping: `--early_stopping_rounds` (default 10)
- **Feature Selection:**
  - Statistical and advanced methods available (see Feature Engineering Report)
- **GPU Acceleration:**
  - Enabled by default if available (`--enable_gpu`)
- **Memory Optimization:**
  - Enabled by default (`--optimize_memory`)
- **Checkpointing and Benchmarking:**
  - Training progress and results are checkpointed and benchmarked for reproducibility.

---

### 3. Training Process

- For each split:
  - Data is split into train, validation, and test sets.
  - Features are preprocessed and selected.
  - Each model is trained on the training set, validated, and then evaluated on the test set.
  - Results are saved and plotted for comparison.
- **Reference:**
  - See `train_models()` in `train_models.py` for the full pipeline.

---

### 4. Evaluation and Model Selection

- **Metric:** ROC AUC (primary), plus others as available.
- **Reporting:**
  - Results for each split are aggregated.
  - Average, std, min, and max ROC AUC are reported for each model.
  - The best model is chosen based on average ROC AUC across splits.
- **Reports:**
  - Final report saved as `models/final_training_report.md`.
  - Model artifacts saved in `models/` (e.g., `lightgbm_*.pkl`).

---

### 5. Interesting Notes

- **Ensemble models** can be trained for improved performance.
- **Advanced feature selection** and **GPU optimization** are available for experimentation.
- **Memory and time tracking** are integrated for efficient large-scale training.

---

### 6. How to Run

- Example: `python train_models.py --models xgboost lightgbm random_forest --optimization optuna --n_trials 100`
- All options are configurable via command-line arguments.

---

## References in Code

- `train_models()` in `train_models.py` handles the training loop.
- `AdvancedModelTrainer` in `src.models.advanced_train` implements model logic.
- `PipelineOptimizer` in `src.utils.pipeline_optimizer` manages optimization, checkpointing, and reporting.

---

# End of Model Selection and Training Report
