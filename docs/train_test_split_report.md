# Train/Test/Validation Split Report

## Overview
This report explains the data splitting strategies used in the stock prediction pipeline, referencing the code in `train_models.py` and the `RobustTrainTestSplit` class. The goal is to create robust, realistic splits that avoid lookahead bias and account for global events.

---

### 1. Splitting Methods Considered

- **Temporal Split:**
  - Simple chronological split (train on past, test on future).
  - Reference: `splitter.temporal_train_test_split()`
- **Event-Aware Split:**
  - Avoids periods around major market events (e.g., earnings, crises).
  - Reference: `splitter.event_aware_split(df, avoid_events=True)`
- **Walk-Forward Split:**
  - Multiple rolling splits to simulate real-world prediction.
  - Reference: `splitter.walk_forward_split(df, n_splits=5)`
- **Expanding Window Split:**
  - Training set grows over time, test set moves forward.
  - Reference: `splitter.expanding_window_split(df)`

---

### 2. Final Choice and Rationale

- **Default:** `event_aware` (can be changed via `--split_method`)
- **Why:**
  - Avoids lookahead bias by ensuring test data is strictly after train data.
  - Excludes periods around major events to prevent information leakage.
  - More realistic for financial time series.

---

### 3. Avoiding Lookahead Bias

- **Implementation:**
  - All splits ensure that no future data is used in training.
  - A gap (`--gap_days`, default 5) is enforced between train and test sets.
  - `splitter.validate_no_lookahead(train, test)` is called for each split.

**Diagram:**
```
Train Data ──────┐
                 │ (gap)
                 ▼
           Test Data

Events: [Earnings, Crises, Splits] are excluded from test set
```

---

### 4. Global Events and Other Considerations

- **Event-aware splitting** avoids periods of abnormal volatility or regime change.
- Ensures that models are evaluated on realistic, out-of-sample data.
- Multiple split strategies are available for experimentation.

---

### 5. Saving and Reproducibility

- **Splits are not saved as files** but are generated in-memory during training.
- All configuration is controlled via command-line arguments to `train_models.py`.
- Results and reports are saved in the `models/` and `reports/` directories.

---

### 6. How to Run

- Example: `python train_models.py --split_method event_aware --gap_days 5`
- All options:
  - `--split_method [temporal|event_aware|walk_forward|expanding_window]`
  - `--gap_days N`

---

## References in Code

- `split_data()` in `train_models.py` orchestrates the split process.
- `RobustTrainTestSplit` in `src.data.train_test_split` implements the logic.

