# Model Training Automation Pipeline

This directory contains scripts for automating model training, feature tracking, and report generation.

## Overview

The automation pipeline includes:

1. **Model Experiment Runner**: Automates running different model configurations and split strategies
2. **Feature Tracking**: Saves and analyzes features used in training
3. **Report Generation**: Creates comprehensive comparative reports across all experiments

## Scripts

### `run_model_pipeline.py`

The master script that orchestrates the entire pipeline. It runs all the necessary steps in sequence.

**Usage:**
```bash
python scripts/run_model_pipeline.py [options]
```

**Options:**
- `--models`: List of models to train (default: xgboost, lightgbm, catboost)
- `--splits`: List of split methods to use (default: temporal, event_aware, walk_forward, expanding_window)
- `--optimization`: Hyperparameter optimization method (default: random)
- `--n_trials`: Number of optimization trials (default: 20)
- `--skip_training`: Skip training and just generate reports
- `--quick_run`: Run a quick test with minimal configuration

### `run_model_experiments.py`

Runs multiple model training experiments with different configurations.

**Usage:**
```bash
python scripts/run_model_experiments.py [options]
```

**Options:**
- `--models`: List of models to train
- `--splits`: List of split methods to use
- `--optimization`: Hyperparameter optimization method
- `--n_trials`: Number of optimization trials
- `--skip_existing`: Skip configurations that have already been run
- `--report_only`: Skip training and only generate the report

### `generate_consolidated_report.py`

Generates a comprehensive report from the results of multiple training runs.

**Usage:**
```bash
python scripts/generate_consolidated_report.py [options]
```

**Options:**
- `--feature_dir`: Directory containing feature data
- `--results_dir`: Directory containing experiment results
- `--output_dir`: Directory to save consolidated report
- `--report_id`: Specific report ID to analyze (default: latest)

## Feature Tracking

The pipeline tracks feature importance and selection across different model configurations. For each training run, it logs:

- Features used in training
- Feature importance scores
- Selected features and rationale
- Performance metrics

This information is saved to the `reports/feature_data` directory and is used to generate insights in the final report.

## Reports

The pipeline generates several types of reports:

1. **Per-Experiment Reports**: Basic performance metrics for each individual experiment
2. **Comparative Reports**: Compare performance across different models and split strategies
3. **Consolidated Reports**: Comprehensive analysis of all experiments with insights and recommendations

Reports are saved to the `reports/experiment_results` and `reports/consolidated_reports` directories.

## Example Workflow

1. **Quick Test Run**:
   ```bash
   python scripts/run_model_pipeline.py --quick_run
   ```

2. **Full Training Pipeline**:
   ```bash
   python scripts/run_model_pipeline.py
   ```

3. **Generate Reports Only**:
   ```bash
   python scripts/run_model_pipeline.py --skip_training
   ```

4. **Train Specific Models**:
   ```bash
   python scripts/run_model_pipeline.py --models xgboost lightgbm --splits temporal event_aware
   ```
