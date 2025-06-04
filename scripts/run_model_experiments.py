#!/usr/bin/env python3
"""
Model Experiment Runner

This script automates the process of running train_models.py across multiple configurations,
tracking performance across different models and train/test split strategies.
It generates a comprehensive comparative report of the results.
"""

import os
import sys
import subprocess
import time
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure proper importing of project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration for experiments
MODEL_TYPES = ['xgboost', 'lightgbm', 'catboost']  # Excluding random_forest - too slow
SPLIT_METHODS = ['temporal', 'event_aware', 'walk_forward', 'expanding_window']
OPTIMIZATION_METHODS = ['random', 'optuna']  # Optimization strategies

# Directories for results
RESULTS_DIR = os.path.join('reports', 'experiment_results')
FEATURE_STORAGE_DIR = os.path.join('reports', 'feature_data')

def parse_args():
    """Parse command line arguments for experiment runner."""
    parser = argparse.ArgumentParser(description='Run stock prediction model experiments')
    
    parser.add_argument('--models', nargs='+', 
                       default=MODEL_TYPES,
                       help='Models to train')
    parser.add_argument('--splits', nargs='+',
                      default=SPLIT_METHODS,
                      help='Train/test split methods')
    parser.add_argument('--optimization', type=str, default='random',
                       choices=['grid', 'random', 'optuna', 'none'],
                       help='Hyperparameter optimization method')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of optimization trials')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip configurations that have already been run')
    parser.add_argument('--feature_selection', type=str, default=None,
                       choices=['f_classif', 'mutual_info', None],
                       help='Feature selection method')
    parser.add_argument('--report_only', action='store_true',
                       help='Skip training and only generate the report')
    parser.add_argument('--save_features', action='store_true',
                       help='Save feature data to disk')
    parser.add_argument('--max_features', type=int, default=None,
                       help='Maximum number of features to select')
    
    return parser.parse_args()

def setup_directories():
    """Create necessary directories for results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FEATURE_STORAGE_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
def get_experiment_id():
    """Generate a unique experiment ID based on timestamp."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def run_single_experiment(model_type, split_method, optimization, n_trials, experiment_id, feature_selection=None, save_features=False, max_features=None):
    """
    Run a single training experiment with specified configuration.
    
    Args:
        model_type: Type of model to train
        split_method: Train/test split method
        optimization: Hyperparameter optimization method
        n_trials: Number of optimization trials
        experiment_id: Unique experiment identifier
        feature_selection: Feature selection method
        save_features: Whether to save feature data
        max_features: Maximum number of features to use
    
    Returns:
        Process result code
    """
    print(f"\n{'-'*80}")
    print(f"RUNNING EXPERIMENT: Model={model_type}, Split={split_method}, Optimization={optimization}")
    print(f"{'-'*80}")
    
    # Build command
    cmd = [
        sys.executable,
        'scripts/train_models.py',
        f'--models', model_type,
        f'--split_method', split_method,
        f'--optimization', optimization,
        f'--n_trials', str(n_trials),
        f'--experiment_name', f"exp_{experiment_id}_{model_type}_{split_method}",
        '--use_enhanced_features',
        '--enable_checkpointing',
        '--enable_benchmarking',
        '--optimize_memory',
        '--feature_importance_selection',
        '--max_features', '50'  # Limit to top 50 features
    ]
      # Add feature selection if specified
    if feature_selection:
        cmd.extend([f'--feature_selection', feature_selection])
    
    # Add save_features if specified
    if save_features:
        cmd.append('--save_features')
    
    # Override max_features if specified
    if max_features is not None:
        # Remove the existing max_features argument
        if '--max_features' in cmd:
            idx = cmd.index('--max_features')
            if idx + 1 < len(cmd):
                cmd.pop(idx + 1)  # Remove the value
            cmd.pop(idx)  # Remove the flag
        # Add the new value
        cmd.extend(['--max_features', str(max_features)])
    
    # Run the command
    start_time = time.time()
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    print(f"Experiment completed in {elapsed_time:.2f} seconds with exit code {result.returncode}")
    return result.returncode

def collect_experiment_results(experiment_id):
    """
    Collect and compile results from all experiments into a unified DataFrame.
    
    Args:
        experiment_id: Unique experiment identifier
    
    Returns:
        DataFrame with consolidated results
    """
    print("\nCollecting experiment results...")
    
    # Find all relevant result files
    result_files = []
    for root, dirs, files in os.walk('models'):
        for file in files:
            if file.startswith('results_') and file.endswith('.pkl'):
                result_files.append(os.path.join(root, file))
    
    # Also look in checkpoints directory
    for root, dirs, files in os.walk('checkpoints'):
        for file in files:
            if file.startswith('results_') and file.endswith('.pkl'):
                result_files.append(os.path.join(root, file))
    
    print(f"Found {len(result_files)} result files")
    
    # Compile results
    all_results = []
    
    for result_file in result_files:
        try:
            results = pd.read_pickle(result_file)
            # Extract configuration from filename if possible
            file_parts = os.path.basename(result_file).replace('.pkl', '').split('_')
            
            # Try to extract model and split method
            model_type = next((m for m in MODEL_TYPES if m in file_parts), "unknown")
            split_method = next((s for s in SPLIT_METHODS if s in file_parts), "unknown")
            
            # Add configuration to results
            if isinstance(results, pd.DataFrame):
                results['model_type'] = model_type
                results['split_method'] = split_method
                results['result_file'] = result_file
                all_results.append(results)
            elif isinstance(results, dict):
                # Handle dictionary format (probably from older runs)
                for model, metrics in results.items():
                    if isinstance(metrics, dict):
                        row = {**metrics, 'model': model, 'model_type': model_type, 
                              'split_method': split_method, 'result_file': result_file}
                        all_results.append(pd.DataFrame([row]))
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    if not all_results:
        print("No valid results found!")
        return pd.DataFrame()
    
    # Combine all results
    try:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
    except Exception as e:
        print(f"Error combining results: {e}")
        return pd.DataFrame()

def collect_feature_data():
    """
    Collect feature importance data from all experiments.
    
    Returns:
        Dictionary of feature importance data by model and split method
    """
    print("\nCollecting feature importance data...")
    
    feature_data = {}
    
    # Find all feature data files
    for root, dirs, files in os.walk(FEATURE_STORAGE_DIR):
        for file in files:
            if file.endswith('_features.json'):
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract configuration from filename
                    parts = os.path.basename(file).replace('_features.json', '').split('_')
                    model_type = next((m for m in MODEL_TYPES if m in parts), "unknown")
                    split_method = next((s for s in SPLIT_METHODS if s in parts), "unknown")
                    
                    key = f"{model_type}_{split_method}"
                    feature_data[key] = data
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    return feature_data

def generate_comparative_report(results_df, feature_data, experiment_id):
    """
    Generate a comprehensive comparative report of all experiments.
    
    Args:
        results_df: DataFrame with consolidated results
        feature_data: Dictionary of feature importance data
        experiment_id: Unique experiment identifier
    """
    print("\nGenerating comparative report...")
    
    if results_df.empty:
        print("No results to report!")
        return
    
    # Create report directory
    report_dir = os.path.join(RESULTS_DIR, f'report_{experiment_id}')
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Performance comparison across models and splits
    try:
        performance_comparison = results_df.pivot_table(
            index='split_method', 
            columns='model', 
            values='roc_auc',
            aggfunc='mean'
        )
        
        # Save comparison table
        performance_comparison.to_csv(os.path.join(report_dir, 'performance_comparison.csv'))
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(performance_comparison, annot=True, cmap='YlGnBu', fmt='.4f')
        plt.title('Model Performance (ROC AUC) by Split Method')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'performance_heatmap.png'))
        
        # Create bar chart comparison
        performance_melted = performance_comparison.reset_index().melt(
            id_vars=['split_method'],
            var_name='model',
            value_name='roc_auc'
        )
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='split_method', y='roc_auc', hue='model', data=performance_melted)
        plt.title('Model Performance by Split Method')
        plt.ylim(performance_melted['roc_auc'].min() * 0.95, performance_melted['roc_auc'].max() * 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'performance_bars.png'))
    except Exception as e:
        print(f"Error generating performance comparison: {e}")
    
    # 2. Best and worst performers
    try:
        # Find best configuration
        best_row = results_df.loc[results_df['roc_auc'].idxmax()]
        worst_row = results_df.loc[results_df['roc_auc'].idxmin()]
        
        performance_summary = {
            'best_configuration': {
                'model': best_row.get('model', 'unknown'),
                'split_method': best_row.get('split_method', 'unknown'),
                'roc_auc': best_row.get('roc_auc', 0),
                'f1_score': best_row.get('f1', 0)
            },
            'worst_configuration': {
                'model': worst_row.get('model', 'unknown'),
                'split_method': worst_row.get('split_method', 'unknown'),
                'roc_auc': worst_row.get('roc_auc', 0),
                'f1_score': worst_row.get('f1', 0)
            }
        }
        
        # Save summary
        with open(os.path.join(report_dir, 'performance_summary.json'), 'w') as f:
            json.dump(performance_summary, f, indent=4)
    except Exception as e:
        print(f"Error finding best/worst performers: {e}")
    
    # 3. Feature importance comparison
    try:
        if feature_data:
            # Aggregate feature importance across models
            feature_importance_agg = {}
            
            for key, data in feature_data.items():
                if 'feature_importance' in data:
                    for feature, importance in data['feature_importance'].items():
                        if feature not in feature_importance_agg:
                            feature_importance_agg[feature] = []
                        feature_importance_agg[feature].append(importance)
            
            # Calculate average importance
            avg_importance = {f: np.mean(imps) for f, imps in feature_importance_agg.items() if imps}
            
            # Sort by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Save top 20 features
            top_features = dict(sorted_features[:20])
            with open(os.path.join(report_dir, 'top_features.json'), 'w') as f:
                json.dump(top_features, f, indent=4)
            
            # Create feature importance visualization
            plt.figure(figsize=(12, 10))
            features = [f for f, _ in sorted_features[:20]]
            importances = [i for _, i in sorted_features[:20]]
            
            bars = plt.barh(features, importances)
            plt.title('Top 20 Feature Importance (Averaged Across Models)')
            plt.xlabel('Average Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'feature_importance.png'))
    except Exception as e:
        print(f"Error generating feature importance comparison: {e}")
    
    # 4. Generate HTML report
    try:
        generate_html_report(results_df, feature_data, performance_comparison, 
                           performance_summary, report_dir, experiment_id)
    except Exception as e:
        print(f"Error generating HTML report: {e}")
    
    print(f"Report generated in {report_dir}")

def generate_html_report(results_df, feature_data, performance_comparison, 
                       performance_summary, report_dir, experiment_id):
    """Generate a comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #e6f7ff; font-weight: bold; }}
            .container {{ margin-bottom: 30px; }}
            .model-comparison img {{ max-width: 100%; height: auto; }}
            .best {{ color: #27ae60; font-weight: bold; }}
            .worst {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Stock Prediction Model Comparison Report</h1>
        <p>Experiment ID: {experiment_id}</p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="container">
            <h2>Performance Comparison</h2>
            <p>The table below shows the ROC AUC scores for different model and split method combinations:</p>
            {performance_comparison.to_html()}
            
            <div class="model-comparison">
                <h3>Performance Visualization</h3>
                <img src="performance_heatmap.png" alt="Performance Heatmap">
                <img src="performance_bars.png" alt="Performance Bar Chart">
            </div>
        </div>
        
        <div class="container">
            <h2>Best and Worst Configurations</h2>
            <h3 class="best">Best Configuration</h3>
            <ul>
                <li>Model: {performance_summary['best_configuration']['model']}</li>
                <li>Split Method: {performance_summary['best_configuration']['split_method']}</li>
                <li>ROC AUC: {performance_summary['best_configuration']['roc_auc']:.4f}</li>
                <li>F1 Score: {performance_summary['best_configuration']['f1_score']:.4f}</li>
            </ul>
            
            <h3 class="worst">Worst Configuration</h3>
            <ul>
                <li>Model: {performance_summary['worst_configuration']['model']}</li>
                <li>Split Method: {performance_summary['worst_configuration']['split_method']}</li>
                <li>ROC AUC: {performance_summary['worst_configuration']['roc_auc']:.4f}</li>
                <li>F1 Score: {performance_summary['worst_configuration']['f1_score']:.4f}</li>
            </ul>
        </div>
        
        <div class="container">
            <h2>Feature Importance</h2>
            <p>The chart below shows the top 20 most important features across all models:</p>
            <img src="feature_importance.png" alt="Feature Importance">
        </div>
        
        <div class="container">
            <h2>Detailed Results</h2>
            <p>The table below shows detailed performance metrics for all experiments:</p>
            {results_df.to_html()}
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(report_dir, 'model_comparison_report.html'), 'w') as f:
        f.write(html_content)

def main():
    """Main function to run experiments and generate report."""
    args = parse_args()
    
    # Setup directories
    setup_directories()
    
    # Generate experiment ID
    experiment_id = get_experiment_id()
    
    print(f"=== STOCK PREDICTION MODEL EXPERIMENTS ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Models: {args.models}")
    print(f"Split methods: {args.splits}")
    print(f"Optimization: {args.optimization}")
    print(f"Trials: {args.n_trials}")
    
    # Skip training if report_only is specified
    if not args.report_only:
        # Run experiments for each model and split method
        for model_type in args.models:
            for split_method in args.splits:                run_single_experiment(
                    model_type=model_type,
                    split_method=split_method,
                    optimization=args.optimization,
                    n_trials=args.n_trials,
                    experiment_id=experiment_id,
                    feature_selection=args.feature_selection,
                    save_features=args.save_features if hasattr(args, 'save_features') else False,
                    max_features=args.max_features
                )
    
    # Collect and analyze results
    results_df = collect_experiment_results(experiment_id)
    feature_data = collect_feature_data()
    
    # Generate report
    generate_comparative_report(results_df, feature_data, experiment_id)
    
    print("\n=== EXPERIMENTS COMPLETE ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Check {os.path.join(RESULTS_DIR, f'report_{experiment_id}')} for the comparative report.")

if __name__ == "__main__":
    main()
