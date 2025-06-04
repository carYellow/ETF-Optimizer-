#!/usr/bin/env python3
"""
Master Script for Stock Prediction Model Training

This script orchestrates the complete model training, evaluation, and reporting pipeline:
1. Runs experiments with different models and split strategies
2. Tracks feature importance and selection
3. Generates comprehensive reports
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete model training pipeline')
    
    parser.add_argument('--models', nargs='+', 
                       default=['xgboost', 'lightgbm', 'catboost'],
                       help='Models to train')
    parser.add_argument('--splits', nargs='+',
                      default=['temporal', 'event_aware', 'walk_forward', 'expanding_window'],
                      help='Train/test split methods')
    parser.add_argument('--optimization', type=str, default='random',
                       choices=['grid', 'random', 'optuna', 'none'],
                       help='Hyperparameter optimization method')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of optimization trials')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and just generate reports')
    parser.add_argument('--quick_run', action='store_true',
                       help='Run a quick test with minimal configuration')
    parser.add_argument('--feature_importance_selection', action='store_true',
                       help='Use feature importance based selection')
    parser.add_argument('--save_features', action='store_true',
                       help='Save feature data to disk')
    parser.add_argument('--max_features', type=int, default=None,
                       help='Maximum number of features to use')
    
    return parser.parse_args()

def run_command(command, description):
    """Run a command and log the output."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'-'*80}\n")
    
    result = subprocess.run(command)
    
    print(f"\n{'-'*80}")
    print(f"COMPLETED: {description} with exit code {result.returncode}")
    print(f"{'='*80}\n")
    
    return result.returncode

def main():
    """Main function to run the complete pipeline."""
    args = parse_args()
    
    start_time = datetime.now()
    print(f"=== STARTING STOCK PREDICTION PIPELINE AT {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Generate experiment ID
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Use quick configuration if requested
    if args.quick_run:
        print("Running in QUICK MODE with minimal configuration")
        args.models = ['lightgbm']  # Just one model
        args.splits = ['temporal']  # Just one split method
        args.n_trials = 5           # Minimal optimization
    
    if not args.skip_training:        # Step 1: Run model experiments
        experiment_cmd = [
            sys.executable,
            'scripts/run_model_experiments.py',
            '--models'] + args.models + [
            '--splits'] + args.splits + [
            '--optimization', args.optimization,
            '--n_trials', str(args.n_trials)
        ]
        
        # Add feature importance selection if enabled
        if hasattr(args, 'feature_importance_selection') and args.feature_importance_selection:
            experiment_cmd.extend(['--feature_selection', 'mutual_info'])
              # Add max features if specified
        if hasattr(args, 'max_features') and args.max_features is not None:
            experiment_cmd.extend(['--max_features', str(args.max_features)])
            
        # Always enable saving features
        experiment_cmd.append('--save_features')
        
        run_command(experiment_cmd, "Model experiments")
    else:
        print("Skipping training as requested")
    
    # Step 2: Generate consolidated report
    report_cmd = [
        sys.executable,
        'scripts/generate_consolidated_report.py'
    ]
    
    run_command(report_cmd, "Generate consolidated report")
    
    # Calculate total runtime
    end_time = datetime.now()
    total_runtime = end_time - start_time
    
    print(f"=== PIPELINE COMPLETED AT {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Total runtime: {total_runtime}")
    print(f"Experiment ID: {experiment_id}")
    print("\nCheck the following directories for results:")
    print(f"- reports/experiment_results")
    print(f"- reports/consolidated_reports")
    print(f"- reports/feature_data")

if __name__ == "__main__":
    main()
