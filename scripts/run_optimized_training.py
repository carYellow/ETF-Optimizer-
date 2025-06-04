#!/usr/bin/env python3
"""
Optimized Training Script

This script runs an optimized training pipeline using all the performance
optimization utilities we've created.
"""

import argparse
import os
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run optimized stock prediction model training')
    
    # Basic training arguments
    parser.add_argument('--models', nargs='+', default=['xgboost', 'lightgbm'],
                       help='Models to train')
    parser.add_argument('--split_method', type=str, default='event_aware',
                       choices=['temporal', 'event_aware', 'walk_forward'],
                       help='Train/test split method')
    parser.add_argument('--optimization', type=str, default='random',
                       choices=['random', 'optuna', 'none'],
                       help='Hyperparameter optimization method')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of optimization trials (reduced for speed)')
    
    # Optimization arguments
    parser.add_argument('--experiment_name', type=str, 
                       default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Name for this experiment run')
    parser.add_argument('--advanced_feature_selection', type=str, default='importance',
                       choices=['importance', 'correlation', 'rfe', 'shap', None],
                       help='Advanced feature selection method to use')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                       help='Number of rounds for early stopping')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use feature caching to speed up computations')
    parser.add_argument('--n_workers', type=int, default=-1,
                       help='Number of worker processes for parallel computation')
    parser.add_argument('--max_features', type=int, default=None, 
                       help='Maximum number of features to keep (default: auto-select 50%)')
    
    # Feature and data arguments
    parser.add_argument('--use_enhanced_features', action='store_true', default=True,
                       help='Use enhanced feature set (recommended)')
    parser.add_argument('--infinity_handling', type=str, default='median',
                       choices=['median', 'mean', 'drop', 'zero'],
                       help='Method to handle infinity and NaN values')
    
    return parser.parse_args()

def main():
    """Run optimized training."""
    args = parse_args()
    
    print("=== OPTIMIZED STOCK PREDICTION MODEL TRAINING ===")
    
    # Construct command with all optimizations enabled
    cmd = [
        "python", "train_models.py",
        f"--models {' '.join(args.models)}",
        f"--split_method {args.split_method}",
        f"--optimization {args.optimization}",
        f"--n_trials {args.n_trials}",
        f"--experiment_name {args.experiment_name}",
        f"--advanced_feature_selection {args.advanced_feature_selection}",
        f"--early_stopping_rounds {args.early_stopping_rounds}",
        f"--infinity_handling {args.infinity_handling}",
        f"--n_workers {args.n_workers}"
    ]
    
    # Add boolean flags
    if args.use_enhanced_features:
        cmd.append("--use_enhanced_features")
    if args.use_cache:
        cmd.append("--use_cache")
    if args.max_features:
        cmd.append(f"--max_features {args.max_features}")
    
    # Add optimization flags
    cmd.extend([
        "--enable_benchmarking",
        "--enable_checkpointing",
        "--enable_gpu",
        "--optimize_memory"
    ])
    
    # Join command
    cmd_str = " ".join(cmd)
    
    print(f"Running command:\n{cmd_str}\n")
    
    # Execute command
    start_time = time.time()
    os.system(cmd_str)
    
    # Calculate execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n=== OPTIMIZED TRAINING COMPLETE ===")
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Experiment name: {args.experiment_name}")
    print("Check reports/optimization/ for performance metrics")

if __name__ == "__main__":
    main()
