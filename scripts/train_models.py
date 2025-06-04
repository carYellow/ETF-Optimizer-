#!/usr/bin/env python3
"""
Main Model Training Script

This script orchestrates the complete model training pipeline with:
- Enhanced feature engineering
- Robust train/test splitting
- Advanced model selection and training
- Pipeline optimization for performance and efficiency
- Feature tracking and storage
- Comprehensive logging and reporting

Each section is clearly commented for maintainability and clarity.
"""

import sys
import pandas as pd
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm
import os
import gc  # For garbage collection
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_loader import StockDataLoader  # Handles data fetching and preprocessing
from src.data.feature_engineering import FeatureGenerator  # Basic feature engineering
from src.data.enhanced_features import EnhancedFeatureGenerator  # Advanced feature engineering
from src.data.train_test_split import RobustTrainTestSplit  # Train/test split strategies
from src.models.advanced_train import AdvancedModelTrainer  # Model selection, training, evaluation
from src.utils.pipeline_optimizer import PipelineOptimizer  # Pipeline optimization utilities
from src.utils.feature_tracker import FeatureTracker  # Feature tracking and storage

# --- Argument Parsing ---
def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    
    parser.add_argument('--use_enhanced_features', action='store_true',
                       help='Use enhanced feature set')
    parser.add_argument('--split_method', type=str, default='event_aware',
                       choices=['temporal', 'event_aware', 'walk_forward', 'expanding_window'],
                       help='Train/test split method')
    parser.add_argument('--models', nargs='+', 
                       default=['xgboost', 'lightgbm', 'random_forest'],
                       help='Models to train')
    parser.add_argument('--optimization', type=str, default='random',
                       choices=['grid', 'random', 'optuna', 'none'],
                       help='Hyperparameter optimization method')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--gap_days', type=int, default=5,
                       help='Gap days between train and test')
    parser.add_argument('--feature_selection', type=str, default=None,
                       choices=['f_classif', 'mutual_info', None],
                       help='Feature selection method')
    parser.add_argument('--n_features', type=int, default=None,
                       help='Number of features to select')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use feature caching to speed up computations')
    parser.add_argument('--no_cache', action='store_false', dest='use_cache',
                       help='Disable feature caching')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of worker processes for parallel computation')
    parser.add_argument('--no_parallel', action='store_false', dest='use_parallel',
                       help='Disable parallel processing')
    parser.add_argument('--use_parallel', action='store_true', default=True,
                       help='Enable parallel processing')
    parser.add_argument('--memory_profile', action='store_true',
                       help='Enable memory profiling (requires memory_profiler package)')
    parser.add_argument('--cache_dir', type=str, default='cache/features',
                       help='Directory to store feature cache')
    parser.add_argument('--feature_importance_selection', action='store_true',
                       help='Use feature importance based selection to reduce dimensionality')
    parser.add_argument('--max_features', type=int, default=None,
                       help='Maximum number of features to select when using feature importance selection')
    
    # Pipeline optimization arguments
    parser.add_argument('--enable_benchmarking', action='store_true', default=True,
                       help='Enable benchmarking and performance tracking')
    parser.add_argument('--disable_benchmarking', action='store_false', dest='enable_benchmarking',
                       help='Disable benchmarking and performance tracking')
    parser.add_argument('--enable_checkpointing', action='store_true', default=True,
                       help='Enable checkpointing to resume training if interrupted')
    parser.add_argument('--disable_checkpointing', action='store_false', dest='enable_checkpointing',
                       help='Disable checkpointing')
    parser.add_argument('--enable_gpu', action='store_true', default=True,
                       help='Enable GPU acceleration if available')
    parser.add_argument('--disable_gpu', action='store_false', dest='enable_gpu',
                       help='Disable GPU acceleration')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to store checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the current experiment (for checkpoint management)')
    parser.add_argument('--advanced_feature_selection', type=str, default=None,
                       choices=['importance', 'correlation', 'rfe', 'shap', None],
                       help='Advanced feature selection method to use')
    parser.add_argument('--infinity_handling', type=str, default='median',
                       choices=['median', 'mean', 'drop', 'zero'],
                       help='Method to handle infinity and NaN values')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                       help='Number of rounds for early stopping')
    parser.add_argument('--optimize_memory', action='store_true', default=True,
                       help='Enable memory usage optimization')
    parser.add_argument('--disable_memory_optimization', action='store_false', dest='optimize_memory',
                       help='Disable memory usage optimization')
    
    # Feature tracking arguments
    parser.add_argument('--save_features', action='store_true', default=True,
                       help='Save extracted features to disk')
    parser.add_argument('--disable_feature_saving', action='store_false', dest='save_features',
                       help='Disable saving features to disk')
    parser.add_argument('--feature_storage_dir', type=str, default='reports/feature_data',
                       help='Directory to store feature data')
    parser.add_argument('--track_feature_importance', action='store_true', default=True,
                       help='Track feature importance metrics')
    parser.add_argument('--disable_feature_importance', action='store_false', dest='track_feature_importance',
                       help='Disable tracking feature importance metrics')
    
    return parser.parse_args()

# --- Data Loading and Preparation ---
def load_and_prepare_data(use_enhanced_features: bool = False, use_cache: bool = True, 
                      n_workers: int = None, use_parallel: bool = True, cache_dir: str = None,
                      feature_importance_selection: bool = False, max_features: int = None,
                      optimizer = None):
    """Load data and generate features."""
    print("=== LOADING DATA ===")
    start_time = time.time()
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.prepare_training_data()
    
    print(f"Loaded {len(stock_data):,} stock records")
    print(f"Loaded {len(sp500_data):,} S&P 500 records")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    # Optimize memory usage if optimizer available
    if optimizer and optimizer.memory_optimizer:
        stock_data = optimizer.optimize_dataframe(stock_data)
        sp500_data = optimizer.optimize_dataframe(sp500_data)
        optimizer.log_memory_usage("After data loading optimization")
    
    # Generate features
    print("\n=== GENERATING FEATURES ===")
    feature_start = time.time()
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    # Free up memory
    del stock_data
    if optimizer:
        optimizer.force_gc()
    else:
        gc.collect()
    
    print(f"Basic feature generation took {time.time() - feature_start:.2f} seconds")
    
    # Optimize dataframe memory usage
    if optimizer and optimizer.memory_optimizer:
        df = optimizer.optimize_dataframe(df)
        optimizer.log_memory_usage("After basic feature generation")
    
    if use_enhanced_features:
        print("\n=== ADDING ENHANCED FEATURES ===")
        enhanced_start = time.time()
        enhanced_generator = EnhancedFeatureGenerator(
            use_cache=use_cache, 
            n_workers=n_workers,
            cache_dir=cache_dir,
            feature_importance_selection=feature_importance_selection,
            max_features=max_features
        )
        df = enhanced_generator.generate_all_features(
            df, 
            sp500_data, 
            use_parallel=use_parallel
        )
        
        # Print feature groups
        feature_groups = enhanced_generator.get_feature_groups()
        print("\nFeature groups:")
        for group, features in feature_groups.items():
            print(f"  {group}: {len(features)} features")
        print(f"Enhanced feature generation took {time.time() - enhanced_start:.2f} seconds")
        
        # Apply advanced feature selection if requested
        if optimizer and hasattr(optimizer, 'feature_selector') and 'Label' in df.columns:
            # Get numeric features (exclude Symbol and Label)
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in ['Label', 'Symbol']]
            X = df[feature_cols]
            y = df['Label']                # Use the optimizer's feature selection capabilities
            if feature_importance_selection and max_features:
                print(f"\n=== APPLYING ADVANCED FEATURE SELECTION ===")
                selected_features = optimizer.select_features(
                    X, y, 
                    method='importance',
                    n_features=max_features,
                )
                
                # Keep only selected features plus Symbol and Label
                keep_cols = selected_features + (['Symbol', 'Label'] if 'Symbol' in df.columns else ['Label'])
                df = df[keep_cols]
                print(f"Reduced features from {len(feature_cols)} to {len(selected_features)} ({len(selected_features)/len(feature_cols)*100:.1f}%)")
    
    # Free up more memory
    del sp500_data
    if optimizer:
        optimizer.force_gc()
    else:
        gc.collect()
    
    # Final memory optimization
    if optimizer and optimizer.memory_optimizer:
        df = optimizer.optimize_dataframe(df)
        optimizer.log_memory_usage("After complete feature generation")
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total data preparation took {time.time() - start_time:.2f} seconds")
    
    return df

# --- Data Splitting ---
def split_data(df: pd.DataFrame, method: str, test_size: float, gap_days: int):
    """Split data using specified method."""
    print(f"\n=== SPLITTING DATA ({method}) ===")
    
    splitter = RobustTrainTestSplit(gap_days=gap_days)
    
    if method == 'temporal':
        train_df, test_df = splitter.temporal_train_test_split(df, test_size=test_size)
        splits = [(train_df, test_df)]
    
    elif method == 'event_aware':
        train_df, test_df = splitter.event_aware_split(df, avoid_events=True)
        splits = [(train_df, test_df)]
    
    elif method == 'walk_forward':
        splits = splitter.walk_forward_split(df, n_splits=5)
    
    elif method == 'expanding_window':
        splits = splitter.expanding_window_split(df)
    
    else:
        raise ValueError(f"Unknown split method: {method}")
    
    # Validate no lookahead
    for i, (train, test) in enumerate(splits):
        splitter.validate_no_lookahead(train, test)
    
    return splits

# --- Feature Preparation for Training ---
def prepare_features_for_training(df: pd.DataFrame):
    """Prepare features and labels for training."""
    # Get numeric features (exclude Symbol and Label)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Label', 'Symbol']]
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['Label']
    
    print(f"\nFeatures prepared: {len(feature_cols)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

# --- Model Training ---
def train_models(splits, model_names, optimization, n_trials, 
                feature_selection=None, n_features=None, optimizer=None,
                advanced_feature_selection=None, early_stopping_rounds=10,
                infinity_handling='median', feature_tracker=None, 
                save_features=False, split_method='unknown'):
    """Train models on all splits."""
    print(f"\n=== TRAINING MODELS ===")
    print(f"Models: {model_names}")
    print(f"Optimization: {optimization}")
    
    all_results = []
    total_splits = len(splits)
    
    for i, (train_df, test_df) in enumerate(splits):
        split_start_time = time.time()
        print(f"\n--- Split {i+1}/{total_splits} ---")
        
        # Optimize memory usage if optimizer available
        if optimizer and optimizer.memory_optimizer:
            train_df = optimizer.optimize_dataframe(train_df)
            test_df = optimizer.optimize_dataframe(test_df)
            optimizer.log_memory_usage(f"After optimizing split {i+1} data")
        
        # Prepare features
        X_train, y_train, feature_cols = prepare_features_for_training(train_df)
        X_test, y_test, _ = prepare_features_for_training(test_df)
        
        # Track original features if feature tracker enabled
        if feature_tracker and save_features:
            print("Tracking original features...")
            feature_tracker.save_features(
                feature_names=feature_cols,
                model_type="all",
                split_method=split_method,
                selection_method="none",
                additional_metrics={
                    'split_index': i,
                    'total_features': len(feature_cols),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'positive_class_ratio': float(y_train.mean())
                }
            )
        
        # Free up memory
        del train_df, test_df
        if optimizer:
            optimizer.force_gc()
        else:
            gc.collect()
        
        # Split train into train/validation
        val_size = 0.2
        n_val = int(len(X_train) * val_size)
        X_val = X_train.iloc[-n_val:]
        y_val = y_train.iloc[-n_val:]
        X_train = X_train.iloc[:-n_val]
        y_train = y_train.iloc[:-n_val]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Apply advanced feature selection if requested
        if optimizer and advanced_feature_selection and optimizer.feature_selector:
            print(f"\nApplying advanced feature selection: {advanced_feature_selection}")
            selected_features = optimizer.select_features(
                X_train, y_train, 
                method=advanced_feature_selection,
                n_features=n_features if n_features else int(len(feature_cols) * 0.5)  # Default to 50% reduction
            )
            
            # Filter features
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]
            
            print(f"Reduced features from {len(feature_cols)} to {len(selected_features)}")
            
            # Track selected features if feature tracker enabled
            if feature_tracker and save_features:
                print(f"Tracking selected features from {advanced_feature_selection}...")
                feature_tracker.save_features(
                    feature_names=selected_features,
                    model_type="all",
                    split_method=split_method,
                    selected_features=selected_features,
                    selection_method=advanced_feature_selection,
                    additional_metrics={
                        'split_index': i,
                        'original_feature_count': len(feature_cols),
                        'selected_feature_count': len(selected_features),
                        'reduction_percentage': (1 - len(selected_features) / len(feature_cols)) * 100
                    }
                )
            
            # Update feature_cols
            feature_cols = selected_features
        
        # Initialize trainer
        trainer = AdvancedModelTrainer()
        
        # Preprocess features
        print("Preprocessing features...")
        preprocess_start = time.time()
        X_train_scaled, X_val_scaled = trainer.preprocess_features(
            X_train, X_val, 
            scaler_type='robust',
            feature_selection=feature_selection,
            n_features=n_features,
            y_train=y_train  # Pass y_train for feature selection
        )
        
        # Get test features using fitted scaler
        X_test_scaled = trainer.scalers['robust'].transform(X_test)
        if feature_selection and n_features:
            selector_key = f"{feature_selection}_{n_features}"
            X_test_scaled = trainer.feature_selectors[selector_key]['selector'].transform(X_test_scaled)
        print(f"Preprocessing completed in {time.time() - preprocess_start:.2f} seconds")
        
        # Free up more memory
        del X_train, X_val, X_test
        if optimizer:
            optimizer.force_gc()
        else:
            gc.collect()
        
        # Checkpoint before training
        if optimizer and optimizer.checkpoint_manager:
            checkpoint_state = {
                "split": i,
                "X_train_shape": X_train_scaled.shape,
                "X_val_shape": X_val_scaled.shape,
                "X_test_shape": X_test_scaled.shape,
                "feature_selection": feature_selection,
                "n_features": n_features,
                "advanced_feature_selection": advanced_feature_selection
            }
            optimizer.checkpoint(f"before_training_split_{i}", checkpoint_state)
        
        # Apply GPU optimization if available
        if optimizer and optimizer.gpu_accelerator and optimizer.gpu_accelerator.gpu_available:
            print("Optimizing models for GPU acceleration...")
            # Get model configs
            model_configs = trainer.get_model_configs()
            
            # Optimize each model for GPU
            for model_name in model_names:
                if model_name in model_configs:
                    model_configs[model_name] = optimizer.optimize_model_for_gpu(
                        model_name, model_configs[model_name]
                    )
        
        # Train models
        print("Training models...")
        if 'ensemble' in model_names:
            # Train ensemble of other models
            base_models = [m for m in model_names if m != 'ensemble']
            ensemble_result = trainer.train_ensemble(
                X_train_scaled, y_train, X_val_scaled, y_val, base_models
            )
            trainer.models['ensemble'] = ensemble_result
        else:
            # Train individual models
            for model_name in tqdm(model_names, desc="Training models"):
                model_start = time.time()
                result = trainer.train_single_model(
                    model_name, X_train_scaled, y_train, 
                    X_val_scaled, y_val,
                    optimization=optimization, n_iter=n_trials,
                    early_stopping_rounds=early_stopping_rounds
                )
                trainer.models[model_name] = result
                print(f"  {model_name} trained in {time.time() - model_start:.2f} seconds")
                
                # Checkpoint after each model
                if optimizer and optimizer.checkpoint_manager:
                    optimizer.checkpoint(
                        f"model_{model_name}_split_{i}", 
                        {"model_name": model_name, "model_result": result}
                    )
                
                # Track feature importance if feature tracker enabled
                if feature_tracker and save_features and model_name in trainer.models:
                    model_obj = trainer.models[model_name].get('model')
                    if model_obj:
                        print(f"Tracking feature importance for {model_name}...")
                        # Get feature names depending on preprocessing
                        if feature_selection and n_features:
                            selector_key = f"{feature_selection}_{n_features}"
                            if selector_key in trainer.feature_selectors:
                                selector = trainer.feature_selectors[selector_key].get('selector')
                                if hasattr(selector, 'get_support'):
                                    selected_idx = selector.get_support(indices=True)
                                    model_features = [feature_cols[i] for i in selected_idx if i < len(feature_cols)]
                                else:
                                    model_features = feature_cols
                            else:
                                model_features = feature_cols
                        else:
                            model_features = feature_cols
                        
                        # Get feature importance
                        feature_importance = feature_tracker.format_feature_importance(model_obj, model_features)
                        
                        if feature_importance:
                            # Analyze feature importance
                            importance_analysis = feature_tracker.analyze_feature_importance(feature_importance)
                            
                            # Save feature data
                            feature_tracker.save_features(
                                feature_names=model_features,
                                feature_importance=feature_importance,
                                model_type=model_name,
                                split_method=split_method,
                                selected_features=model_features,
                                selection_method=f"{feature_selection}_{n_features}" if feature_selection else "none",
                                additional_metrics={
                                    'split_index': i,
                                    'importance_analysis': importance_analysis,
                                    'model_performance': {
                                        'roc_auc': result.get('roc_auc', 0),
                                        'f1': result.get('f1', 0),
                                        'accuracy': result.get('accuracy', 0)
                                    }
                                }
                            )
        
        # Evaluate on test set
        print("\n--- Test Set Evaluation ---")
        results_df = trainer.evaluate_models(X_test_scaled, y_test)
        print(results_df)
        
        # Plot comparison
        trainer.plot_model_comparison(results_df)
        
        # Save models
        trainer.save_models()
        
        # Generate report
        trainer.generate_training_report(results_df)
        
        all_results.append({
            'split': i,
            'results': results_df,
            'trainer': trainer
        })
        
        # Final checkpoint for this split
        if optimizer and optimizer.checkpoint_manager:
            optimizer.checkpoint(
                f"split_{i}_complete", 
                {
                    "split": i,
                    "results": results_df.to_dict(),
                    "training_time": time.time() - split_start_time
                }
            )
        
        print(f"Split {i+1} completed in {time.time() - split_start_time:.2f} seconds")
    
    return all_results

# --- Reporting ---
def generate_final_report(all_results):
    """Generate final training report across all splits."""
    print("\n=== FINAL REPORT ===")
    
    # Aggregate results across splits
    model_scores = {}
    
    for result in all_results:
        results_df = result['results']
        for _, row in results_df.iterrows():
            model = row['model']
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(row['roc_auc'])
    
    # Calculate average scores
    avg_scores = []
    for model, scores in model_scores.items():
        avg_scores.append({
            'model': model,
            'avg_roc_auc': np.mean(scores),
            'std_roc_auc': np.std(scores),
            'min_roc_auc': np.min(scores),
            'max_roc_auc': np.max(scores)
        })
    
    avg_df = pd.DataFrame(avg_scores).sort_values('avg_roc_auc', ascending=False)
    
    print("\nAverage Model Performance Across Splits:")
    print(avg_df.to_string(index=False))
    
    # Save report
    with open('models/final_training_report.md', 'w') as f:
        f.write("# Final Training Report\n\n")
        f.write("## Average Performance Across Splits\n\n")
        f.write(avg_df.to_markdown(index=False))
        f.write("\n\n## Best Model\n\n")
        f.write(f"Best average model: {avg_df.iloc[0]['model']}\n")
        f.write(f"Average ROC AUC: {avg_df.iloc[0]['avg_roc_auc']:.4f} ± {avg_df.iloc[0]['std_roc_auc']:.4f}\n")

# --- Memory Profiling ---
# Memory profiling is optional and only used if memory_profiler is installed.
try:
    from memory_profiler import profile
except ImportError:
    # Create a dummy profile decorator if memory_profiler is not installed
    def profile(func):
        return func

# --- Main Pipeline ---
def main():
    """
    Main function to run the training pipeline:
    1. Load and preprocess data
    2. Feature engineering (basic/enhanced)
    3. Train/test split (robust, event-aware)
    4. Model training and selection
    5. Evaluation and reporting
    6. Pipeline optimization (optional)
    7. Feature tracking and storage
    """
    args = parse_args()
    
    print("=== STOCK PREDICTION MODEL TRAINING ===")
    print(f"Configuration: {vars(args)}")
    
    # Initialize the pipeline optimizer
    optimizer = PipelineOptimizer(
        output_dir="reports/optimization",
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        random_state=42,
        n_jobs=args.n_workers if args.n_workers else -1,
        enable_gpu=args.enable_gpu,
        enable_memory_optimization=args.optimize_memory,
        enable_checkpointing=args.enable_checkpointing,
        enable_benchmarking=args.enable_benchmarking,
        verbose=True
    )
    
    # Initialize feature tracker
    if args.save_features:
        os.makedirs(args.feature_storage_dir, exist_ok=True)
        feature_tracker = FeatureTracker(storage_dir=args.feature_storage_dir)
    
    # Save configuration as a checkpoint
    if args.enable_checkpointing:
        optimizer.checkpoint_manager.save_config(vars(args))
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    total_start_time = time.time()
    
    # Load and prepare data with optimization
    df = optimizer.track_time("data_loading")(
        optimizer.track_memory("data_loading")(load_and_prepare_data)
    )(
        use_enhanced_features=args.use_enhanced_features,
        use_cache=args.use_cache,
        n_workers=args.n_workers,
        use_parallel=args.use_parallel,
        cache_dir=args.cache_dir,
        feature_importance_selection=args.feature_importance_selection,
        max_features=args.max_features,
        optimizer=optimizer  # Pass the optimizer to the function
    )
    
    # Checkpoint after data preparation
    if args.enable_checkpointing:
        optimizer.checkpoint("data_preparation_complete", {
            "data_shape": df.shape,
            "memory_usage_mb": optimizer.memory_optimizer.get_memory_usage() if optimizer.memory_optimizer else None
        })
    
    # Split data with optimization
    splits = optimizer.track_time("data_splitting")(split_data)(
        df, args.split_method, args.test_size, args.gap_days
    )
    
    # Free up memory
    optimizer.force_gc()
    del df
    gc.collect()
    
    # Train models with optimization
    all_results = optimizer.track_time("model_training")(
        optimizer.track_memory("model_training")(train_models)
    )(
        splits, args.models, args.optimization, args.n_trials,
        args.feature_selection, args.n_features,
        optimizer=optimizer,  # Pass the optimizer to the function
        advanced_feature_selection=args.advanced_feature_selection,
        early_stopping_rounds=args.early_stopping_rounds,
        infinity_handling=args.infinity_handling,
        feature_tracker=feature_tracker if args.save_features else None,
        save_features=args.save_features,
        split_method=args.split_method
    )
    
    # Generate final report
    generate_final_report(all_results)
    
    # Generate optimization report
    optimizer.generate_optimization_report()
    
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("Check the models/ directory for saved models and reports.")
    print("Check the reports/optimization/ directory for performance reports.")

if __name__ == "__main__":
    # Entry point for script execution
    try:
        main()
    except Exception as e:
        print(f"Error during training pipeline: {e}")