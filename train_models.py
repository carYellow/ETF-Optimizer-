#!/usr/bin/env python3
"""
Main Model Training Script

This script orchestrates the complete model training pipeline with:
- Enhanced feature engineering
- Robust train/test splitting
- Advanced model selection and training
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureGenerator
from src.data.enhanced_features import EnhancedFeatureGenerator
from src.data.train_test_split import RobustTrainTestSplit
from src.models.advanced_train import AdvancedModelTrainer

def parse_args():
    """Parse command line arguments."""
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
    
    return parser.parse_args()

def load_and_prepare_data(use_enhanced_features: bool = False):
    """Load data and generate features."""
    print("=== LOADING DATA ===")
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.prepare_training_data()
    
    print(f"Loaded {len(stock_data):,} stock records")
    print(f"Loaded {len(sp500_data):,} S&P 500 records")
    
    # Generate features
    print("\n=== GENERATING FEATURES ===")
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    if use_enhanced_features:
        print("\n=== ADDING ENHANCED FEATURES ===")
        enhanced_generator = EnhancedFeatureGenerator()
        df = enhanced_generator.generate_all_features(df, sp500_data)
        
        # Print feature groups
        feature_groups = enhanced_generator.get_feature_groups()
        print("\nFeature groups:")
        for group, features in feature_groups.items():
            print(f"  {group}: {len(features)} features")
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

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

def train_models(splits, model_names, optimization, n_trials, 
                feature_selection=None, n_features=None):
    """Train models on all splits."""
    print(f"\n=== TRAINING MODELS ===")
    print(f"Models: {model_names}")
    print(f"Optimization: {optimization}")
    
    all_results = []
    
    for i, (train_df, test_df) in enumerate(splits):
        print(f"\n--- Split {i+1}/{len(splits)} ---")
        
        # Prepare features
        X_train, y_train, feature_cols = prepare_features_for_training(train_df)
        X_test, y_test, _ = prepare_features_for_training(test_df)
        
        # Split train into train/validation
        val_size = 0.2
        n_val = int(len(X_train) * val_size)
        X_val = X_train.iloc[-n_val:]
        y_val = y_train.iloc[-n_val:]
        X_train = X_train.iloc[:-n_val]
        y_train = y_train.iloc[:-n_val]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Initialize trainer
        trainer = AdvancedModelTrainer()
        
        # Preprocess features
        X_train_scaled, X_val_scaled = trainer.preprocess_features(
            X_train, X_val, 
            scaler_type='robust',
            feature_selection=feature_selection,
            n_features=n_features
        )
        
        # Get test features using fitted scaler
        X_test_scaled = trainer.scalers['robust'].transform(X_test)
        if feature_selection and n_features:
            selector_key = f"{feature_selection}_{n_features}"
            X_test_scaled = trainer.feature_selectors[selector_key]['selector'].transform(X_test_scaled)
        
        # Train models
        if 'ensemble' in model_names:
            # Train ensemble of other models
            base_models = [m for m in model_names if m != 'ensemble']
            ensemble_result = trainer.train_ensemble(
                X_train_scaled, y_train, X_val_scaled, y_val, base_models
            )
            trainer.models['ensemble'] = ensemble_result
        else:
            # Train individual models
            for model_name in model_names:
                result = trainer.train_single_model(
                    model_name, X_train_scaled, y_train, 
                    X_val_scaled, y_val,
                    optimization=optimization, n_iter=n_trials
                )
                trainer.models[model_name] = result
        
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
    
    return all_results

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

def main():
    """Main training function."""
    args = parse_args()
    
    print("=== STOCK PREDICTION MODEL TRAINING ===")
    print(f"Configuration: {vars(args)}")
    
    # Load and prepare data
    df = load_and_prepare_data(use_enhanced_features=args.use_enhanced_features)
    
    # Split data
    splits = split_data(df, args.split_method, args.test_size, args.gap_days)
    
    # Train models
    all_results = train_models(
        splits, args.models, args.optimization, args.n_trials,
        args.feature_selection, args.n_features
    )
    
    # Generate final report
    generate_final_report(all_results)
    
    print("\n=== TRAINING COMPLETE ===")
    print("Check the models/ directory for saved models and reports.")

if __name__ == "__main__":
    main() 