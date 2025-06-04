#!/usr/bin/env python3
"""
Performance Benchmarking Script

This script benchmarks the optimized stock prediction model training pipeline
to measure improvements in training time and memory usage.
"""

import time
import argparse
import pandas as pd
import numpy as np
import gc
import os
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureGenerator
from src.data.enhanced_features import EnhancedFeatureGenerator
from src.data.train_test_split import RobustTrainTestSplit
from src.models.advanced_train import AdvancedModelTrainer

def measure_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def benchmark_feature_generation(use_cache=False, use_parallel=False, n_workers=None, 
                              use_vectorized=False, feature_importance=False, max_features=None):
    """Benchmark feature generation performance."""
    # Record starting memory
    start_memory = measure_memory()
    start_time = time.time()
    
    # Load data
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.prepare_training_data()
    
    # Generate basic features
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    # Enhanced features
    enhanced_generator = EnhancedFeatureGenerator(
        use_cache=use_cache,
        n_workers=n_workers,
        feature_importance_selection=feature_importance,
        max_features=max_features
    )
    
    # Generate enhanced features
    df = enhanced_generator.generate_all_features(
        df,
        sp500_data,
        use_parallel=use_parallel,
        use_vectorized=use_vectorized
    )
    
    # Record metrics
    total_time = time.time() - start_time
    peak_memory = measure_memory() - start_memory
    
    # Clean up
    del df, stock_data, sp500_data
    gc.collect()
    
    return {
        "total_time": total_time,
        "peak_memory_mb": peak_memory,
        "feature_count": len(enhanced_generator.feature_names)
    }

def benchmark_model_training(model_type, use_early_stopping=True, use_validation=True):
    """Benchmark model training performance."""
    # Record starting metrics
    start_memory = measure_memory()
    start_time = time.time()
    
    # Load data (limited subset for benchmarking)
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.prepare_training_data(max_symbols=10)  # Limit symbols for benchmark
    
    # Generate features
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    # Enhanced features
    enhanced_generator = EnhancedFeatureGenerator(use_cache=True, n_workers=4)
    df = enhanced_generator.generate_all_features(df, sp500_data, use_parallel=True)
    
    # Split data
    splitter = RobustTrainTestSplit(gap_days=5)
    train_df, test_df = splitter.temporal_train_test_split(df, test_size=0.2)
    
    # Prepare features
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Label', 'Symbol']]
    X_train = train_df[feature_cols]
    y_train = train_df['Label']
    X_test = test_df[feature_cols]
    y_test = test_df['Label']
    
    # Validation split if using early stopping
    if use_validation:
        val_size = 0.2
        n_val = int(len(X_train) * val_size)
        X_val = X_train.iloc[-n_val:]
        y_val = y_train.iloc[-n_val:]
        X_train = X_train.iloc[:-n_val]
        y_train = y_train.iloc[:-n_val]
    else:
        X_val, y_val = None, None
    
    # Train model
    trainer = AdvancedModelTrainer()
    
    if model_type == 'xgboost':
        model, train_time = trainer.train_xgboost(
            X_train, y_train, X_val, y_val, 
            use_early_stopping=use_early_stopping
        )
    elif model_type == 'lightgbm':
        model, train_time = trainer.train_lightgbm(
            X_train, y_train, X_val, y_val,
            use_early_stopping=use_early_stopping
        )
    elif model_type == 'catboost':
        model, train_time = trainer.train_catboost(
            X_train, y_train, X_val, y_val,
            use_early_stopping=use_early_stopping
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Test model
    accuracy, f1, precision, recall = trainer.evaluate_model(model, X_test, y_test)
    
    # Record metrics
    total_time = time.time() - start_time
    peak_memory = measure_memory() - start_memory
    
    # Clean up
    del df, train_df, test_df, X_train, y_train, X_test, y_test
    if use_validation:
        del X_val, y_val
    gc.collect()
    
    return {
        "total_time": total_time,
        "train_time": train_time,
        "peak_memory_mb": peak_memory,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def run_benchmarks(output_file=None):
    """Run all benchmarks and save results."""
    results = []
    
    print("=== FEATURE GENERATION BENCHMARKS ===")
    
    # Baseline - no optimizations
    print("\nBaseline (no optimizations)...")
    baseline = benchmark_feature_generation(use_cache=False, use_parallel=False)
    baseline["name"] = "Baseline"
    results.append(baseline)
    print(f"Time: {baseline['total_time']:.2f}s, Memory: {baseline['peak_memory_mb']:.2f}MB, Features: {baseline['feature_count']}")
    
    # Cache only
    print("\nWith caching...")
    cache_only = benchmark_feature_generation(use_cache=True, use_parallel=False)
    cache_only["name"] = "Cache Only"
    results.append(cache_only)
    print(f"Time: {cache_only['total_time']:.2f}s, Memory: {cache_only['peak_memory_mb']:.2f}MB, Features: {cache_only['feature_count']}")
    
    # Parallel only
    print("\nWith parallelism...")
    parallel_only = benchmark_feature_generation(use_cache=False, use_parallel=True, n_workers=4)
    parallel_only["name"] = "Parallel Only"
    results.append(parallel_only)
    print(f"Time: {parallel_only['total_time']:.2f}s, Memory: {parallel_only['peak_memory_mb']:.2f}MB, Features: {parallel_only['feature_count']}")
    
    # Vectorized features
    print("\nWith vectorized features...")
    vectorized = benchmark_feature_generation(use_cache=False, use_parallel=False, use_vectorized=True)
    vectorized["name"] = "Vectorized"
    results.append(vectorized)
    print(f"Time: {vectorized['total_time']:.2f}s, Memory: {vectorized['peak_memory_mb']:.2f}MB, Features: {vectorized['feature_count']}")
    
    # Feature importance selection
    print("\nWith feature importance selection...")
    feat_importance = benchmark_feature_generation(use_cache=False, use_parallel=False, feature_importance=True, max_features=50)
    feat_importance["name"] = "Feature Selection"
    results.append(feat_importance)
    print(f"Time: {feat_importance['total_time']:.2f}s, Memory: {feat_importance['peak_memory_mb']:.2f}MB, Features: {feat_importance['feature_count']}")
    
    # Full optimization
    print("\nFull optimization...")
    full_opt = benchmark_feature_generation(use_cache=True, use_parallel=True, n_workers=4, 
                                        use_vectorized=True, feature_importance=True, max_features=50)
    full_opt["name"] = "Full Optimization"
    results.append(full_opt)
    print(f"Time: {full_opt['total_time']:.2f}s, Memory: {full_opt['peak_memory_mb']:.2f}MB, Features: {full_opt['feature_count']}")
    
    print("\n=== MODEL TRAINING BENCHMARKS ===")
    
    models = ['xgboost', 'lightgbm', 'catboost']
    
    for model in models:
        # Without early stopping
        print(f"\n{model.upper()} without early stopping...")
        no_early = benchmark_model_training(model, use_early_stopping=False)
        no_early["name"] = f"{model} - No Early Stop"
        results.append(no_early)
        print(f"Time: {no_early['total_time']:.2f}s, Train Time: {no_early['train_time']:.2f}s, Memory: {no_early['peak_memory_mb']:.2f}MB")
        print(f"Accuracy: {no_early['accuracy']:.4f}, F1: {no_early['f1_score']:.4f}")
        
        # With early stopping
        print(f"\n{model.upper()} with early stopping...")
        with_early = benchmark_model_training(model, use_early_stopping=True)
        with_early["name"] = f"{model} - Early Stop"
        results.append(with_early)
        print(f"Time: {with_early['total_time']:.2f}s, Train Time: {with_early['train_time']:.2f}s, Memory: {with_early['peak_memory_mb']:.2f}MB")
        print(f"Accuracy: {with_early['accuracy']:.4f}, F1: {with_early['f1_score']:.4f}")
    
    # Save results to CSV
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    # Generate visualization
    plot_benchmarks(results)
    
    return results

def plot_benchmarks(results):
    """Create visualizations of benchmark results."""
    # Create directory for plots
    os.makedirs("reports", exist_ok=True)
    
    # Filter results into feature generation and model training
    feature_results = [r for r in results if "Features" in r["name"] or "Baseline" in r["name"] or "Cache" in r["name"] or "Parallel" in r["name"] or "Vectorized" in r["name"] or "Selection" in r["name"] or "Optimization" in r["name"]]
    model_results = [r for r in results if any(model in r["name"] for model in ["xgboost", "lightgbm", "catboost"])]
    
    # Plot feature generation time comparison
    plt.figure(figsize=(12, 6))
    names = [r["name"] for r in feature_results]
    times = [r["total_time"] for r in feature_results]
    plt.bar(names, times)
    plt.title("Feature Generation Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/feature_generation_time.png")
    
    # Plot model training time comparison
    plt.figure(figsize=(12, 6))
    names = [r["name"] for r in model_results]
    times = [r["train_time"] for r in model_results]
    plt.bar(names, times)
    plt.title("Model Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/model_training_time.png")
    
    # Plot memory usage
    plt.figure(figsize=(12, 6))
    names = [r["name"] for r in results]
    memory = [r["peak_memory_mb"] for r in results]
    plt.bar(names, memory)
    plt.title("Peak Memory Usage Comparison")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/memory_usage.png")
    
    # Plot feature count comparison
    plt.figure(figsize=(12, 6))
    names = [r["name"] for r in feature_results if "feature_count" in r]
    counts = [r["feature_count"] for r in feature_results if "feature_count" in r]
    plt.bar(names, counts)
    plt.title("Feature Count Comparison")
    plt.ylabel("Number of Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/feature_count.png")
    
    # For model results, plot accuracy comparison
    if any("accuracy" in r for r in model_results):
        plt.figure(figsize=(12, 6))
        names = [r["name"] for r in model_results if "accuracy" in r]
        accuracy = [r["accuracy"] for r in model_results if "accuracy" in r]
        plt.bar(names, accuracy)
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("reports/model_accuracy.png")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark the stock prediction model pipeline')
    parser.add_argument('--output', type=str, default=f'reports/benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                       help='Output file for benchmark results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=== STOCK PREDICTION MODEL PIPELINE BENCHMARKS ===")
    run_benchmarks(args.output)
