#!/usr/bin/env python3
"""
Performance Report Generator

This script generates comprehensive performance reports for the optimized stock prediction model,
including execution time, memory usage, and model performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import argparse
import glob
from tabulate import tabulate

def load_benchmark_results(file_path):
    """Load benchmark results from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark results file not found: {file_path}")
    
    return pd.read_csv(file_path)

def load_multiple_results(directory="reports"):
    """Load all benchmark results from a directory."""
    files = glob.glob(f"{directory}/benchmark_results_*.csv")
    if not files:
        raise FileNotFoundError(f"No benchmark results found in {directory}")
        
    all_results = []
    for file in files:
        df = pd.read_csv(file)
        # Add date from filename
        date_str = os.path.basename(file).split('_')[2].split('.')[0]
        df['date'] = date_str
        all_results.append(df)
    
    return pd.concat(all_results)

def generate_time_comparison_chart(df, output_file="reports/time_comparison.png"):
    """Generate time comparison chart for feature generation and model training."""
    plt.figure(figsize=(15, 8))
    
    # Filter for feature generation
    feature_df = df[df['name'].str.contains('Baseline|Cache|Parallel|Vectorized|Selection|Optimization')]
    feature_df = feature_df.sort_values('total_time')
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='name', y='total_time', data=feature_df)
    plt.title('Feature Generation Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Filter for model training
    model_df = df[df['name'].str.contains('xgboost|lightgbm|catboost')]
    model_df = model_df.sort_values('train_time')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='name', y='train_time', data=model_df)
    plt.title('Model Training Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_memory_usage_chart(df, output_file="reports/memory_usage.png"):
    """Generate memory usage comparison chart."""
    plt.figure(figsize=(15, 6))
    
    df_sorted = df.sort_values('peak_memory_mb')
    
    sns.barplot(x='name', y='peak_memory_mb', data=df_sorted)
    plt.title('Peak Memory Usage (MB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_feature_count_chart(df, output_file="reports/feature_count.png"):
    """Generate feature count comparison chart."""
    plt.figure(figsize=(15, 6))
    
    # Filter for feature generation results with feature_count
    feature_df = df[df['name'].str.contains('Baseline|Cache|Parallel|Vectorized|Selection|Optimization')]
    feature_df = feature_df[feature_df['feature_count'].notna()]
    feature_df = feature_df.sort_values('feature_count', ascending=False)
    
    sns.barplot(x='name', y='feature_count', data=feature_df)
    plt.title('Feature Count Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_model_performance_chart(df, output_file="reports/model_performance.png"):
    """Generate model performance comparison chart."""
    plt.figure(figsize=(15, 10))
    
    # Filter for model training results
    model_df = df[df['name'].str.contains('xgboost|lightgbm|catboost')]
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='name', y='accuracy', data=model_df)
    plt.title('Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='name', y='f1_score', data=model_df)
    plt.title('F1 Score')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='name', y='precision', data=model_df)
    plt.title('Precision')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='name', y='recall', data=model_df)
    plt.title('Recall')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_time_vs_performance_chart(df, output_file="reports/time_vs_performance.png"):
    """Generate time vs. performance chart for model training."""
    plt.figure(figsize=(15, 6))
    
    # Filter for model training results
    model_df = df[df['name'].str.contains('xgboost|lightgbm|catboost')]
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='train_time', y='accuracy', hue='name', data=model_df, s=100)
    plt.title('Training Time vs. Accuracy')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='train_time', y='f1_score', hue='name', data=model_df, s=100)
    plt.title('Training Time vs. F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_optimization_summary(df):
    """Generate summary of optimization impacts."""
    # Time improvement for feature generation
    baseline_time = df[df['name'] == 'Baseline']['total_time'].values[0]
    optimized_time = df[df['name'] == 'Full Optimization']['total_time'].values[0]
    time_improvement = (baseline_time - optimized_time) / baseline_time * 100
    
    # Memory improvement
    baseline_memory = df[df['name'] == 'Baseline']['peak_memory_mb'].values[0]
    optimized_memory = df[df['name'] == 'Full Optimization']['peak_memory_mb'].values[0]
    memory_improvement = (baseline_memory - optimized_memory) / baseline_memory * 100
    
    # Feature reduction
    baseline_features = df[df['name'] == 'Baseline']['feature_count'].values[0]
    optimized_features = df[df['name'] == 'Full Optimization']['feature_count'].values[0]
    feature_reduction = (baseline_features - optimized_features) / baseline_features * 100
    
    # Early stopping impact
    model_types = ['xgboost', 'lightgbm', 'catboost']
    early_stopping_impacts = []
    
    for model in model_types:
        no_early = df[df['name'] == f'{model} - No Early Stop']
        with_early = df[df['name'] == f'{model} - Early Stop']
        
        if len(no_early) > 0 and len(with_early) > 0:
            time_diff = (no_early['train_time'].values[0] - with_early['train_time'].values[0]) / no_early['train_time'].values[0] * 100
            acc_diff = (with_early['accuracy'].values[0] - no_early['accuracy'].values[0]) / no_early['accuracy'].values[0] * 100
            
            early_stopping_impacts.append({
                'model': model,
                'time_improvement': time_diff,
                'accuracy_impact': acc_diff
            })
    
    summary = {
        'feature_generation': {
            'time_improvement_pct': time_improvement,
            'memory_improvement_pct': memory_improvement,
            'feature_reduction_pct': feature_reduction
        },
        'early_stopping_impact': early_stopping_impacts
    }
    
    return summary

def generate_html_report(df, output_file="reports/performance_report.html"):
    """Generate comprehensive HTML report."""
    # Create charts
    time_chart = generate_time_comparison_chart(df)
    memory_chart = generate_memory_usage_chart(df)
    feature_chart = generate_feature_count_chart(df)
    performance_chart = generate_model_performance_chart(df)
    time_perf_chart = generate_time_vs_performance_chart(df)
    
    # Get optimization summary
    summary = generate_optimization_summary(df)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction Model Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .chart {{ margin: 30px 0; text-align: center; }}
            .chart img {{ max-width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stock Prediction Model Performance Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Optimization Summary</h2>
            <div class="summary">
                <h3>Feature Generation Improvements</h3>
                <ul>
                    <li>Time Improvement: {summary['feature_generation']['time_improvement_pct']:.2f}%</li>
                    <li>Memory Improvement: {summary['feature_generation']['memory_improvement_pct']:.2f}%</li>
                    <li>Feature Reduction: {summary['feature_generation']['feature_reduction_pct']:.2f}%</li>
                </ul>
                
                <h3>Early Stopping Impact</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Time Improvement</th>
                        <th>Accuracy Impact</th>
                    </tr>
    """
    
    for impact in summary['early_stopping_impact']:
        html_content += f"""
                    <tr>
                        <td>{impact['model']}</td>
                        <td>{impact['time_improvement']:.2f}%</td>
                        <td>{impact['accuracy_impact']:.2f}%</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <h2>Execution Time Comparison</h2>
            <div class="chart">
                <img src="time_comparison.png" alt="Time Comparison">
            </div>
            
            <h2>Memory Usage Comparison</h2>
            <div class="chart">
                <img src="memory_usage.png" alt="Memory Usage">
            </div>
            
            <h2>Feature Count Comparison</h2>
            <div class="chart">
                <img src="feature_count.png" alt="Feature Count">
            </div>
            
            <h2>Model Performance Metrics</h2>
            <div class="chart">
                <img src="model_performance.png" alt="Model Performance">
            </div>
            
            <h2>Time vs. Performance</h2>
            <div class="chart">
                <img src="time_vs_performance.png" alt="Time vs Performance">
            </div>
            
            <h2>Raw Benchmark Data</h2>
    """
    
    # Add table of raw data
    html_content += df.to_html(index=False)
    
    html_content += """
            <div class="footer">
                <p>Stock Prediction Model Optimization Project</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

def generate_markdown_report(df, output_file="reports/performance_report.md"):
    """Generate comprehensive Markdown report."""
    # Get optimization summary
    summary = generate_optimization_summary(df)
    
    # Generate Markdown content
    md_content = f"""# Stock Prediction Model Performance Report

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Summary

### Feature Generation Improvements
- Time Improvement: {summary['feature_generation']['time_improvement_pct']:.2f}%
- Memory Improvement: {summary['feature_generation']['memory_improvement_pct']:.2f}%
- Feature Reduction: {summary['feature_generation']['feature_reduction_pct']:.2f}%

### Early Stopping Impact

| Model | Time Improvement | Accuracy Impact |
|-------|-----------------|----------------|
"""
    
    for impact in summary['early_stopping_impact']:
        md_content += f"| {impact['model']} | {impact['time_improvement']:.2f}% | {impact['accuracy_impact']:.2f}% |\n"
    
    md_content += """
## Execution Time Comparison

![Time Comparison](time_comparison.png)

## Memory Usage Comparison

![Memory Usage](memory_usage.png)

## Feature Count Comparison

![Feature Count](feature_count.png)

## Model Performance Metrics

![Model Performance](model_performance.png)

## Time vs. Performance

![Time vs Performance](time_vs_performance.png)

## Raw Benchmark Data

"""
    
    # Add table of raw data as markdown
    md_content += tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    return output_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate performance reports for stock prediction model optimization')
    parser.add_argument('--input', type=str, default=None,
                       help='Input benchmark results CSV file')
    parser.add_argument('--output_html', type=str, default="reports/performance_report.html",
                       help='Output HTML report file')
    parser.add_argument('--output_md', type=str, default="reports/performance_report.md",
                       help='Output Markdown report file')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Load benchmark results
    if args.input:
        df = load_benchmark_results(args.input)
    else:
        # Try to find the most recent benchmark results
        try:
            df = load_multiple_results()
        except FileNotFoundError:
            print("No benchmark results found. Please run benchmark_performance.py first or provide an input file.")
            return
    
    print("Generating performance reports...")
    
    # Generate charts
    generate_time_comparison_chart(df)
    generate_memory_usage_chart(df)
    generate_feature_count_chart(df)
    generate_model_performance_chart(df)
    generate_time_vs_performance_chart(df)
    
    # Generate reports
    html_file = generate_html_report(df, args.output_html)
    md_file = generate_markdown_report(df, args.output_md)
    
    print(f"HTML report generated: {html_file}")
    print(f"Markdown report generated: {md_file}")

if __name__ == "__main__":
    main()
