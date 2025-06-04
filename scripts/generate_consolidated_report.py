#!/usr/bin/env python3
"""
Generate Consolidated Training Report

This script analyzes feature and model data from multiple training runs
and generates a comprehensive report on model performance and feature importance.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import argparse

# Ensure proper importing of project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directories for data
FEATURE_DATA_DIR = os.path.join('reports', 'feature_data')
RESULTS_DIR = os.path.join('reports', 'experiment_results')
OUTPUT_DIR = os.path.join('reports', 'consolidated_reports')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate consolidated training report')
    
    parser.add_argument('--feature_dir', type=str, default=FEATURE_DATA_DIR,
                      help='Directory containing feature data')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                      help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                      help='Directory to save consolidated report')
    parser.add_argument('--report_id', type=str, default=None,
                      help='Specific report ID to analyze (default: latest)')
    
    return parser.parse_args()

def collect_feature_data(feature_dir):
    """
    Collect all feature data from JSON files.
    
    Args:
        feature_dir: Directory containing feature data files
        
    Returns:
        List of feature data dictionaries
    """
    feature_data = []
    
    for root, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith('_features.json'):
                try:
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Add file info
                    data['filename'] = file
                    data['filepath'] = filepath
                    
                    feature_data.append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    print(f"Collected {len(feature_data)} feature data files")
    return feature_data

def collect_results_data(results_dir, report_id=None):
    """
    Collect results data from experiment reports.
    
    Args:
        results_dir: Directory containing experiment results
        report_id: Specific report ID to analyze (default: latest)
        
    Returns:
        Dictionary with results data
    """
    if report_id:
        report_dir = os.path.join(results_dir, f'report_{report_id}')
    else:
        # Find latest report
        report_dirs = [d for d in os.listdir(results_dir) if d.startswith('report_')]
        if not report_dirs:
            print("No report directories found")
            return {}
        
        report_dirs.sort(reverse=True)
        report_dir = os.path.join(results_dir, report_dirs[0])
        report_id = report_dirs[0].replace('report_', '')
    
    results_data = {
        'report_id': report_id,
        'report_dir': report_dir,
        'tables': {},
        'plots': [],
        'summaries': {}
    }
    
    # Load performance comparison
    perf_csv = os.path.join(report_dir, 'performance_comparison.csv')
    if os.path.exists(perf_csv):
        results_data['tables']['performance'] = pd.read_csv(perf_csv)
    
    # Load performance summary
    summary_json = os.path.join(report_dir, 'performance_summary.json')
    if os.path.exists(summary_json):
        with open(summary_json, 'r') as f:
            results_data['summaries']['performance'] = json.load(f)
    
    # Find plots
    for file in os.listdir(report_dir):
        if file.endswith('.png'):
            results_data['plots'].append(os.path.join(report_dir, file))
    
    # Load top features
    top_features_json = os.path.join(report_dir, 'top_features.json')
    if os.path.exists(top_features_json):
        with open(top_features_json, 'r') as f:
            results_data['summaries']['top_features'] = json.load(f)
    
    return results_data

def analyze_feature_importance(feature_data):
    """
    Analyze feature importance across models and splits.
    
    Args:
        feature_data: List of feature data dictionaries
        
    Returns:
        Dictionary with feature importance analysis
    """
    # Aggregate feature importance
    all_features = {}
    model_features = {}
    
    for data in feature_data:
        model_type = data.get('model_type', 'unknown')
        
        if 'feature_importance' in data:
            # Track by model type
            if model_type not in model_features:
                model_features[model_type] = {}
            
            # Add features to model-specific dict
            for feature, importance in data['feature_importance'].items():
                if feature not in model_features[model_type]:
                    model_features[model_type][feature] = []
                model_features[model_type][feature].append(importance)
                
                # Add to all features dict
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
    
    # Calculate average importance
    avg_importance = {f: np.mean(imps) for f, imps in all_features.items() if imps}
    
    # Calculate model-specific importance
    model_avg_importance = {}
    for model, features in model_features.items():
        model_avg_importance[model] = {f: np.mean(imps) for f, imps in features.items() if imps}
    
    # Get top features overall
    top_features = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Get top features by model
    model_top_features = {}
    for model, features in model_avg_importance.items():
        model_top_features[model] = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {
        'top_features': top_features,
        'model_top_features': model_top_features,
        'feature_count': len(all_features),
        'model_feature_count': {model: len(features) for model, features in model_features.items()}
    }

def analyze_model_performance(results_data):
    """
    Analyze model performance across splits.
    
    Args:
        results_data: Dictionary with results data
        
    Returns:
        Dictionary with model performance analysis
    """
    if 'tables' not in results_data or 'performance' not in results_data['tables']:
        return {}
    
    performance_df = results_data['tables']['performance']
    
    # Calculate overall best model
    if 'split_method' in performance_df.columns:
        # If we have split methods, calculate average performance by model
        model_performance = performance_df.melt(
            id_vars=['split_method'],
            var_name='model',
            value_name='roc_auc'
        ).groupby('model')['roc_auc'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        best_model = model_performance.loc[model_performance['mean'].idxmax()]
        
        # Calculate best split method for each model
        best_splits = {}
        for model in performance_df.columns[1:]:  # Skip split_method column
            best_split_idx = performance_df[model].idxmax()
            best_split = performance_df.iloc[best_split_idx]['split_method']
            best_splits[model] = {
                'split_method': best_split,
                'roc_auc': performance_df.iloc[best_split_idx][model]
            }
    else:
        # Simple case - just find the best model
        best_model_name = performance_df.iloc[:, 1:].max().idxmax()
        best_model = {
            'model': best_model_name,
            'mean': performance_df[best_model_name].max()
        }
        best_splits = {}
    
    return {
        'best_model': best_model.to_dict() if isinstance(best_model, pd.Series) else best_model,
        'best_splits': best_splits
    }

def generate_markdown_report(feature_analysis, performance_analysis, results_data, output_dir):
    """
    Generate a comprehensive markdown report.
    
    Args:
        feature_analysis: Dictionary with feature importance analysis
        performance_analysis: Dictionary with model performance analysis
        results_data: Dictionary with results data
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_id = results_data.get('report_id', 'unknown')
    
    # Start building the report
    md_content = f"""# Stock Prediction Model Training Report

**Report ID:** {report_id}  
**Generated:** {report_time}

## Model Performance Summary

"""
    
    # Add performance summary
    if performance_analysis and 'best_model' in performance_analysis:
        best_model = performance_analysis['best_model']
        md_content += f"### Best Performing Model\n\n"
        md_content += f"- **Model:** {best_model.get('model', 'Unknown')}\n"
        md_content += f"- **Average ROC AUC:** {best_model.get('mean', 0):.4f}\n"
        
        if 'std' in best_model:
            md_content += f"- **Standard Deviation:** {best_model.get('std', 0):.4f}\n"
        if 'min' in best_model:
            md_content += f"- **Min ROC AUC:** {best_model.get('min', 0):.4f}\n"
        if 'max' in best_model:
            md_content += f"- **Max ROC AUC:** {best_model.get('max', 0):.4f}\n"
        
        md_content += "\n"
    
    # Add best split methods for each model
    if performance_analysis and 'best_splits' in performance_analysis and performance_analysis['best_splits']:
        md_content += "### Best Split Method for Each Model\n\n"
        md_content += "| Model | Best Split Method | ROC AUC |\n"
        md_content += "|-------|------------------|--------:|\n"
        
        for model, data in performance_analysis['best_splits'].items():
            md_content += f"| {model} | {data['split_method']} | {data['roc_auc']:.4f} |\n"
        
        md_content += "\n"
    
    # Add performance table
    if 'tables' in results_data and 'performance' in results_data['tables']:
        md_content += "### Performance Comparison Across Split Methods\n\n"
        md_content += results_data['tables']['performance'].to_markdown(index=False)
        md_content += "\n\n"
    
    # Add feature importance analysis
    md_content += "## Feature Importance Analysis\n\n"
    
    if feature_analysis and 'top_features' in feature_analysis:
        md_content += "### Top 20 Most Important Features (Average Across All Models)\n\n"
        md_content += "| Feature | Importance |\n"
        md_content += "|---------|----------:|\n"
        
        for feature, importance in feature_analysis['top_features'].items():
            md_content += f"| {feature} | {importance:.6f} |\n"
        
        md_content += "\n"
    
    # Add model-specific feature importance
    if feature_analysis and 'model_top_features' in feature_analysis:
        md_content += "### Top 10 Features by Model\n\n"
        
        for model, features in feature_analysis['model_top_features'].items():
            md_content += f"#### {model}\n\n"
            md_content += "| Feature | Importance |\n"
            md_content += "|---------|----------:|\n"
            
            for feature, importance in features.items():
                md_content += f"| {feature} | {importance:.6f} |\n"
            
            md_content += "\n"
    
    # Add insights and recommendations
    md_content += "## Insights and Recommendations\n\n"
    
    # Best model recommendation
    if performance_analysis and 'best_model' in performance_analysis:
        best_model = performance_analysis['best_model']
        best_model_name = best_model.get('model', 'Unknown')
        
        md_content += "### Model Selection\n\n"
        md_content += f"Based on the experimental results, **{best_model_name}** is the recommended model "
        md_content += f"with an average ROC AUC of {best_model.get('mean', 0):.4f}. "
        
        # Add split method recommendation if available
        if performance_analysis.get('best_splits') and best_model_name in performance_analysis['best_splits']:
            best_split = performance_analysis['best_splits'][best_model_name]['split_method']
            md_content += f"For optimal performance, use the **{best_split}** train/test split strategy.\n\n"
        else:
            md_content += "\n\n"
    
    # Feature recommendations
    if feature_analysis and 'top_features' in feature_analysis:
        md_content += "### Feature Engineering\n\n"
        md_content += "The following feature engineering recommendations are based on the importance analysis:\n\n"
        
        # Get top 5 features
        top_5 = list(feature_analysis['top_features'].keys())[:5]
        
        md_content += "1. **Focus on Key Features**: The most predictive features are:\n"
        for feature in top_5:
            md_content += f"   - {feature}\n"
        
        md_content += "\n2. **Feature Selection**: Consider using only the top 20-30 features for better model efficiency "
        md_content += "without significant performance loss.\n\n"
        
        # Model-specific feature recommendations
        if 'model_top_features' in feature_analysis:
            md_content += "3. **Model-Specific Feature Sets**: For best results with specific models:\n"
            
            for model, features in feature_analysis['model_top_features'].items():
                top_3 = list(features.keys())[:3]
                md_content += f"   - **{model}**: Focus on {', '.join(top_3)}\n"
    
    # Save the markdown report
    report_path = os.path.join(output_dir, f'consolidated_report_{report_id}.md')
    with open(report_path, 'w') as f:
        f.write(md_content)
    
    print(f"Markdown report saved to {report_path}")
    
    # Also generate HTML version
    try:
        import markdown
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        # Add CSS styling
        html_styled = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Prediction Model Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3, h4 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        html_path = os.path.join(output_dir, f'consolidated_report_{report_id}.html')
        with open(html_path, 'w') as f:
            f.write(html_styled)
        
        print(f"HTML report saved to {html_path}")
    except ImportError:
        print("markdown package not found. HTML report not generated.")

def main():
    """Main function to generate the consolidated report."""
    args = parse_args()
    
    print("=== GENERATING CONSOLIDATED TRAINING REPORT ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect data
    feature_data = collect_feature_data(args.feature_dir)
    results_data = collect_results_data(args.results_dir, args.report_id)
    
    if not feature_data:
        print("No feature data found!")
        return
    
    if not results_data:
        print("No results data found!")
        return
    
    # Analyze data
    feature_analysis = analyze_feature_importance(feature_data)
    performance_analysis = analyze_model_performance(results_data)
    
    # Generate report
    generate_markdown_report(
        feature_analysis,
        performance_analysis,
        results_data,
        args.output_dir
    )
    
    print("=== REPORT GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()
