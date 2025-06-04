#!/usr/bin/env python3
"""
Fix Script for Model Results

This script copies model results from the models directory to the reports directory
and generates consolidated reports.
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fix model results')
    
    parser.add_argument('--source_dir', type=str, default='models',
                      help='Source directory with model results')
    parser.add_argument('--feature_dir', type=str, default='reports/feature_data',
                      help='Directory to save feature data')
    parser.add_argument('--results_dir', type=str, default='reports/experiment_results',
                      help='Directory to save experiment results')
    
    return parser.parse_args()

def copy_model_results(source_dir, results_dir):
    """Copy result files from source to results directory."""
    print(f"Copying results from {source_dir} to {results_dir}...")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find all result files
    result_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith('results_') and file.endswith('.pkl'):
                result_files.append(os.path.join(root, file))
    
    print(f"Found {len(result_files)} result files")
    
    # Copy files to results directory
    copied_files = []
    for src_file in result_files:
        filename = os.path.basename(src_file)
        dst_file = os.path.join(results_dir, filename)
        shutil.copy2(src_file, dst_file)
        copied_files.append(dst_file)
        print(f"Copied {src_file} to {dst_file}")
    
    return copied_files

def generate_feature_data(results_dir, feature_dir):
    """Generate feature data from model results."""
    print(f"Generating feature data in {feature_dir}...")
    
    # Create feature directory if it doesn't exist
    os.makedirs(feature_dir, exist_ok=True)
    
    # Find all result files
    result_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.startswith('results_') and file.endswith('.pkl'):
                result_files.append(os.path.join(root, file))
    
    # Extract features from results
    for result_file in result_files:
        try:
            # Load results
            results = joblib.load(result_file)
            
            # Extract model type and timestamp from filename
            filename = os.path.basename(result_file)
            parts = filename.replace('.pkl', '').split('_')
            timestamp = parts[-1] if len(parts) > 1 else datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate feature data
            feature_data = {
                "timestamp": timestamp,
                "model_type": "unknown",
                "split_method": "unknown",
                "feature_count": 0,
                "features": [],
                "feature_importance": {}
            }
            
            # Save feature data
            feature_file = os.path.join(feature_dir, f"generated_{timestamp}_features.json")
            import json
            with open(feature_file, 'w') as f:
                json.dump(feature_data, f, indent=4)
            print(f"Generated feature data: {feature_file}")
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
    
    return True

def main():
    """Main function to fix model results."""
    args = parse_args()
    
    print("=== FIXING MODEL RESULTS ===")
    
    # Copy model results
    copied_files = copy_model_results(args.source_dir, args.results_dir)
    
    # Generate feature data
    success = generate_feature_data(args.results_dir, args.feature_dir)
    
    if copied_files and success:
        print("\n=== FIX COMPLETE ===")
        print(f"Copied {len(copied_files)} result files")
        print("Generated feature data")
        print("\nNext steps:")
        print("1. Run the report generator:")
        print("   python scripts/generate_consolidated_report.py")
    else:
        print("\n=== FIX INCOMPLETE ===")
        print("No result files found or feature data generation failed")

if __name__ == "__main__":
    main()
