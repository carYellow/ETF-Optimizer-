#!/usr/bin/env python3
"""
Quick Test Script for Model Training Pipeline

This script runs a quick test of the model training pipeline with minimal configuration
to verify that everything is working correctly.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_quick_test():
    """Run a quick test of the model training pipeline."""
    print("=== RUNNING QUICK TEST OF MODEL TRAINING PIPELINE ===")
    start_time = datetime.now()
    
    # Create necessary result directories first
    os.makedirs('reports/experiment_results', exist_ok=True)
    os.makedirs('reports/feature_data', exist_ok=True)
    os.makedirs('reports/consolidated_reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
      
    # Run the pipeline with minimal configuration
    cmd = [
        sys.executable,
        'scripts/run_model_pipeline.py',
        '--quick_run',
        '--feature_importance_selection',
        '--save_features',
        '--max_features', '20'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n=== TEST COMPLETED IN {duration} ===")
    print(f"Exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ Test passed successfully!")
    else:
        print("❌ Test failed!")

if __name__ == "__main__":
    run_quick_test()
