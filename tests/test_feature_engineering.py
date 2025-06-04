#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from src.data.feature_engineering import FeatureGenerator

def test_feature_engineering():
    """Test the feature engineering pipeline to ensure it works properly."""
    
    print("Loading data...")
    # Load the data
    stock_data = pd.read_csv('data/raw/stock_data.csv', index_col=0, parse_dates=True)
    sp500_data = pd.read_csv('data/raw/sp500_data.csv', index_col=0, parse_dates=True)
    
    print(f"Original stock data shape: {stock_data.shape}")
    print(f"Original SP500 data shape: {sp500_data.shape}")
    
    # Initialize feature generator
    feature_generator = FeatureGenerator()
    
    print("\nRunning feature engineering...")
    try:
        df = feature_generator.prepare_features(stock_data, sp500_data)
        print(f"✅ SUCCESS! Feature engineering completed successfully.")
        print(f"Final data shape: {df.shape}")
        print(f"Number of symbols: {df['Symbol'].nunique()}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Features: {list(df.columns)}")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        print(f"\nNaN counts:")
        print(nan_counts[nan_counts > 0])
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_feature_engineering() 