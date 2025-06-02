#!/usr/bin/env python3
"""
Test script to verify all improvements are working correctly.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.data.data_loader import StockDataLoader
        from src.data.feature_engineering import FeatureGenerator
        from src.data.enhanced_features import EnhancedFeatureGenerator
        from src.data.train_test_split import RobustTrainTestSplit
        from src.models.advanced_train import AdvancedModelTrainer
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_feature_engineering():
    """Test enhanced feature engineering."""
    print("\nTesting feature engineering...")
    
    # Check if talib is available
    try:
        import talib
    except ImportError:
        print("⚠ TA-Lib not installed. Enhanced features require TA-Lib.")
        print("  To install on Mac: brew install ta-lib && pip install ta-lib")
        return False
    
    try:
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        dummy_stock = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100),
            'Symbol': 'TEST'
        }, index=dates)
        
        from src.data.enhanced_features import EnhancedFeatureGenerator
        enhancer = EnhancedFeatureGenerator()
        
        # Test individual feature groups
        df = dummy_stock.copy()
        df = enhancer.add_advanced_technical_indicators(df)
        print(f"✓ Technical indicators: {len([c for c in df.columns if c not in dummy_stock.columns])} features added")
        
        df = enhancer.add_volume_indicators(df)
        print(f"✓ Volume indicators working")
        
        df = enhancer.add_volatility_features(df)
        print(f"✓ Volatility features working")
        
        return True
    except Exception as e:
        print(f"✗ Feature engineering error: {e}")
        return False

def test_train_test_split():
    """Test robust train/test splitting."""
    print("\nTesting train/test splitting...")
    try:
        from src.data.train_test_split import RobustTrainTestSplit
        
        # Create dummy data
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        dummy_data = pd.DataFrame({
            'feature1': np.random.randn(len(dates)),
            'feature2': np.random.randn(len(dates)),
            'label': np.random.randint(0, 2, len(dates))
        }, index=dates)
        
        splitter = RobustTrainTestSplit(gap_days=5)
        
        # Test temporal split
        train, test = splitter.temporal_train_test_split(dummy_data)
        print(f"✓ Temporal split: {len(train)} train, {len(test)} test samples")
        
        # Test event-aware split
        train, test = splitter.event_aware_split(dummy_data)
        print(f"✓ Event-aware split working")
        
        # Test walk-forward split
        splits = splitter.walk_forward_split(dummy_data, n_splits=3)
        print(f"✓ Walk-forward split: {len(splits)} splits created")
        
        # Validate no lookahead
        splitter.validate_no_lookahead(train, test)
        print(f"✓ No lookahead bias detected")
        
        return True
    except Exception as e:
        print(f"✗ Train/test split error: {e}")
        return False

def test_model_training():
    """Test advanced model training."""
    print("\nTesting model training...")
    
    # Check for required packages
    missing_packages = []
    for package in ['xgboost', 'lightgbm', 'catboost', 'optuna']:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠ Missing packages: {', '.join(missing_packages)}")
        print("  Install with: pip install " + ' '.join(missing_packages))
        print("  Testing with basic models only...")
    
    try:
        from src.models.advanced_train import AdvancedModelTrainer
        
        # Create dummy data
        X_train = pd.DataFrame(np.random.randn(1000, 10))
        y_train = np.random.randint(0, 2, 1000)
        X_val = pd.DataFrame(np.random.randn(200, 10))
        y_val = np.random.randint(0, 2, 200)
        
        trainer = AdvancedModelTrainer()
        
        # Test preprocessing
        X_train_scaled, X_val_scaled = trainer.preprocess_features(X_train, X_val)
        print(f"✓ Feature preprocessing working")
        
        # Test model configs
        configs = trainer.get_model_configs()
        print(f"✓ {len(configs)} model configurations available")
        
        # Test single model training (quick test with logistic regression)
        result = trainer.train_single_model(
            'logistic_regression', X_train_scaled, y_train, 
            X_val_scaled, y_val, optimization='none'
        )
        print(f"✓ Model training working - AUC: {result['roc_auc']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== TESTING IMPROVEMENTS ===\n")
    
    tests = [
        test_imports,
        test_feature_engineering,
        test_train_test_split,
        test_model_training
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n=== SUMMARY ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All improvements are working correctly!")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 