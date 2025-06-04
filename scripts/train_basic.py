#!/usr/bin/env python3
"""
Basic training script that demonstrates improvements without requiring all dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureGenerator
from src.data.train_test_split import RobustTrainTestSplit

def main():
    print("=== BASIC STOCK PREDICTION TRAINING ===\n")
    
    # Load data
    print("Loading data...")
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.load_raw_data()
    
    print(f"Loaded {len(stock_data):,} stock records")
    print(f"Loaded {len(sp500_data):,} S&P 500 records")
    
    # Generate features
    print("\nGenerating features...")
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    print(f"Feature dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Use robust train/test split
    print("\n=== ROBUST TRAIN/TEST SPLIT ===")
    splitter = RobustTrainTestSplit(gap_days=5)
    
    # Event-aware split
    train_df, test_df = splitter.event_aware_split(df, avoid_events=True)
    
    # Validate no lookahead
    splitter.validate_no_lookahead(train_df, test_df)
    
    # Prepare features
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Label', 'Symbol']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Label']
    X_test = test_df[feature_cols]
    y_test = test_df['Label']
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n=== TRAINING MODELS ===")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - ROC AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Feature importance for Random Forest
    print("\n=== TOP FEATURES (Random Forest) ===")
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:30} {row['importance']:.4f}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Best model: {max(results.items(), key=lambda x: x[1]['auc'])[0]}")
    print(f"Best ROC AUC: {max(results.values(), key=lambda x: x['auc'])['auc']:.4f}")
    
    print("\n✓ Training complete!")
    print("\nKey improvements demonstrated:")
    print("- Event-aware train/test split avoiding major market events")
    print("- 5-day gap between train and test to prevent lookahead bias")
    print("- Proper handling of temporal data")
    print("- Feature importance analysis")
    
    print("\nTo unlock more features:")
    print("1. Run: bash setup_improvements.sh")
    print("2. Then: python train_models.py --use_enhanced_features")

if __name__ == "__main__":
    main() 