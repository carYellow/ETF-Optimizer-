"""
Feature Tracking Module

This module provides utilities for tracking, saving, and analyzing features used in model training.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class FeatureTracker:
    """Tracks and saves feature data during model training."""
    
    def __init__(self, storage_dir: str = 'reports/feature_data'):
        """
        Initialize the feature tracker.
        
        Args:
            storage_dir: Directory to store feature data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def save_features(self, 
                     feature_names: List[str], 
                     feature_importance: Dict[str, float] = None,
                     model_type: str = 'unknown',
                     split_method: str = 'unknown',
                     selected_features: List[str] = None,
                     selection_method: str = None,
                     additional_metrics: Dict[str, Any] = None):
        """
        Save feature data to disk.
        
        Args:
            feature_names: List of feature names
            feature_importance: Dictionary mapping feature names to importance scores
            model_type: Type of model used
            split_method: Train/test split method used
            selected_features: List of features selected for training
            selection_method: Method used for feature selection
            additional_metrics: Additional metrics to save
        """
        filename = f"{model_type}_{split_method}_{self.timestamp}_features.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Prepare data for storage
        data = {
            'timestamp': self.timestamp,
            'model_type': model_type,
            'split_method': split_method,
            'feature_count': len(feature_names),
            'features': feature_names,
        }
        
        # Add feature importance if available
        if feature_importance:
            data['feature_importance'] = feature_importance
        
        # Add selected features if available
        if selected_features:
            data['selected_features'] = selected_features
            data['selected_feature_count'] = len(selected_features)
            data['selection_method'] = selection_method
        
        # Add additional metrics if available
        if additional_metrics:
            data.update(additional_metrics)
        
        # Save to disk
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Feature data saved to {filepath}")
        return filepath
    
    def format_feature_importance(self, model, feature_names):
        """
        Extract and format feature importance from a trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Different models store feature importance differently
        if hasattr(model, 'feature_importances_'):
            # Random Forest, XGBoost, LightGBM
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importance'):
            # CatBoost
            importances = model.feature_importance()
        else:
            return None
        
        # Create dictionary of feature importance
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            if i < len(importances):
                feature_importance[feature] = float(importances[i])
        
        # Sort by importance
        feature_importance = {k: v for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=True
        )}
        
        return feature_importance
    
    def analyze_feature_importance(self, feature_importance, top_n=20):
        """
        Analyze feature importance and generate insights.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with feature importance analysis
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        
        # Calculate total importance
        total_importance = sum(feature_importance.values())
        
        # Get top N features
        top_features = sorted_features[:top_n]
        
        # Calculate percentage of total importance for top features
        top_importance_pct = sum(imp for _, imp in top_features) / total_importance * 100
        
        # Categorize features
        feature_categories = self._categorize_features([f for f, _ in sorted_features])
        
        # Generate analysis
        analysis = {
            'top_features': dict(top_features),
            'top_importance_percentage': top_importance_pct,
            'feature_category_distribution': feature_categories
        }
        
        return analysis
    
    def _categorize_features(self, features):
        """
        Categorize features based on their names.
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary with counts of features in each category
        """
        categories = {
            'price': 0,
            'volume': 0,
            'technical': 0,
            'fundamental': 0,
            'sentiment': 0,
            'market': 0,
            'other': 0
        }
        
        for feature in features:
            if any(term in feature.lower() for term in ['price', 'close', 'open', 'high', 'low']):
                categories['price'] += 1
            elif 'volume' in feature.lower():
                categories['volume'] += 1
            elif any(term in feature.lower() for term in ['rsi', 'macd', 'ema', 'sma', 'bollinger']):
                categories['technical'] += 1
            elif any(term in feature.lower() for term in ['pe', 'eps', 'revenue', 'earnings']):
                categories['fundamental'] += 1
            elif any(term in feature.lower() for term in ['sentiment', 'news', 'social']):
                categories['sentiment'] += 1
            elif any(term in feature.lower() for term in ['sp500', 'market', 'index']):
                categories['market'] += 1
            else:
                categories['other'] += 1
        
        return categories
