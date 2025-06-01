import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any
import os

class ModelTrainer:
    def __init__(self, 
                 model_dir: str = "models",
                 n_splits: int = 5,
                 test_size: float = 0.2):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_dir (str): Directory to save models
            n_splits (int): Number of splits for time series cross-validation
            test_size (float): Proportion of data to use for testing
        """
        self.model_dir = model_dir
        self.n_splits = n_splits
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Select features (exclude non-feature columns)
        feature_columns = [col for col in df.columns if col not in [
            'Symbol', 'Label', 'Returns', 'Returns_SP500'
        ]]
        
        X = df[feature_columns]
        y = df['Label']
        
        return X, y
    
    def train_test_split(self, 
                        X: pd.DataFrame,
                        y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform time-based train/test split.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test sets
        """
        # Calculate split index
        split_idx = int(len(X) * (1 - self.test_size))
        
        # Split data
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'feature_importance': feature_importance
        }
    
    def evaluate(self, 
                X_test: pd.DataFrame,
                y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self):
        """
        Save the trained model and scaler.
        """
        # Save model
        joblib.dump(self.model, os.path.join(self.model_dir, 'model.joblib'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
    
    def load_model(self):
        """
        Load the trained model and scaler.
        """
        # Load model
        self.model = joblib.load(os.path.join(self.model_dir, 'model.joblib'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train and evaluate the model.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            
        Returns:
            Dict[str, Any]: Training and evaluation results
        """
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        
        # Train model
        training_results = self.train(X_train, y_train)
        
        # Evaluate model
        evaluation_metrics = self.evaluate(X_test, y_test)
        
        # Save model
        self.save_model()
        
        return {
            'training_results': training_results,
            'evaluation_metrics': evaluation_metrics
        } 