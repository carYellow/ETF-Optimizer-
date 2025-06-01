import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from ..data.feature_engineering import FeatureGenerator
from .train import ModelTrainer

class StockPredictor:
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the StockPredictor.
        
        Args:
            model_dir (str): Directory containing trained model
        """
        self.model_trainer = ModelTrainer(model_dir=model_dir)
        self.feature_generator = FeatureGenerator()
        
        # Load trained model
        self.model_trainer.load_model()
    
    def fetch_latest_data(self, 
                         symbol: str,
                         days_back: int = 100) -> pd.DataFrame:
        """
        Fetch latest data for a given stock.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days of historical data to fetch
            
        Returns:
            pd.DataFrame: Latest stock data
        """
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch stock data
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        df['Symbol'] = symbol
        
        return df
    
    def prepare_prediction_data(self,
                              stock_data: pd.DataFrame,
                              sp500_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction.
        
        Args:
            stock_data (pd.DataFrame): Stock data
            sp500_data (pd.DataFrame): S&P 500 index data
            
        Returns:
            pd.DataFrame: Prepared data for prediction
        """
        # Generate features
        df = self.feature_generator.prepare_features(stock_data, sp500_data)
        
        return df
    
    def predict(self,
               symbol: str,
               date: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction for a given stock and date.
        
        Args:
            symbol (str): Stock symbol
            date (str, optional): Date for prediction in YYYY-MM-DD format.
                                If None, uses latest available date.
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Fetch latest data
        stock_data = self.fetch_latest_data(symbol)
        sp500_data = self.fetch_latest_data("^GSPC")
        
        # Prepare data
        df = self.prepare_prediction_data(stock_data, sp500_data)
        
        # If date is provided, filter data
        if date:
            df = df[df.index <= date]
        
        # Get latest data point
        latest_data = df.iloc[-1:].copy()
        
        # Prepare features
        X, _ = self.model_trainer.prepare_features(latest_data)
        
        # Scale features
        X_scaled = self.model_trainer.scaler.transform(X)
        
        # Make prediction
        prediction = self.model_trainer.model.predict(X_scaled)[0]
        probability = self.model_trainer.model.predict_proba(X_scaled)[0][1]
        
        return {
            'symbol': symbol,
            'date': latest_data.index[0].strftime('%Y-%m-%d'),
            'prediction': bool(prediction),
            'probability': float(probability),
            'features': X.iloc[0].to_dict()
        } 