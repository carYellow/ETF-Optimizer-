import pandas as pd
import numpy as np
from typing import List, Optional
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class FeatureGenerator:
    def __init__(self, windows: List[int] = [5, 10, 20, 50]):
        """
        Initialize the FeatureGenerator.
        
        Args:
            windows (List[int]): List of windows for moving averages
        """
        self.windows = windows
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Dataframe with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Ensure we have a proper index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        # Reset index to avoid duplicate labels
        df = df.reset_index().rename(columns={'index': 'date'})
        
        # Calculate 1-day returns first
        df['Returns_1d'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change())
        # Add log returns
        df['Log_Return_1d'] = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        
        # Calculate returns for other windows
        for window in self.windows:
            if window > 1:
                df[f'Returns_{window}d'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(window))
                df[f'Log_Return_{window}d'] = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x / x.shift(window)))
        
        # Calculate volatility using the 1-day returns
        for window in self.windows:
            df[f'Volatility_{window}d'] = df.groupby('Symbol')['Returns_1d'].transform(
                lambda x: x.rolling(window).std()
            )
        
        # Add volume features
        df['Volume_MA_5'] = df.groupby('Symbol')['Volume'].transform(
            lambda x: x.rolling(5).mean()
        )
        df['Volume_MA_20'] = df.groupby('Symbol')['Volume'].transform(
            lambda x: x.rolling(20).mean()
        )
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
        # Add volume delta
        df['Volume_Delta'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.diff())
        # Add VWAP
        vwap = df.groupby('Symbol').apply(lambda x: VolumeWeightedAveragePrice(
            high=x['High'], low=x['Low'], close=x['Close'], volume=x['Volume']).volume_weighted_average_price())
        vwap = vwap.reset_index(level=0, drop=True)
        df['VWAP'] = vwap
        
        # Add price momentum indicators
        for window in self.windows:
            df[f'Price_MA_{window}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'Price_Std_{window}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window).std()
            )
            df[f'Price_Momentum_{window}d'] = df['Close'] / df[f'Price_MA_{window}d'] - 1
            # Add z-score
            df[f'Price_Zscore_{window}d'] = (df['Close'] - df[f'Price_MA_{window}d']) / df[f'Price_Std_{window}d']
        
        # Add momentum for 3, 5, 10 days
        for window in [3, 5, 10]:
            df[f'Momentum_{window}d'] = df.groupby('Symbol')['Close'].transform(lambda x: x / x.shift(window) - 1)
        
        # Add RSI
        for window in self.windows:
            delta = df.groupby('Symbol')['Close'].transform(lambda x: x.diff())
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = df.groupby('Symbol')[gain.name].transform(
                lambda x: x.rolling(window).mean()
            )
            avg_loss = df.groupby('Symbol')[loss.name].transform(
                lambda x: x.rolling(window).mean()
            )
            
            rs = avg_gain / avg_loss
            df[f'RSI_{window}d'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        def ewm_mean(x, span):
            return x.ewm(span=span, adjust=False).mean()
        
        exp1 = df.groupby('Symbol')['Close'].transform(lambda x: ewm_mean(x, 12))
        exp2 = df.groupby('Symbol')['Close'].transform(lambda x: ewm_mean(x, 26))
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df.groupby('Symbol')['MACD'].transform(lambda x: ewm_mean(x, 9))
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Add Bollinger Bands
        for window in self.windows:
            df[f'BB_Middle_{window}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'BB_Std_{window}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window).std()
            )
            df[f'BB_Upper_{window}d'] = df[f'BB_Middle_{window}d'] + (df[f'BB_Std_{window}d'] * 2)
            df[f'BB_Lower_{window}d'] = df[f'BB_Middle_{window}d'] - (df[f'BB_Std_{window}d'] * 2)
            df[f'BB_Width_{window}d'] = (df[f'BB_Upper_{window}d'] - df[f'BB_Lower_{window}d']) / df[f'BB_Middle_{window}d']
        
        return df
    
    def add_market_features(self, 
                          df: pd.DataFrame,
                          sp500_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market-related features.
        
        Args:
            df (pd.DataFrame): Stock data
            sp500_data (pd.DataFrame): S&P 500 index data
            
        Returns:
            pd.DataFrame: Dataframe with added market features
        """
        # Calculate S&P 500 returns
        sp500_data['Returns_1d'] = sp500_data['Close'].pct_change()
        sp500_data['Returns_5d'] = sp500_data['Close'].pct_change(5)
        
        # Merge with stock data
        df = pd.merge(
            df,
            sp500_data[['Returns_1d', 'Returns_5d']],
            left_index=True,
            right_index=True,
            suffixes=('', '_SP500')
        )
        
        # Calculate relative strength
        df['Relative_strength_1d'] = df['Returns_1d'] - df['Returns_1d_SP500']
        df['Relative_strength_5d'] = df['Returns_5d'] - df['Returns_5d_SP500']
        
        # Calculate rolling beta (20d window)
        window = 20
        def calc_beta(x):
            return x['Returns_1d'].rolling(window).cov(x['Returns_1d_SP500']) / x['Returns_1d_SP500'].rolling(window).var()
        beta = df.groupby('Symbol', group_keys=False).apply(calc_beta)
        if isinstance(beta, pd.DataFrame):
            beta = beta.iloc[:, 0]
        df['Beta_20d'] = beta
        
        return df
    
    def prepare_features(self,
                        stock_data: pd.DataFrame,
                        sp500_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all features for model training.
        
        Args:
            stock_data (pd.DataFrame): Stock data
            sp500_data (pd.DataFrame): S&P 500 index data
            
        Returns:
            pd.DataFrame: Dataframe with all features
        """
        # Add technical indicators
        df = self.add_technical_indicators(stock_data)
        
        # Add market features
        df = self.add_market_features(df, sp500_data)
        
        # Ensure index is DatetimeIndex before extracting dayofweek
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df['DayOfWeek'] = df.index.dayofweek
        
        # Add lagged features for returns and volume (1, 2, 3 days)
        for lag in [1, 2, 3]:
            df[f'Returns_1d_lag{lag}'] = df.groupby('Symbol')['Returns_1d'].shift(lag)
            df[f'Volume_lag{lag}'] = df.groupby('Symbol')['Volume'].shift(lag)
        
        # Add label: 1 if stock's 5d forward return > S&P 500's 5d forward return, else 0
        df['Label'] = (df['Returns_5d'] > df['Returns_5d_SP500']).astype(int)
        # Drop rows with NaN values
        df = df.dropna()
        
        return df