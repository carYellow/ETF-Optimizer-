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
        # Make a copy to avoid modifying the original data
        sp500_copy = sp500_data.copy()
        df_copy = df.copy()
        
        # Calculate S&P 500 returns for multiple windows
        sp500_copy['Returns_1d'] = sp500_copy['Close'].pct_change()
        sp500_copy['Returns_5d'] = sp500_copy['Close'].pct_change(5)
        
        # Add additional S&P 500 return windows
        for window in [10, 20, 60]:
            sp500_copy[f'Returns_{window}d'] = sp500_copy['Close'].pct_change(window)
        
        # Ensure we have a date column for merging
        if 'date' not in df_copy.columns and 'Date' not in df_copy.columns:
            df_copy = df_copy.reset_index()
            # Handle different possible index names
            if df_copy.columns[0] == 'index':
                df_copy = df_copy.rename(columns={'index': 'Date'})
            elif df_copy.columns[0] in ['date', 'Date']:
                df_copy = df_copy.rename(columns={df_copy.columns[0]: 'Date'})
            else:
                df_copy['Date'] = df_copy.index
        elif 'date' in df_copy.columns:
            df_copy = df_copy.rename(columns={'date': 'Date'})
        
        if 'Date' not in sp500_copy.columns:
            sp500_copy = sp500_copy.reset_index()
            if sp500_copy.columns[0] == 'index':
                sp500_copy = sp500_copy.rename(columns={'index': 'Date'})
            elif sp500_copy.columns[0] in ['date', 'Date']:
                sp500_copy = sp500_copy.rename(columns={sp500_copy.columns[0]: 'Date'})
            else:
                sp500_copy['Date'] = sp500_copy.index
    
        # Ensure Date columns are datetime and normalized
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        sp500_copy['Date'] = pd.to_datetime(sp500_copy['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        
        # Prepare SP500 columns for merging
        sp500_columns = ['Date'] + [f'Returns_{w}d' for w in [1, 5, 10, 20, 60]]
        sp500_columns = [col for col in sp500_columns if col in sp500_copy.columns]
        
        # Merge with stock data using Date column
        df_merged = pd.merge(
            df_copy,
            sp500_copy[sp500_columns],
            on='Date',
            how='left',  # Use left join to preserve all stock data
            suffixes=('', '_SP500')
        )
        
        # Set Date back as index
        df_merged = df_merged.set_index('Date')
        
        # Calculate relative strength (handle NaN values from merge)
        df_merged['Relative_strength_1d'] = df_merged['Returns_1d'] - df_merged['Returns_1d_SP500']
        df_merged['Relative_strength_5d'] = df_merged['Returns_5d'] - df_merged['Returns_5d_SP500']
        
        # Add historical outperformance indicators
        for window in [1, 5, 10, 20, 60]:
            if f'Returns_{window}d' in df_merged.columns and f'Returns_{window}d_SP500' in df_merged.columns:
                # Create binary feature: 1 if stock outperformed S&P500, 0 otherwise
                df_merged[f'Outperformed_SP500_{window}d'] = (
                    df_merged[f'Returns_{window}d'] > df_merged[f'Returns_{window}d_SP500']
                ).astype(int)
                
                # Add relative outperformance margin 
                df_merged[f'Outperformance_Margin_{window}d'] = (
                    df_merged[f'Returns_{window}d'] - df_merged[f'Returns_{window}d_SP500']
                )
        
        # Calculate rolling beta (20d window)
        window = 20
        def calc_beta(group):
            cov_val = group['Returns_1d'].rolling(window).cov(group['Returns_1d_SP500'])
            var_val = group['Returns_1d_SP500'].rolling(window).var()
            return cov_val / var_val
        
        df_merged['Beta_20d'] = df_merged.groupby('Symbol', group_keys=False).apply(calc_beta).values
        
        return df_merged
    
    def prepare_features(self,
                        stock_data: pd.DataFrame,
                        sp500_data: pd.DataFrame,
                        drop_future_data: bool = True) -> pd.DataFrame:
        """
        Prepare all features for model training.
        
        Args:
            stock_data (pd.DataFrame): Stock data
            sp500_data (pd.DataFrame): S&P 500 index data
            drop_future_data (bool): Whether to drop future-looking features to prevent data leakage
            
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
        
        # Add lagged outperformance features
        for lag in [1, 2, 5]:
            for window in [5, 10, 20]:
                column = f'Outperformed_SP500_{window}d'
                if column in df.columns:
                    df[f'{column}_lag{lag}'] = df.groupby('Symbol')[column].shift(lag)
        
        # Add label: 1 if stock's 5d forward return > S&P 500's 5d forward return, else 0
        # Calculate forward returns (shift negative to look forward)
        df['Future_Returns_5d'] = df.groupby('Symbol')['Returns_5d'].shift(-5)
        df['Future_Returns_5d_SP500'] = df['Returns_5d_SP500'].shift(-5)
        df['Label'] = (df['Future_Returns_5d'] > df['Future_Returns_5d_SP500']).astype(int)
        
        # IMPORTANT: Remove future data to prevent data leakage
        if drop_future_data:
            future_columns = ['Future_Returns_5d', 'Future_Returns_5d_SP500']
            print(f"Dropping future data columns to prevent leakage: {future_columns}")
            df = df.drop(columns=future_columns)
        
        # Intelligent NaN handling - preserve data while ensuring quality
        print(f"Dataset shape before NaN handling: {df.shape}")
        print(f"NaN counts by column:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
        
        # Essential columns that cannot have NaN values
        essential_columns = [
            'Close', 'Volume', 'Returns_1d', 'Returns_1d_SP500'
        ]
        
        # Drop rows where essential columns have NaN values
        initial_rows = len(df)
        df = df.dropna(subset=essential_columns)
        print(f"Rows dropped due to missing essential data: {initial_rows - len(df)}")
        
        # For each stock, keep only rows after we have sufficient historical data
        # This ensures technical indicators have enough data to be meaningful
        min_data_points = max(self.windows) + 5  # e.g., 55 days if max window is 50
        
        def filter_sufficient_data(group):
            """Keep only rows after we have sufficient historical data."""
            if len(group) < min_data_points:
                return pd.DataFrame()  # Return empty if insufficient data
            # Keep rows starting from min_data_points
            return group.iloc[min_data_points-1:]
        
        # Apply filtering by symbol
        df_filtered = df.groupby('Symbol', group_keys=False).apply(filter_sufficient_data)
        print(f"Rows after ensuring sufficient historical data: {len(df_filtered)}")
        
        # Reset index to ensure clean DataFrame structure, handling Symbol column properly
        if len(df_filtered) > 0:
            # Only reset index if we have data
            if 'Symbol' in df_filtered.index.names:
                df_filtered = df_filtered.reset_index()
            elif df_filtered.index.name and df_filtered.index.name != 'Date':
                df_filtered = df_filtered.reset_index(drop=True)
        else:
            # If no data, return empty DataFrame with proper columns
            return pd.DataFrame()
        
        # Fill remaining NaN values with appropriate strategies
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
        
        # Forward fill first, then backward fill for any remaining NaNs
        if 'Symbol' in df_filtered.columns:
            # Process each numeric column individually to avoid the ambiguity error
            for col in numeric_columns:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered.groupby('Symbol')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
        else:
            # Fallback: fill without grouping if Symbol column is not available
            df_filtered[numeric_columns] = df_filtered[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # For any still-remaining NaNs, fill with column median
        for col in numeric_columns:
            if df_filtered[col].isnull().any():
                median_val = df_filtered[col].median()
                df_filtered[col] = df_filtered[col].fillna(median_val)
                print(f"Filled {col} remaining NaNs with median: {median_val}")
        
        print(f"Final dataset shape: {df_filtered.shape}")
        print(f"Remaining NaN values: {df_filtered.isnull().sum().sum()}")
        
        return df_filtered