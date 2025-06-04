"""
Enhanced Feature Engineering Module

This module adds additional financial features beyond basic technical indicators.
With feature caching, multiprocessing support, and memory optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib
import time
import gc
import os
import hashlib
import pickle
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

class EnhancedFeatureGenerator:
    """Generate advanced financial features for stock prediction."""
    
    def __init__(self, use_cache: bool = True, n_workers: int = None, cache_dir: str = None,
                feature_importance_selection: bool = False, max_features: int = None):
        self.feature_names = []
        self.feature_groups = {}
        self.use_cache = use_cache
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)  # Default to CPU count - 1
        self.cache_dir = cache_dir or Path('cache/features')
        self.feature_importance_selection = feature_importance_selection
        self.max_features = max_features
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Feature cache directory: {self.cache_dir}")
            
    def _get_cache_key(self, symbol: str, df_hash: str) -> str:
        """Generate a unique hash key for the dataframe."""
        return f"{symbol}_{df_hash}"
    
    def _compute_df_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the dataframe to check if it has changed."""
        # Use a subset of columns and first/last rows to compute hash
        # This is much faster than hashing the entire dataframe
        cols_to_hash = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_present = [col for col in cols_to_hash if col in df.columns]
        
        if not cols_present:
            return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        
        # Take first and last 5 rows of relevant columns
        if len(df) > 10:
            sample_df = pd.concat([df[cols_present].head(5), df[cols_present].tail(5)])
        else:
            sample_df = df[cols_present]
            
        # Compute hash
        df_hash = hashlib.md5(pd.util.hash_pandas_object(sample_df).values).hexdigest()
        return df_hash
    
    def _get_cached_features(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Try to load cached features for the symbol."""
        if not self.use_cache:
            return df, False
            
        df_hash = self._compute_df_hash(df)
        cache_key = self._get_cache_key(symbol, df_hash)
        cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_df = pickle.load(f)
                return cached_df, True
            except Exception as e:
                print(f"Error loading cache for {symbol}: {e}")
                
        return df, False
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save the computed features to cache."""
        if not self.use_cache:
            return
            
        df_hash = self._compute_df_hash(df)
        cache_key = self._get_cache_key(symbol, df_hash)
        cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f, protocol=4)  # Protocol 4 for better performance
        except Exception as e:
            print(f"Error saving cache for {symbol}: {e}")
            # If cache fails, continue without it
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        print("Adding advanced technical indicators...")
        start_time = time.time()
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # MACD with signal
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Stochastic Oscillator
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Average Directional Index (ADX)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Money Flow Index
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Williams %R
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Ultimate Oscillator
        df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
        
        tech_features = ['BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position',
                         'MACD', 'MACD_signal', 'MACD_hist', 'STOCH_K', 'STOCH_D',
                         'ADX', 'CCI', 'MFI', 'WILLR', 'ULTOSC']
        
        self.feature_groups['technical'] = tech_features
        print(f"Added {len(tech_features)} technical indicators in {time.time() - start_time:.2f} seconds")
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volume-based indicators."""
        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Chaikin A/D Line
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Rate of Change
        df['VROC'] = df['Volume'].pct_change(10) * 100
        
        # Price Volume Trend
        df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # Ease of Movement
        distance_moved = (df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2
        emv = distance_moved / (df['Volume'] / 1e6 / (df['High'] - df['Low']))
        df['EMV'] = emv.rolling(window=14).mean()
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Average True Range
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Normalized ATR
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # True Range
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
        
        # Historical Volatility (different windows)
        for window in [5, 10, 20, 60]:
            returns = df['Close'].pct_change()
            df[f'HV_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Parkinson Volatility
        df['Parkinson_Vol'] = np.sqrt(
            252 / (4 * np.log(2)) * 
            ((np.log(df['High'] / df['Low'])) ** 2).rolling(window=20).mean()
        )
        
        # Garman-Klass Volatility
        df['GK_Vol'] = np.sqrt(
            252 * (
                0.5 * (np.log(df['High'] / df['Low'])) ** 2 -
                (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open'])) ** 2
            ).rolling(window=20).mean()
        )
        
        return df
    
    def add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features."""
        # Common candlestick patterns
        patterns = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'ENGULFING': talib.CDLENGULFING,
            'MORNING_STAR': talib.CDLMORNINGSTAR,
            'EVENING_STAR': talib.CDLEVENINGSTAR,
            'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
            'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS
        }
        
        for name, func in patterns.items():
            df[f'Pattern_{name}'] = func(df['Open'], df['High'], df['Low'], df['Close'])
            # Convert to binary (pattern present or not)
            df[f'Pattern_{name}'] = (df[f'Pattern_{name}'] != 0).astype(int)
        
        # Count total patterns
        pattern_cols = [col for col in df.columns if col.startswith('Pattern_')]
        df['Pattern_Count'] = df[pattern_cols].sum(axis=1)
        
        return df
    
    def add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # High-Low spread
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Close-Open spread
        df['CO_Spread'] = (df['Close'] - df['Open']) / df['Open']
        
        # Volume-weighted average price deviation
        df['VWAP_Deviation'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # Amihud illiquidity measure
        df['Amihud_Illiquidity'] = abs(df['Returns_1d']) / (df['Volume'] * df['Close'] / 1e6)
        
        # Roll's bid-ask spread estimate
        returns = df['Close'].pct_change()
        df['Roll_Spread'] = 2 * np.sqrt(abs(returns.rolling(window=20).cov(returns.shift(1))))
        
        # Kyle's lambda (price impact)
        df['Kyle_Lambda'] = abs(returns) / (df['Volume'] / df['Volume'].rolling(window=20).mean())
        
        return df
    
    def add_regime_indicators(self, df: pd.DataFrame, sp500_data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        # Bull/Bear market indicator (200-day MA)
        sp500_ma200 = sp500_data['Close'].rolling(window=200).mean()
        sp500_data['Bull_Market'] = (sp500_data['Close'] > sp500_ma200).astype(int)
        
        # Merge with stock data
        df = pd.merge(
            df, 
            sp500_data[['Bull_Market']], 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # VIX regimes (if available)
        # High volatility regime: VIX > 20
        # This would require VIX data, placeholder for now
        
        # Market drawdown
        sp500_cummax = sp500_data['Close'].expanding().max()
        sp500_drawdown = (sp500_data['Close'] - sp500_cummax) / sp500_cummax
        sp500_data['Market_Drawdown'] = sp500_drawdown
        
        df = pd.merge(
            df, 
            sp500_data[['Market_Drawdown']], 
            left_index=True, 
            right_index=True, 
            how='left',
            suffixes=('', '_Market')
        )
        
        return df
    
    def add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: Index is not DatetimeIndex. Converting index to DatetimeIndex.")
            try:
                # Try to convert if index contains datetime-like strings
                df.index = pd.to_datetime(df.index)
            except:
                print("Error: Cannot convert index to DatetimeIndex. Skipping time-based features.")
                return df
        
        # Day of week
        df['DayOfWeek'] = df.index.dayofweek
        
        # Month of year
        df['Month'] = df.index.month
        
        # Quarter
        df['Quarter'] = df.index.quarter
        
        # Is month end/start
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsMonthStart'] = df.index.is_month_start.astype(int)
        
        # Is quarter end/start
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        df['IsQuarterStart'] = df.index.is_quarter_start.astype(int)
        
        # Trading days from month start - safely handle groupby
        try:
            df['TradingDayOfMonth'] = df.groupby([df.index.year, df.index.month]).cumcount() + 1
        except AttributeError:
            # Fallback implementation if groupby fails
            print("Warning: Using fallback method for TradingDayOfMonth calculation")
            df['year_month'] = df.index.to_period('M')
            df['TradingDayOfMonth'] = df.groupby('year_month').cumcount() + 1
            df.drop('year_month', axis=1, inplace=True)
        
        return df
    
    def add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features (stock vs market/sector)."""
        # This requires sector data, placeholder for relative strength
        # Calculate rolling relative strength
        if 'Returns_1d_SP500' in df.columns:
            df['RelativeStrength_5d'] = (
                df['Returns_1d'].rolling(window=5).mean() / 
                df['Returns_1d_SP500'].rolling(window=5).mean()
            )
            df['RelativeStrength_20d'] = (
                df['Returns_1d'].rolling(window=20).mean() / 
                df['Returns_1d_SP500'].rolling(window=20).mean()
            )
            
            # Relative volume
            df['RelativeVolume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        return df
    
    def _process_symbol(self, symbol: str, symbol_df: pd.DataFrame) -> pd.DataFrame:
        """Process a single symbol to generate features. Used for parallel processing."""
        if len(symbol_df) < 50:  # Skip if not enough data
            return None
            
        # Check cache first
        cached_df, cache_hit = self._get_cached_features(symbol, symbol_df)
        if cache_hit:
            return cached_df
            
        # Generate features
        try:
            # Convert to float32 right at the beginning to reduce memory usage
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in symbol_df.columns:
                    symbol_df[col] = symbol_df[col].astype(np.float32)
            
            # Add vectorized price features (more efficient)
            self.add_vectorized_price_features(symbol_df)
            
            # Add traditional technical indicators
            self.add_advanced_technical_indicators(symbol_df)
            self.add_volume_indicators(symbol_df)
            self.add_volatility_features(symbol_df)
            self.add_pattern_recognition(symbol_df)

            # Ensure 0 is a valid category for all categorical columns before filling with 0
            for col in symbol_df.select_dtypes(include=['category']).columns:
                if 0 not in symbol_df[col].cat.categories:
                    symbol_df[col] = symbol_df[col].cat.add_categories([0])

            # Handle potential NaN values
            symbol_df = symbol_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Save to cache
            self._save_to_cache(symbol, symbol_df)
            
            return symbol_df
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None
    
    def _process_symbol_wrapper(self, args):
        """Wrapper function for processing a symbol that can be pickled."""
        symbol, symbol_df = args
        return self._process_symbol(symbol, symbol_df)
    
    def generate_all_features(self, df: pd.DataFrame, sp500_data: Optional[pd.DataFrame] = None, 
                              use_parallel: bool = True) -> pd.DataFrame:
        """Generate all enhanced features."""
        total_start = time.time()
        
        # To avoid modifying the original
        df = df.copy()
        
        # Optimize dtypes to reduce memory usage
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Process symbols in batches to reduce memory usage
        symbols = df['Symbol'].unique()
        total_symbols = len(symbols)
        
        print(f"Generating enhanced features for {total_symbols} symbols...")
        print(f"Cache {'enabled' if self.use_cache else 'disabled'}, Parallel processing {'enabled' if use_parallel else 'disabled'}")
        
        # Track cache hits
        cache_hits = 0
        
        # Process in batches of 50 symbols
        batch_size = 50
        all_dfs = []
        
        for i in range(0, total_symbols, batch_size):
            batch_symbols = symbols[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_symbols+batch_size-1)//batch_size} ({len(batch_symbols)} symbols)")
            
            batch_df = df[df['Symbol'].isin(batch_symbols)]
            batch_results = []
            
            # Use multiprocessing to process symbols in parallel
            if use_parallel and self.n_workers > 1:
                symbol_dfs = []
                
                for symbol in batch_symbols:
                    symbol_df = batch_df[batch_df['Symbol'] == symbol].copy()
                    symbol_dfs.append((symbol, symbol_df))
                
                # Process symbols in parallel with a top-level function that can be pickled
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    results = list(tqdm(
                        executor.map(self._process_symbol_wrapper, symbol_dfs),
                        total=len(symbol_dfs),
                        desc="Processing symbols in parallel"
                    ))
                
                # Filter out None results and add to batch_results
                batch_results.extend([r for r in results if r is not None])
                
                # Clean up to free memory
                del symbol_dfs, results
                gc.collect()
            else:
                # Sequential processing
                for symbol in tqdm(batch_symbols, desc="Symbols"):
                    # Get data for this symbol
                    symbol_df = batch_df[batch_df['Symbol'] == symbol].copy()
                    
                    if len(symbol_df) < 50:  # Skip if not enough data
                        continue
                    
                    # Check cache first
                    cached_df, cache_hit = self._get_cached_features(symbol, symbol_df)
                    if cache_hit:
                        cache_hits += 1
                        batch_results.append(cached_df)
                        continue
                    
                    # Add technical indicators
                    self.add_advanced_technical_indicators(symbol_df)
                    self.add_volume_indicators(symbol_df)
                    self.add_volatility_features(symbol_df)
                    self.add_pattern_recognition(symbol_df)

                    # Ensure 0 is a valid category for all categorical columns before filling with 0
                    for col in symbol_df.select_dtypes(include=['category']).columns:
                        if 0 not in symbol_df[col].cat.categories:
                            symbol_df[col] = symbol_df[col].cat.add_categories([0])

                    # Handle potential NaN values
                    symbol_df = symbol_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                    # Save to cache
                    self._save_to_cache(symbol, symbol_df)
                    
                    # Add to list
                    batch_results.append(symbol_df)
                    
                    # Force garbage collection to free memory
                    del symbol_df
                    gc.collect()
            
            # Combine batch results
            if batch_results:
                batch_combined = pd.concat(batch_results)
                all_dfs.append(batch_combined)
                
                # Clean up to free memory
                del batch_results, batch_combined
                gc.collect()
        
        if self.use_cache:
            print(f"Cache hits: {cache_hits}/{total_symbols} symbols ({cache_hits/total_symbols*100:.1f}%)")
        
        # Combine all dataframes
        print("Combining all processed data...")
        df = pd.concat(all_dfs)
        del all_dfs
        gc.collect()
        
        # Add market-wide features
        if sp500_data is not None:
            print("Adding market microstructure features...")
            df = self.add_market_microstructure(df)
            
            print("Adding regime indicators...")
            df = self.add_regime_indicators(df, sp500_data)
            
            print("Adding cross-sectional features...")
            df = self.add_cross_sectional_features(df)
        
        # Add time-based features
        print("Adding time-based features...")
        df = self.add_time_based_features(df)
        
        # Collect feature names
        feature_cols = [col for col in df.columns if col not in ['Symbol', 'Label', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        self.feature_names = feature_cols
        
        # Organize features into groups
        self.feature_groups = self.get_feature_groups()
        self.feature_groups['all'] = feature_cols
        
        # Handle remaining NaN values
        df = df.fillna(0)
        
        # Convert to float32 to reduce memory usage
        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Perform feature selection if enabled
        if self.feature_importance_selection and 'Label' in df.columns:
            df = self.select_features_by_importance(df)
        
        print(f"Enhanced feature generation complete - {len(self.feature_names)} total features")
        print(f"Total enhanced feature generation time: {time.time() - total_start:.2f} seconds")
        
        return df
    
    def add_vectorized_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features using vectorized operations for better performance."""
        print("Adding vectorized price features...")
        start_time = time.time()
        
        # Calculate returns more efficiently
        # Use pct_change with vectorized operations instead of loops
        for period in [1, 2, 3, 5, 10, 20, 30]:
            col_name = f'Returns_{period}d'
            df[col_name] = df.groupby('Symbol')['Close'].pct_change(period).astype(np.float32)
        
        # Calculate rolling volatility more efficiently
        for window in [5, 10, 20, 30]:
            col_name = f'Volatility_{window}d'
            # Use pandas rolling with vectorized operations
            df[col_name] = df.groupby('Symbol')['Close'].pct_change().rolling(window).std().astype(np.float32)
        
        # Price momentum indicators (vectorized)
        for window in [5, 10, 20, 50]:
            # Price relative to moving average
            ma_col = f'MA_{window}'
            df[ma_col] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window).mean()
            ).astype(np.float32)
            
            # Price to MA ratio - fully vectorized
            df[f'Price_to_MA_{window}'] = (df['Close'] / df[ma_col]).astype(np.float32)
            
            # Rate of change - vectorized
            df[f'ROC_{window}'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.pct_change(window)
            ).astype(np.float32)
        
        # Heikin-Ashi Candles (more accurate price action indicator)
        df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close'])/4
        
        # Calculate previous HA_Open by symbol group
        df['HA_Open'] = df.groupby('Symbol')['HA_Close'].shift(1)
        
        # For the first entry in each group, use regular open
        first_idx = df.groupby('Symbol').head(1).index
        df.loc[first_idx, 'HA_Open'] = df.loc[first_idx, 'Open']
        
        # Complete Heikin-Ashi calculations
        df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        # Calculate trend strength using HA candles
        df['HA_Trend'] = (df['HA_Close'] - df['HA_Open']).astype(np.float32)
        
        # Get list of added features
        price_features = [col for col in df.columns if col.startswith(('Returns_', 'Volatility_', 'MA_', 'Price_to_MA_', 'ROC_', 'HA_'))]
        
        self.feature_groups['price'] = price_features
        print(f"Added {len(price_features)} vectorized price features in {time.time() - start_time:.2f} seconds")
        
        return df
    
    def select_features_by_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select most important features using a tree-based model.
        This reduces dimensionality and can improve both training time and model performance.
        """
        if not self.feature_importance_selection:
            return df
            
        print("\nPerforming feature selection based on importance...")
        start_time = time.time()
        
        # Get feature columns (exclude non-feature columns)
        non_feature_cols = ['Symbol', 'Label', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        # Need both X and y for feature selection
        if 'Label' not in df.columns:
            print("Warning: 'Label' column not found. Skipping feature selection.")
            return df
            
        # Sample the dataset if it's large to speed up selection
        sample_size = min(50000, len(df))
        if len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
            
        X = df_sample[feature_cols].copy()
        y = df_sample['Label'].copy()
        
        # Handle NaNs and infinities for feature selection
        # Replace infinities with NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Calculate median for each column
        medians = X.median()
        
        # Fill NaNs with median values
        X = X.fillna(medians)
        
        # Check for any remaining problematic values
        if X.isnull().any().any():
            print("Warning: Still have NaN values after filling. Filling with zeros.")
            X = X.fillna(0)
            
        # Count number of fixed values
        infinite_count = df_sample[feature_cols].isin([np.inf, -np.inf]).sum().sum()
        nan_count = df_sample[feature_cols].isna().sum().sum()
        print(f"Fixed {infinite_count} infinite values and {nan_count} NaN values for feature selection")
        
        # Convert data to float32 for faster processing
        X = X.astype(np.float32)
        
        # Define a lightweight model for feature selection
        selector_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            n_jobs=min(4, self.n_workers),  # Limit threads for this task
            random_state=42,
            verbosity=0
        )
        
        # Define the max number of features to select
        max_features = self.max_features or int(len(feature_cols) * 0.5)  # Default to 50% of features
        
        # Create selector
        selector = SelectFromModel(
            estimator=selector_model, 
            max_features=max_features,
            threshold='median'  # Use median as threshold for feature importance
        )
        
        try:
            # Fit selector
            selector.fit(X, y)
            
            # Get selected feature mask and names
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Keep only selected features and non-feature columns
            keep_columns = selected_features + [col for col in df.columns if col in non_feature_cols]
            df_reduced = df[keep_columns]
            
            reduction_pct = (1 - len(selected_features) / len(feature_cols)) * 100
            print(f"Feature selection complete: Reduced from {len(feature_cols)} to {len(selected_features)} features ({reduction_pct:.1f}% reduction)")
            print(f"Feature selection took {time.time() - start_time:.2f} seconds")
            
            # Update feature names and groups
            self.feature_names = selected_features
            
            # Update feature groups to only include selected features
            for group, features in self.feature_groups.items():
                self.feature_groups[group] = [f for f in features if f in selected_features]
                
            return df_reduced
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            print("Continuing with all features")
            return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by groups."""
        groups = {
            'technical': [],
            'volume': [],
            'volatility': [],
            'patterns': [],
            'microstructure': [],
            'regime': [],
            'time': [],
            'cross_sectional': []
        }
        
        for feature in self.feature_names:
            if any(ind in feature for ind in ['MACD', 'RSI', 'SMA', 'EMA', 'BB_', 'STOCH', 'ADX', 'CCI', 'WILLR', 'ULTOSC']):
                groups['technical'].append(feature)
            elif any(ind in feature for ind in ['Volume', 'OBV', 'AD', 'VROC', 'PVT', 'EMV', 'MFI']):
                groups['volume'].append(feature)
            elif any(ind in feature for ind in ['ATR', 'HV_', 'Vol', 'TRANGE']):
                groups['volatility'].append(feature)
            elif 'Pattern' in feature:
                groups['patterns'].append(feature)
            elif any(ind in feature for ind in ['Spread', 'Illiquidity', 'Lambda', 'VWAP']):
                groups['microstructure'].append(feature)
            elif any(ind in feature for ind in ['Bull', 'Bear', 'Drawdown', 'Regime']):
                groups['regime'].append(feature)
            elif any(ind in feature for ind in ['Day', 'Month', 'Quarter', 'Week']):
                groups['time'].append(feature)
            elif any(ind in feature for ind in ['Relative', 'Cross']):
                groups['cross_sectional'].append(feature)
        
        return groups