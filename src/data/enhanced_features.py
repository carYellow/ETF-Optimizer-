"""
Enhanced Feature Engineering Module

This module adds additional financial features beyond basic technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib

class EnhancedFeatureGenerator:
    """Generate advanced financial features for stock prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
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
        
        # Trading days from month start
        df['TradingDayOfMonth'] = df.groupby([df.index.year, df.index.month]).cumcount() + 1
        
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
    
    def generate_all_features(self, df: pd.DataFrame, sp500_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate all enhanced features."""
        print("Adding advanced technical indicators...")
        df = self.add_advanced_technical_indicators(df)
        
        print("Adding volume indicators...")
        df = self.add_volume_indicators(df)
        
        print("Adding volatility features...")
        df = self.add_volatility_features(df)
        
        print("Adding pattern recognition...")
        df = self.add_pattern_recognition(df)
        
        print("Adding market microstructure features...")
        df = self.add_market_microstructure(df)
        
        if sp500_data is not None:
            print("Adding regime indicators...")
            df = self.add_regime_indicators(df, sp500_data)
        
        print("Adding time-based features...")
        df = self.add_time_based_features(df)
        
        print("Adding cross-sectional features...")
        df = self.add_cross_sectional_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Symbol', 'Label']]
        
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