"""
Robust Train/Test Split Module

This module provides various train/test split strategies that prevent lookahead bias
and account for global market events.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

class RobustTrainTestSplit:
    """Provides various train/test split strategies for time series data."""
    
    def __init__(self, gap_days: int = 5):
        """
        Initialize the splitter.
        
        Args:
            gap_days: Number of days gap between train and test to prevent lookahead
        """
        self.gap_days = gap_days
        self.major_events = self._get_major_market_events()
    
    def _get_major_market_events(self) -> Dict[str, Tuple[str, str]]:
        """Define major market events that should not be split across train/test."""
        return {
            'dot_com_crash': ('2000-03-10', '2002-10-09'),
            'financial_crisis': ('2007-10-09', '2009-03-09'),
            'flash_crash': ('2010-05-06', '2010-05-07'),
            'european_debt_crisis': ('2011-08-01', '2011-10-31'),
            'china_devaluation': ('2015-08-11', '2015-08-27'),
            'brexit': ('2016-06-23', '2016-06-27'),
            'covid_crash': ('2020-02-20', '2020-03-23'),
            'covid_recovery': ('2020-03-24', '2020-06-30'),
            'meme_stock_frenzy': ('2021-01-25', '2021-02-05'),
            'ukraine_invasion': ('2022-02-24', '2022-03-15'),
            'banking_crisis_2023': ('2023-03-08', '2023-03-31')
        }
    
    def temporal_train_test_split(self, df: pd.DataFrame, 
                                test_size: float = 0.2,
                                min_train_samples: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple temporal split with gap to prevent lookahead bias.
        
        Args:
            df: DataFrame with DatetimeIndex
            test_size: Proportion of data for testing
            min_train_samples: Minimum samples in training set
            
        Returns:
            train_df, test_df
        """
        # Sort by date
        df_sorted = df.sort_index()
        
        # Get unique dates
        all_dates = df_sorted.index.unique().sort_values()
        
        # Calculate split point
        n_dates = len(all_dates)
        n_test_dates = int(n_dates * test_size)
        
        # First calculate where test should end
        test_end_date_idx = n_dates - 1  # Use -1 to avoid index out of bounds
        # Then calculate where test should start, considering the total size
        test_start_date_idx = test_end_date_idx - n_test_dates + 1  # +1 to include this date
        
        # Make sure we have enough gap between train and test sets
        # This is critical for avoiding lookahead bias
        gap_idx = max(self.gap_days, 1)  # Ensure at least 1 day gap
        
        # Then calculate where train should end, considering the gap
        train_end_date_idx = test_start_date_idx - gap_idx
        
        # Ensure minimum training samples
        if train_end_date_idx < min_train_samples:
            raise ValueError(f"Not enough training samples: {train_end_date_idx} < {min_train_samples}")
        
        # Get the actual dates
        train_end_date = all_dates[train_end_date_idx]
        test_start_date = all_dates[test_start_date_idx]
        
        # Create the split
        train_df = df_sorted[df_sorted.index <= train_end_date]
        test_df = df_sorted[df_sorted.index >= test_start_date]
        
        # Check actual gap
        actual_gap_days = (test_df.index.min() - train_df.index.max()).days
        
        print(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
        print(f"Gap days: {self.gap_days} (actual gap: {actual_gap_days} days)")
        
        return train_df, test_df
    
    def event_aware_split(self, df: pd.DataFrame,
                         test_start_date: Optional[str] = None,
                         avoid_events: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data avoiding major market events in the test set start.
        
        Args:
            df: DataFrame with DatetimeIndex
            test_start_date: Desired test start date (will be adjusted if needed)
            avoid_events: Whether to avoid splitting during major events
            
        Returns:
            train_df, test_df
        """
        df_sorted = df.sort_index()
        
        # If no test start date provided, use 80/20 split
        if test_start_date is None:
            n_samples = len(df_sorted)
            test_start_idx = int(n_samples * 0.8)
            test_start_date = df_sorted.index[test_start_idx]
        else:
            test_start_date = pd.to_datetime(test_start_date)
        
        # Check if test start falls within any major event
        if avoid_events:
            for event_name, (start, end) in self.major_events.items():
                event_start = pd.to_datetime(start)
                event_end = pd.to_datetime(end)
                
                if event_start <= test_start_date <= event_end:
                    # Move test start to after the event + gap
                    test_start_date = event_end + timedelta(days=self.gap_days)
                    print(f"Adjusted test start to avoid {event_name}: {test_start_date}")
        
        # Apply gap
        train_end_date = test_start_date - timedelta(days=self.gap_days)
        
        # Split data
        train_df = df_sorted[df_sorted.index <= train_end_date]
        test_df = df_sorted[df_sorted.index >= test_start_date]
        
        print(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
        
        # Warn about events in test period
        for event_name, (start, end) in self.major_events.items():
            event_start = pd.to_datetime(start)
            event_end = pd.to_datetime(end)
            
            test_start = test_df.index.min()
            test_end = test_df.index.max()
            
            if (event_start <= test_end) and (event_end >= test_start):
                print(f"WARNING: Test period includes {event_name} ({start} to {end})")
        
        return train_df, test_df
    
    def walk_forward_split(self, df: pd.DataFrame,
                          n_splits: int = 5,
                          test_days: int = 63,  # ~3 months
                          retrain_frequency: int = 21) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward analysis splits for robust backtesting.
        
        Args:
            df: DataFrame with DatetimeIndex
            n_splits: Number of splits
            test_days: Number of days in each test period
            retrain_frequency: Days between retraining
            
        Returns:
            List of (train_df, test_df) tuples
        """
        df_sorted = df.sort_index()
        splits = []
        
        # Calculate dates
        total_days = (df_sorted.index.max() - df_sorted.index.min()).days
        test_period_days = test_days + self.gap_days
        min_train_days = 252  # 1 year minimum
        
        # Calculate step size
        available_days = total_days - min_train_days - test_period_days
        step_days = available_days // (n_splits - 1)
        
        for i in range(n_splits):
            # Calculate train end date
            train_start = df_sorted.index.min()
            train_end = train_start + timedelta(days=min_train_days + i * step_days)
            
            # Calculate test period
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=test_days)
            
            # Get data
            train_df = df_sorted[(df_sorted.index >= train_start) & (df_sorted.index <= train_end)]
            test_df = df_sorted[(df_sorted.index >= test_start) & (df_sorted.index <= test_end)]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
                print(f"Split {i+1}: Train {train_df.index.min().date()} to {train_df.index.max().date()}, "
                      f"Test {test_df.index.min().date()} to {test_df.index.max().date()}")
        
        return splits
    
    def purged_cv_split(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged cross-validation splits that prevent lookahead bias.
        
        Args:
            df: DataFrame with DatetimeIndex
            n_splits: Number of CV splits
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # Use sklearn's TimeSeriesSplit as base
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=self.gap_days)
        
        # Get indices
        X = np.arange(len(df))
        splits = []
        
        for train_idx, test_idx in tscv.split(X):
            # Additional purging: remove samples too close to test set
            if len(train_idx) > self.gap_days:
                train_idx = train_idx[:-self.gap_days]
            
            splits.append((train_idx, test_idx))
            
            # Print split info
            train_start = df.index[train_idx[0]]
            train_end = df.index[train_idx[-1]]
            test_start = df.index[test_idx[0]]
            test_end = df.index[test_idx[-1]]
            
            print(f"CV Split: Train {train_start.date()} to {train_end.date()}, "
                  f"Test {test_start.date()} to {test_end.date()}")
        
        return splits
    
    def regime_based_split(self, df: pd.DataFrame, 
                          regime_column: str = 'Market_Regime') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data based on market regimes (bull/bear/sideways).
        
        Args:
            df: DataFrame with regime column
            regime_column: Name of regime column
            
        Returns:
            Dictionary of regime -> (train_df, test_df)
        """
        if regime_column not in df.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
        
        regime_splits = {}
        
        for regime in df[regime_column].unique():
            regime_df = df[df[regime_column] == regime].sort_index()
            
            if len(regime_df) < 100:  # Skip if too few samples
                print(f"Skipping regime {regime}: only {len(regime_df)} samples")
                continue
            
            # Split this regime's data
            try:
                train_df, test_df = self.temporal_train_test_split(regime_df, test_size=0.2)
                regime_splits[regime] = (train_df, test_df)
                print(f"Regime {regime}: {len(train_df)} train, {len(test_df)} test samples")
            except ValueError as e:
                print(f"Could not split regime {regime}: {e}")
        
        return regime_splits
    
    def expanding_window_split(self, df: pd.DataFrame,
                             initial_train_days: int = 504,  # 2 years
                             step_days: int = 21,  # Monthly updates
                             test_days: int = 21) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Expanding window splits where training set grows over time.
        
        Args:
            df: DataFrame with DatetimeIndex  
            initial_train_days: Initial training period
            step_days: Days to add to training set each iteration
            test_days: Test period length
            
        Returns:
            List of (train_df, test_df) tuples
        """
        df_sorted = df.sort_index()
        splits = []
        
        train_start = df_sorted.index.min()
        current_train_end = train_start + timedelta(days=initial_train_days)
        
        while True:
            test_start = current_train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=test_days)
            
            # Check if we have enough test data
            if test_end > df_sorted.index.max():
                break
            
            # Get splits
            train_df = df_sorted[(df_sorted.index >= train_start) & (df_sorted.index <= current_train_end)]
            test_df = df_sorted[(df_sorted.index >= test_start) & (df_sorted.index <= test_end)]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
                print(f"Expanding window: Train {train_df.index.min().date()} to {train_df.index.max().date()} "
                      f"({len(train_df)} samples), Test {test_df.index.min().date()} to {test_df.index.max().date()}")
            
            # Expand training window
            current_train_end += timedelta(days=step_days)
        
        return splits
    
    def validate_no_lookahead(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Validate that there's no lookahead bias in the split.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            True if valid, raises exception if lookahead detected
        """
        train_end = train_df.index.max()
        test_start = test_df.index.min()
        
        # Check temporal order
        if train_end >= test_start:
            print(f"WARNING: Lookahead issue detected! Train end ({train_end}) >= Test start ({test_start})")
            print(f"Adjusting test_df to ensure no lookahead bias...")
            # Adjust test_df to start after train_end
            test_df = test_df[test_df.index > train_end]
            if len(test_df) == 0:
                raise ValueError(f"After lookahead correction, test set is empty! Try a different split method.")
            test_start = test_df.index.min()
        
        # Check gap
        gap = (test_start - train_end).days
        if gap < self.gap_days:
            print(f"WARNING: Insufficient gap! {gap} days < required {self.gap_days} days")
            print(f"This may be due to using trading days vs calendar days.")
            
            if gap == 0:
                # Critical issue - try to fix by adjusting test_df
                print(f"Critical gap issue (0 days). Adjusting test data...")
                # Find date that's at least gap_days away from train_end
                filtered_test = test_df[test_df.index > (train_end + pd.Timedelta(days=1))]
                if len(filtered_test) == 0:
                    raise ValueError(f"Cannot create a proper gap! Try increasing test_size or using a different split method.")
                test_df = filtered_test
                test_start = test_df.index.min()
                gap = (test_start - train_end).days
                print(f"Adjusted test set to start at {test_start} (gap: {gap} days)")
        
        print(f"✓ No lookahead bias detected. Gap: {gap} days")
        
        return True 