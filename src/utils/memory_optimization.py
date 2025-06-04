#!/usr/bin/env python3
"""
Memory Optimization Utilities

This module provides utilities for optimizing memory usage during model training,
including datatype optimization, chunking, and memory monitoring.
"""

import pandas as pd
import numpy as np
import gc
import os
import psutil
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Provides memory optimization utilities."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB
    
    @staticmethod
    def log_memory_usage(label: str) -> None:
        """Log current memory usage with a label."""
        memory_mb = MemoryOptimizer.get_memory_usage()
        logger.info(f"Memory usage at {label}: {memory_mb:.2f} MB")
    
    @staticmethod
    def force_gc() -> None:
        """Force garbage collection and log memory usage before and after."""
        before = MemoryOptimizer.get_memory_usage()
        gc.collect()
        after = MemoryOptimizer.get_memory_usage()
        logger.info(f"Garbage collection: {before:.2f} MB -> {after:.2f} MB (freed {before-after:.2f} MB)")
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Optimize memory usage of a DataFrame by downcasting numeric types
        and using category type for string columns with low cardinality.
        
        Args:
            df: DataFrame to optimize
            verbose: Whether to print memory usage info
            
        Returns:
            Optimized DataFrame
        """
        start_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if verbose:
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Memory usage before optimization: {start_memory:.2f} MB")
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize string columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 0.5 * len(df):  # Convert to category if low cardinality
                df[col] = df[col].astype('category')
        
        # Handle datetime columns - keep as is, they're already memory efficient
        
        end_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        reduction = (start_memory - end_memory) / start_memory * 100
        
        if verbose:
            logger.info(f"Memory usage after optimization: {end_memory:.2f} MB")
            logger.info(f"Memory reduced by {reduction:.2f}%")
        
        return df
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """
        Split a DataFrame into manageable chunks for processing.
        
        Args:
            df: DataFrame to split into chunks
            chunk_size: Size of each chunk in number of rows
            
        Returns:
            List of DataFrame chunks
        """
        return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    @staticmethod
    def process_in_chunks(df: pd.DataFrame, process_func: Callable, 
                          chunk_size: int = 10000, *args, **kwargs) -> pd.DataFrame:
        """
        Process a large DataFrame in chunks and combine results.
        
        Args:
            df: DataFrame to process
            process_func: Function to apply to each chunk (must return a DataFrame)
            chunk_size: Size of each chunk in number of rows
            *args, **kwargs: Additional arguments to pass to process_func
            
        Returns:
            Processed DataFrame
        """
        chunks = MemoryOptimizer.chunk_dataframe(df, chunk_size)
        logger.info(f"Processing DataFrame of shape {df.shape} in {len(chunks)} chunks")
        
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} of size {len(chunk)}")
            result = process_func(chunk, *args, **kwargs)
            results.append(result)
            # Force garbage collection after each chunk
            MemoryOptimizer.force_gc()
        
        # Combine results
        if isinstance(results[0], pd.DataFrame):
            combined = pd.concat(results, axis=0)
            return combined
        else:
            return results
    
    @staticmethod
    def clean_infinity_values(df: pd.DataFrame, replace_with: str = 'median') -> pd.DataFrame:
        """
        Clean infinity values in a DataFrame.
        
        Args:
            df: DataFrame to clean
            replace_with: How to replace infinite values ('median', 'mean', or a specific value)
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace infinities with NaNs
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Count NaN values
        nan_count = df_clean.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Found {nan_count} NaN values (including converted infinities)")
        
        # Replace NaNs
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                if replace_with == 'median':
                    fill_value = df_clean[col].median()
                elif replace_with == 'mean':
                    fill_value = df_clean[col].mean()
                else:
                    fill_value = replace_with
                
                df_clean[col] = df_clean[col].fillna(fill_value)
        
        return df_clean
    
    @staticmethod
    def memory_usage_by_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate memory usage by column in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        memory_usage_df = pd.DataFrame({
            'Column': memory_usage.index,
            'Memory_MB': memory_usage.values / 1024 / 1024,
            'Memory_Percent': memory_usage.values / memory_usage.sum() * 100
        }).sort_values('Memory_MB', ascending=False)
        
        return memory_usage_df
    
    @staticmethod
    def identify_memory_intensive_columns(df: pd.DataFrame, threshold_percent: float = 5.0) -> List[str]:
        """
        Identify memory-intensive columns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            threshold_percent: Minimum percentage of total memory to be considered intensive
            
        Returns:
            List of memory-intensive column names
        """
        memory_usage_df = MemoryOptimizer.memory_usage_by_column(df)
        intensive_cols = memory_usage_df[memory_usage_df['Memory_Percent'] > threshold_percent]['Column'].tolist()
        
        return intensive_cols
    
    @staticmethod
    def check_available_memory(required_mb: float = 0) -> Tuple[float, bool]:
        """
        Check available system memory.
        
        Args:
            required_mb: Amount of memory required for an operation (MB)
            
        Returns:
            Tuple of (available_memory_mb, has_enough_memory)
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        has_enough = available_mb > required_mb
        
        logger.info(f"Available memory: {available_mb:.2f} MB")
        if required_mb > 0:
            logger.info(f"Required memory: {required_mb:.2f} MB")
            logger.info(f"Sufficient memory: {has_enough}")
        
        return available_mb, has_enough


def memory_optimized(func: Callable) -> Callable:
    """
    Decorator to apply memory optimization to a function.
    Forces garbage collection before and after the function call.
    """
    def wrapper(*args, **kwargs):
        MemoryOptimizer.log_memory_usage(f"Before {func.__name__}")
        MemoryOptimizer.force_gc()
        
        result = func(*args, **kwargs)
        
        MemoryOptimizer.force_gc()
        MemoryOptimizer.log_memory_usage(f"After {func.__name__}")
        
        return result
    
    return wrapper


def estimate_memory_requirements(num_rows: int, num_cols: int, 
                              dtypes: Dict[str, str] = None) -> float:
    """
    Estimate memory requirements for a DataFrame with given dimensions.
    
    Args:
        num_rows: Number of rows
        num_cols: Number of columns
        dtypes: Dictionary mapping column names to dtypes
        
    Returns:
        Estimated memory requirement in MB
    """
    if dtypes is None:
        # Assume float64 for all columns if not specified
        bytes_per_element = 8
    else:
        # Calculate based on provided dtypes
        bytes_per_element = 0
        for dtype in dtypes.values():
            if dtype == 'float64':
                bytes_per_element += 8
            elif dtype == 'float32':
                bytes_per_element += 4
            elif dtype in ['int64', 'datetime64[ns]']:
                bytes_per_element += 8
            elif dtype in ['int32', 'uint32']:
                bytes_per_element += 4
            elif dtype in ['int16', 'uint16']:
                bytes_per_element += 2
            elif dtype in ['int8', 'uint8', 'bool']:
                bytes_per_element += 1
            else:  # Object or category, very rough estimate
                bytes_per_element += 100  # Conservative estimate for strings
        
        bytes_per_element /= len(dtypes)
    
    # Add some overhead for DataFrame structure
    estimated_bytes = num_rows * num_cols * bytes_per_element * 1.1
    estimated_mb = estimated_bytes / 1024 / 1024
    
    return estimated_mb
