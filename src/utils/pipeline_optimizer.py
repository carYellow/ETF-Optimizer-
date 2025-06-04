#!/usr/bin/env python3
"""
Pipeline Optimization Integration

This module integrates all optimization utilities into a single interface for 
use in the stock prediction training pipeline.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Import optimization modules
from src.utils.benchmarking import BenchmarkTracker
from src.utils.memory_optimization import MemoryOptimizer
from src.utils.checkpointing import CheckpointManager
from src.utils.gpu_acceleration import GPUAccelerator
from src.utils.advanced_feature_selection import AdvancedFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineOptimizer:
    """Integrates all optimization utilities for the training pipeline."""
    
    def __init__(self, 
                output_dir: str = "reports/optimization",
                checkpoint_dir: str = "checkpoints",
                experiment_name: Optional[str] = None,
                random_state: int = 42,
                n_jobs: int = -1,
                enable_gpu: bool = True,
                enable_memory_optimization: bool = True,
                enable_checkpointing: bool = True,
                enable_benchmarking: bool = True,
                verbose: bool = True):
        """
        Initialize the pipeline optimizer.
        
        Args:
            output_dir: Directory for reports and visualizations
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name of the experiment
            random_state: Random seed
            n_jobs: Number of parallel jobs
            enable_gpu: Whether to enable GPU acceleration
            enable_memory_optimization: Whether to enable memory optimization
            enable_checkpointing: Whether to enable checkpointing
            enable_benchmarking: Whether to enable benchmarking
            verbose: Whether to log detailed information
        """
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize components based on flags
        self.benchmark_tracker = BenchmarkTracker(output_dir=os.path.join(output_dir, "benchmarks")) if enable_benchmarking else None
        self.memory_optimizer = MemoryOptimizer() if enable_memory_optimization else None
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir, experiment_name=experiment_name) if enable_checkpointing else None
        self.gpu_accelerator = GPUAccelerator(verbose=verbose) if enable_gpu else None
        self.feature_selector = AdvancedFeatureSelector(output_dir=os.path.join(output_dir, "feature_selection"), 
                                                      random_state=random_state, 
                                                      n_jobs=n_jobs)
        
        if verbose:
            self._log_initialization_status()
    
    def _log_initialization_status(self) -> None:
        """Log the initialization status of components."""
        logger.info("Pipeline Optimizer initialized with:")
        logger.info(f"  Benchmarking: {'Enabled' if self.benchmark_tracker else 'Disabled'}")
        logger.info(f"  Memory Optimization: {'Enabled' if self.memory_optimizer else 'Disabled'}")
        logger.info(f"  Checkpointing: {'Enabled' if self.checkpoint_manager else 'Disabled'}")
        
        if self.gpu_accelerator:
            if self.gpu_accelerator.gpu_available:
                logger.info(f"  GPU Acceleration: Enabled ({self.gpu_accelerator.gpu_count} GPUs available)")
            else:
                logger.info("  GPU Acceleration: Enabled but no GPUs detected")
        else:
            logger.info("  GPU Acceleration: Disabled")
    
    def optimize_dataframe(self, df, verbose=True):
        """Optimize a DataFrame's memory usage."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_dataframe(df, verbose=verbose)
        return df
    
    def track_time(self, name):
        """Decorator to track execution time of a function."""
        if self.benchmark_tracker:
            return self.benchmark_tracker.track_time(name)
        
        # Return a no-op decorator if benchmarking is disabled
        def no_op_decorator(func):
            return func
        return no_op_decorator
    
    def track_memory(self, name):
        """Decorator to track memory usage of a function."""
        if self.benchmark_tracker:
            return self.benchmark_tracker.track_memory(name)
        
        # Return a no-op decorator if benchmarking is disabled
        def no_op_decorator(func):
            return func
        return no_op_decorator
    
    def checkpoint(self, name, state, metrics=None):
        """Save a checkpoint of the current state."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.save_checkpoint(state, name, metrics=metrics)
        return None
    
    def load_checkpoint(self, checkpoint_path):
        """Load a saved checkpoint."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.load_checkpoint(checkpoint_path)
        return None
    
    def optimize_model_for_gpu(self, model_name, model_config):
        """
        Optimize model configuration for GPU acceleration if available.
        
        Args:
            model_name: Name of the model
            model_config: Original model configuration
            
        Returns:
            Updated model configuration optimized for GPU
        """
        if not self.gpu_accelerator or not self.gpu_accelerator.gpu_available:
            return model_config
        
        # Create a copy to avoid modifying the original
        config = model_config.copy()
        
        # Optimize specific models for GPU
        if model_name == 'xgboost':
            # XGBoost GPU configuration
            if 'model' in config:
                config['model'].set_params(tree_method='gpu_hist')
            
            if 'param_space' in config:
                # Add GPU-specific parameters to search space
                config['param_space']['tree_method'] = ['gpu_hist']
                config['param_space']['gpu_id'] = [0]  # Use first GPU
        
        elif model_name == 'lightgbm':
            # LightGBM GPU configuration
            if 'model' in config:
                config['model'].set_params(device='gpu')
            
            if 'param_space' in config:
                config['param_space']['device'] = ['gpu']
                # Remove parameters that conflict with GPU usage
                if 'gpu_platform_id' not in config['param_space']:
                    config['param_space']['gpu_platform_id'] = [0]
                if 'gpu_device_id' not in config['param_space']:
                    config['param_space']['gpu_device_id'] = [0]
        
        elif model_name == 'catboost':
            # CatBoost GPU configuration
            if 'model' in config:
                config['model'].set_params(task_type='GPU')
            
            if 'param_space' in config:
                config['param_space']['task_type'] = ['GPU']
                
        return config
    
    def select_features(self, X, y, method='importance', **kwargs):
        """
        Select features using the specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('importance', 'correlation', 'rfe', 'shap')
            **kwargs: Additional arguments for the specific method
            
        Returns:
            List of selected feature names
        """
        if method == 'importance':
            return self.feature_selector.importance_based_selection(X, y, **kwargs)
        elif method == 'correlation':
            return self.feature_selector.correlation_based_selection(X, **kwargs)
        elif method == 'rfe':
            return self.feature_selector.recursive_feature_elimination(X, y, **kwargs)
        elif method == 'shap':
            return self.feature_selector.shap_based_selection(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def generate_optimization_report(self):
        """Generate a comprehensive optimization report."""
        if self.benchmark_tracker:
            self.benchmark_tracker.generate_report()
        
        # Generate feature selection report if features were selected
        if hasattr(self.feature_selector, 'selected_features') and self.feature_selector.selected_features:
            self.feature_selector.generate_report()
        
        logger.info(f"Optimization reports generated in {self.output_dir}")
    
    def force_gc(self):
        """Force garbage collection and log memory usage."""
        if self.memory_optimizer:
            self.memory_optimizer.force_gc()
    
    def log_memory_usage(self, label):
        """Log current memory usage with a label."""
        if self.memory_optimizer:
            self.memory_optimizer.log_memory_usage(label)
