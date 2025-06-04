#!/usr/bin/env python3
"""
GPU Acceleration Utilities

This module provides utilities for using GPU acceleration with tree-based models
like XGBoost and LightGBM, if available.
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Provides GPU acceleration utilities for ML models."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the GPU accelerator.
        
        Args:
            verbose: Whether to print GPU information
        """
        self.verbose = verbose
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_info = {}
        
        # Check for GPU availability
        self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> None:
        """Check for available GPUs and their capabilities."""
        # First try CUDA (NVIDIA)
        try:
            import cupy as cp
            self.gpu_available = True
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
            
            if self.verbose:
                logger.info(f"Found {self.gpu_count} CUDA-compatible GPU(s)")
            
            # Get GPU info
            for i in range(self.gpu_count):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                self.gpu_info[f"cuda:{i}"] = {
                    "name": device_props["name"].decode(),
                    "compute_capability": f"{device_props['major']}.{device_props['minor']}",
                    "total_memory_mb": device_props["totalGlobalMem"] / (1024 * 1024)
                }
                
                if self.verbose:
                    logger.info(f"GPU {i}: {self.gpu_info[f'cuda:{i}']['name']}, "
                              f"{self.gpu_info[f'cuda:{i}']['total_memory_mb']:.0f} MB")
            
            return
        except (ImportError, ModuleNotFoundError):
            if self.verbose:
                logger.info("CUDA (cupy) not available")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error checking CUDA availability: {str(e)}")
        
        # If CUDA not available, try ROCm (AMD)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_available = True
                self.gpu_count = len(gpus)
                
                if self.verbose:
                    logger.info(f"Found {self.gpu_count} ROCm-compatible GPU(s)")
                
                # Get GPU info via TensorFlow
                for i, gpu in enumerate(gpus):
                    self.gpu_info[f"rocm:{i}"] = {
                        "name": gpu.name,
                        "device_type": gpu.device_type
                    }
                    
                    if self.verbose:
                        logger.info(f"GPU {i}: {gpu.name}")
                
                return
        except (ImportError, ModuleNotFoundError):
            if self.verbose:
                logger.info("ROCm (via TensorFlow) not available")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error checking ROCm availability: {str(e)}")
        
        # If we get here, no GPU was found
        if self.verbose:
            logger.info("No GPU acceleration available. Using CPU only.")
    
    def get_xgboost_params(self, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized XGBoost parameters for GPU acceleration if available.
        
        Args:
            base_params: Base XGBoost parameters to update
            
        Returns:
            Updated parameters dictionary with GPU settings
        """
        params = base_params.copy() if base_params else {}
        
        if not self.gpu_available:
            # Use CPU optimizations
            params.update({
                'tree_method': 'hist',  # Faster CPU algorithm
                'n_jobs': -1  # Use all CPU cores
            })
            return params
        
        # GPU is available
        if self.gpu_count > 0:
            params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            
            if self.verbose:
                logger.info("Using XGBoost with GPU acceleration")
        
        return params
    
    def get_lightgbm_params(self, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized LightGBM parameters for GPU acceleration if available.
        
        Args:
            base_params: Base LightGBM parameters to update
            
        Returns:
            Updated parameters dictionary with GPU settings
        """
        params = base_params.copy() if base_params else {}
        
        if not self.gpu_available:
            # Use CPU optimizations
            params.update({
                'n_jobs': -1,  # Use all CPU cores
                'verbose': -1
            })
            return params
        
        # GPU is available
        if self.gpu_count > 0:
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1
            })
            
            if self.verbose:
                logger.info("Using LightGBM with GPU acceleration")
        
        return params
    
    def get_catboost_params(self, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized CatBoost parameters for GPU acceleration if available.
        
        Args:
            base_params: Base CatBoost parameters to update
            
        Returns:
            Updated parameters dictionary with GPU settings
        """
        params = base_params.copy() if base_params else {}
        
        if not self.gpu_available:
            # Use CPU optimizations
            params.update({
                'thread_count': -1  # Use all CPU cores
            })
            return params
        
        # GPU is available
        if self.gpu_count > 0:
            params.update({
                'task_type': 'GPU',
                'devices': '0'
            })
            
            if self.verbose:
                logger.info("Using CatBoost with GPU acceleration")
        
        return params
    
    def optimize_for_device(self, algorithm: str, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized parameters for the specified algorithm based on available hardware.
        
        Args:
            algorithm: One of 'xgboost', 'lightgbm', 'catboost'
            base_params: Base parameters to update
            
        Returns:
            Updated parameters dictionary with hardware-specific optimizations
        """
        if algorithm.lower() == 'xgboost':
            return self.get_xgboost_params(base_params)
        elif algorithm.lower() == 'lightgbm':
            return self.get_lightgbm_params(base_params)
        elif algorithm.lower() == 'catboost':
            return self.get_catboost_params(base_params)
        else:
            logger.warning(f"Unknown algorithm: {algorithm}. Returning base parameters unchanged.")
            return base_params.copy() if base_params else {}


# Global GPU accelerator
_gpu_accelerator = None

def get_gpu_accelerator(verbose: bool = True) -> GPUAccelerator:
    """
    Get the global GPU accelerator instance.
    
    Args:
        verbose: Whether to print GPU information
        
    Returns:
        GPUAccelerator instance
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(verbose=verbose)
    return _gpu_accelerator

def optimize_model_for_gpu(model_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Optimize model parameters for GPU if available.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        params: Base parameters to update
        
    Returns:
        Updated parameters dictionary
    """
    return get_gpu_accelerator().optimize_for_device(model_type, params)
