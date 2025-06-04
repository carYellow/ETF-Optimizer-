#!/usr/bin/env python3
"""
Checkpointing Utilities

This module provides utilities for saving and loading training checkpoints,
allowing training to be resumed if interrupted.
"""

import os
import json
import joblib
import pickle
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages checkpoints for model training."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", experiment_name: Optional[str] = None):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            experiment_name: Name of the experiment (defaults to timestamp)
        """
        self.checkpoint_dir = checkpoint_dir
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(checkpoint_dir, experiment_name)
        
        # Create checkpoint directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'checkpoints': [],
            'config': {},
            'metrics': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Save initial metadata
        self._save_metadata()
        
        logger.info(f"Checkpoint manager initialized at {self.experiment_dir}")
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.metadata['config'] = config
        self._save_metadata()
        
        # Also save as separate file for easy access
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Config saved to {config_path}")
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_name: str, 
                      metrics: Optional[Dict[str, Any]] = None,
                      save_format: str = 'joblib') -> str:
        """
        Save a training checkpoint.
        
        Args:
            state: Dictionary containing training state (models, scalers, etc.)
            checkpoint_name: Name of the checkpoint
            metrics: Optional dictionary of metrics
            save_format: Format to save the checkpoint ('joblib', 'pickle')
            
        Returns:
            Path to the saved checkpoint file
        """
        # Create checkpoint file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = f"{checkpoint_name}_{timestamp}"
        
        if save_format == 'joblib':
            checkpoint_path = os.path.join(self.experiment_dir, f"{checkpoint_file}.joblib")
            joblib.dump(state, checkpoint_path)
        elif save_format == 'pickle':
            checkpoint_path = os.path.join(self.experiment_dir, f"{checkpoint_file}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        # Update metadata
        checkpoint_info = {
            'name': checkpoint_name,
            'file': os.path.basename(checkpoint_path),
            'timestamp': timestamp,
            'format': save_format
        }
        
        if metrics:
            checkpoint_info['metrics'] = metrics
            
            # Also update overall metrics history
            for name, value in metrics.items():
                if name not in self.metadata['metrics']:
                    self.metadata['metrics'][name] = []
                self.metadata['metrics'][name].append({
                    'value': value,
                    'timestamp': timestamp,
                    'checkpoint': checkpoint_name
                })
        
        self.metadata['checkpoints'].append(checkpoint_info)
        self._save_metadata()
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_name: Optional[str] = None, 
                       checkpoint_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load (loads latest if multiple)
            checkpoint_file: Specific checkpoint file to load
            
        Returns:
            Dictionary containing training state
        """
        if checkpoint_file:
            # Load specific file
            checkpoint_path = os.path.join(self.experiment_dir, checkpoint_file)
        elif checkpoint_name:
            # Find latest checkpoint with matching name
            matching_checkpoints = [cp for cp in self.metadata['checkpoints'] if cp['name'] == checkpoint_name]
            if not matching_checkpoints:
                raise ValueError(f"No checkpoint found with name: {checkpoint_name}")
            
            # Sort by timestamp (latest first)
            sorted_checkpoints = sorted(matching_checkpoints, key=lambda x: x['timestamp'], reverse=True)
            checkpoint_path = os.path.join(self.experiment_dir, sorted_checkpoints[0]['file'])
        else:
            # Load latest checkpoint
            if not self.metadata['checkpoints']:
                raise ValueError("No checkpoints available to load")
            
            sorted_checkpoints = sorted(self.metadata['checkpoints'], key=lambda x: x['timestamp'], reverse=True)
            checkpoint_path = os.path.join(self.experiment_dir, sorted_checkpoints[0]['file'])
        
        # Load checkpoint
        if checkpoint_path.endswith('.joblib'):
            state = joblib.load(checkpoint_path)
        elif checkpoint_path.endswith('.pkl'):
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return state
    
    def get_checkpoint_info(self, checkpoint_name: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get information about checkpoints.
        
        Args:
            checkpoint_name: Optional name to filter checkpoints
            
        Returns:
            List of checkpoint info dictionaries or a single info dictionary
        """
        if checkpoint_name:
            matching_checkpoints = [cp for cp in self.metadata['checkpoints'] if cp['name'] == checkpoint_name]
            if not matching_checkpoints:
                raise ValueError(f"No checkpoint found with name: {checkpoint_name}")
                
            if len(matching_checkpoints) == 1:
                return matching_checkpoints[0]
            return matching_checkpoints
        
        return self.metadata['checkpoints']
    
    def get_metrics_history(self, metric_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metrics history.
        
        Args:
            metric_name: Optional name to filter metrics
            
        Returns:
            Dictionary of metric histories
        """
        if metric_name:
            if metric_name not in self.metadata['metrics']:
                raise ValueError(f"No metric found with name: {metric_name}")
            return {metric_name: self.metadata['metrics'][metric_name]}
        
        return self.metadata['metrics']
    
    def plot_metrics_history(self, metric_names: Optional[List[str]] = None) -> None:
        """
        Plot metrics history.
        
        Args:
            metric_names: Optional list of metric names to plot
        """
        import matplotlib.pyplot as plt
        
        if not metric_names:
            metric_names = list(self.metadata['metrics'].keys())
        
        if not metric_names:
            logger.warning("No metrics available to plot")
            return
        
        for metric_name in metric_names:
            if metric_name not in self.metadata['metrics']:
                logger.warning(f"Metric not found: {metric_name}")
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Extract values and timestamps
            values = [m['value'] for m in self.metadata['metrics'][metric_name]]
            timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metadata['metrics'][metric_name]]
            checkpoints = [m['checkpoint'] for m in self.metadata['metrics'][metric_name]]
            
            # Create the plot
            plt.plot(timestamps, values, 'o-')
            plt.xlabel('Timestamp')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} History')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.experiment_dir, f"{metric_name}_history.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"{metric_name} history plot saved to {plot_path}")
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiments.
        
        Returns:
            List of experiment names
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        experiments = [d for d in os.listdir(self.checkpoint_dir) 
                      if os.path.isdir(os.path.join(self.checkpoint_dir, d))]
        return experiments
    
    def load_experiment(self, experiment_name: str) -> None:
        """
        Load an existing experiment.
        
        Args:
            experiment_name: Name of the experiment to load
        """
        experiment_dir = os.path.join(self.checkpoint_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        metadata_path = os.path.join(experiment_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata not found for experiment: {experiment_name}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        
        logger.info(f"Loaded experiment: {experiment_name}")
    
    def delete_checkpoint(self, checkpoint_file: str) -> None:
        """
        Delete a checkpoint file.
        
        Args:
            checkpoint_file: Name of the checkpoint file to delete
        """
        checkpoint_path = os.path.join(self.experiment_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
        
        os.remove(checkpoint_path)
        
        # Update metadata
        self.metadata['checkpoints'] = [cp for cp in self.metadata['checkpoints'] 
                                      if cp['file'] != checkpoint_file]
        self._save_metadata()
        
        logger.info(f"Deleted checkpoint: {checkpoint_file}")
    
    def create_resume_state(self, last_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a state dictionary for resuming training.
        
        Args:
            last_checkpoint: Name of the last checkpoint to use
            
        Returns:
            Dictionary containing the resume state
        """
        state = self.load_checkpoint(checkpoint_name=last_checkpoint)
        
        # Add resume metadata
        state['_resume_info'] = {
            'resumed_from': last_checkpoint if last_checkpoint else 'latest',
            'resumed_at': datetime.now().isoformat(),
            'experiment_name': self.experiment_name
        }
        
        return state


# Global checkpoint manager
_checkpoint_manager = None

def get_checkpoint_manager(checkpoint_dir: str = "checkpoints", experiment_name: Optional[str] = None) -> CheckpointManager:
    """
    Get the global checkpoint manager instance.
    
    Args:
        checkpoint_dir: Directory to store checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        CheckpointManager instance
    """
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(checkpoint_dir, experiment_name)
    return _checkpoint_manager

def save_checkpoint(state: Dict[str, Any], checkpoint_name: str, metrics: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a training checkpoint using the global checkpoint manager.
    
    Args:
        state: Dictionary containing training state
        checkpoint_name: Name of the checkpoint
        metrics: Optional dictionary of metrics
        
    Returns:
        Path to the saved checkpoint file
    """
    return get_checkpoint_manager().save_checkpoint(state, checkpoint_name, metrics)

def load_checkpoint(checkpoint_name: Optional[str] = None, checkpoint_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a training checkpoint using the global checkpoint manager.
    
    Args:
        checkpoint_name: Name of the checkpoint to load
        checkpoint_file: Specific checkpoint file to load
        
    Returns:
        Dictionary containing training state
    """
    return get_checkpoint_manager().load_checkpoint(checkpoint_name, checkpoint_file)

def save_config(config: Dict[str, Any]) -> None:
    """
    Save experiment configuration using the global checkpoint manager.
    
    Args:
        config: Dictionary of configuration parameters
    """
    get_checkpoint_manager().save_config(config)
