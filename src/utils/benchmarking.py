#!/usr/bin/env python3
"""
Benchmarking Utilities

This module provides utilities for benchmarking and profiling the training pipeline,
including time and memory usage tracking.
"""

import time
import gc
import os
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import json
from typing import Dict, List, Callable, Any, Optional, Union

class BenchmarkTracker:
    """Tracks benchmark metrics across the training pipeline."""
    
    def __init__(self, output_dir: str = "reports/benchmarks"):
        self.output_dir = output_dir
        self.timings = {}
        self.memory_usage = {}
        self.metrics = {}
        self.start_time = time.time()
        self.checkpoints = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def track_time(self, name: str) -> Callable:
        """Decorator to track execution time of a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if name not in self.timings:
                    self.timings[name] = []
                
                self.timings[name].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': end_time - start_time
                })
                
                return result
            return wrapper
        return decorator
    
    def track_memory(self, name: str) -> Callable:
        """Decorator to track memory usage of a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Force garbage collection before measuring
                gc.collect()
                
                # Get memory usage before
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                # Force garbage collection after function execution
                gc.collect()
                
                # Get memory usage after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                if name not in self.memory_usage:
                    self.memory_usage[name] = []
                
                self.memory_usage[name].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_change_mb': memory_after - memory_before
                })
                
                return result
            return wrapper
        return decorator
    
    def checkpoint(self, name: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Record a checkpoint with optional metrics."""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': time.time() - self.start_time,
            'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        self.checkpoints.append(checkpoint)
    
    def add_metric(self, category: str, name: str, value: Any) -> None:
        """Add a custom metric."""
        if category not in self.metrics:
            self.metrics[category] = {}
        
        if name not in self.metrics[category]:
            self.metrics[category][name] = []
        
        self.metrics[category][name].append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'value': value
        })
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        report_path = os.path.join(self.output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Time performance
            f.write("## Time Performance\n\n")
            if self.timings:
                f.write("| Function | Calls | Total Time (s) | Avg Time (s) | Min Time (s) | Max Time (s) |\n")
                f.write("|----------|-------|---------------|--------------|--------------|-------------|\n")
                
                for name, times in self.timings.items():
                    durations = [t['duration'] for t in times]
                    total = sum(durations)
                    avg = total / len(durations)
                    min_time = min(durations)
                    max_time = max(durations)
                    
                    f.write(f"| {name} | {len(durations)} | {total:.2f} | {avg:.2f} | {min_time:.2f} | {max_time:.2f} |\n")
            else:
                f.write("No timing data recorded.\n\n")
            
            # Memory usage
            f.write("\n## Memory Usage\n\n")
            if self.memory_usage:
                f.write("| Function | Calls | Avg Memory Before (MB) | Avg Memory After (MB) | Avg Change (MB) | Max Change (MB) |\n")
                f.write("|----------|-------|------------------------|----------------------|----------------|----------------|\n")
                
                for name, usages in self.memory_usage.items():
                    before = [u['memory_before_mb'] for u in usages]
                    after = [u['memory_after_mb'] for u in usages]
                    changes = [u['memory_change_mb'] for u in usages]
                    
                    avg_before = sum(before) / len(before)
                    avg_after = sum(after) / len(after)
                    avg_change = sum(changes) / len(changes)
                    max_change = max(changes)
                    
                    f.write(f"| {name} | {len(usages)} | {avg_before:.2f} | {avg_after:.2f} | {avg_change:.2f} | {max_change:.2f} |\n")
            else:
                f.write("No memory usage data recorded.\n\n")
            
            # Checkpoints
            f.write("\n## Checkpoints\n\n")
            if self.checkpoints:
                f.write("| Checkpoint | Timestamp | Elapsed Time (s) | Memory (MB) |\n")
                f.write("|------------|-----------|------------------|------------|\n")
                
                for cp in self.checkpoints:
                    metrics_str = ""
                    if 'metrics' in cp:
                        metrics_str = " | " + " | ".join([f"{k}: {v}" for k, v in cp['metrics'].items()])
                    
                    f.write(f"| {cp['name']} | {cp['timestamp']} | {cp['elapsed_time']:.2f} | {cp['memory_mb']:.2f}{metrics_str} |\n")
            else:
                f.write("No checkpoints recorded.\n\n")
            
            # Custom metrics
            f.write("\n## Custom Metrics\n\n")
            if self.metrics:
                for category, metrics in self.metrics.items():
                    f.write(f"### {category}\n\n")
                    
                    for name, values in metrics.items():
                        if isinstance(values[0]['value'], (int, float)):
                            avg_value = sum(v['value'] for v in values) / len(values)
                            f.write(f"- {name}: {avg_value:.4f} (avg of {len(values)} records)\n")
                        else:
                            f.write(f"- {name}: {len(values)} records\n")
            else:
                f.write("No custom metrics recorded.\n\n")
        
        # Generate visualizations
        self._generate_visualizations()
        
        return report_path
    
    def _generate_visualizations(self) -> None:
        """Generate visualizations of benchmarking data."""
        # Time performance chart
        if self.timings:
            plt.figure(figsize=(12, 6))
            
            functions = []
            avg_times = []
            
            for name, times in self.timings.items():
                functions.append(name)
                durations = [t['duration'] for t in times]
                avg_times.append(sum(durations) / len(durations))
            
            # Sort by average time
            sorted_indices = np.argsort(avg_times)[::-1]
            functions = [functions[i] for i in sorted_indices]
            avg_times = [avg_times[i] for i in sorted_indices]
            
            plt.barh(functions, avg_times)
            plt.xlabel('Average Time (seconds)')
            plt.title('Average Function Execution Time')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'time_performance.png'), dpi=300)
            plt.close()
        
        # Memory usage chart
        if self.memory_usage:
            plt.figure(figsize=(12, 6))
            
            functions = []
            avg_changes = []
            
            for name, usages in self.memory_usage.items():
                functions.append(name)
                changes = [u['memory_change_mb'] for u in usages]
                avg_changes.append(sum(changes) / len(changes))
            
            # Sort by average memory change
            sorted_indices = np.argsort(avg_changes)[::-1]
            functions = [functions[i] for i in sorted_indices]
            avg_changes = [avg_changes[i] for i in sorted_indices]
            
            plt.barh(functions, avg_changes)
            plt.xlabel('Average Memory Change (MB)')
            plt.title('Average Memory Usage Change')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'memory_usage.png'), dpi=300)
            plt.close()
        
        # Checkpoint timeline
        if self.checkpoints:
            plt.figure(figsize=(14, 8))
            
            names = [cp['name'] for cp in self.checkpoints]
            times = [cp['elapsed_time'] for cp in self.checkpoints]
            memories = [cp['memory_mb'] for cp in self.checkpoints]
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            color = 'tab:blue'
            ax1.set_xlabel('Checkpoint')
            ax1.set_ylabel('Elapsed Time (s)', color=color)
            ax1.plot(names, times, 'o-', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            plt.xticks(rotation=45, ha='right')
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Memory Usage (MB)', color=color)
            ax2.plot(names, memories, 'o-', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Checkpoint Timeline')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'checkpoint_timeline.png'), dpi=300)
            plt.close()
    
    def save_data(self) -> None:
        """Save all benchmarking data to JSON for later analysis."""
        data = {
            'timings': self.timings,
            'memory_usage': self.memory_usage,
            'checkpoints': self.checkpoints,
            'metrics': self.metrics
        }
        
        data_path = os.path.join(self.output_dir, f"benchmark_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Benchmark data saved to {data_path}")


# Singleton instance for global access
_benchmark_tracker = None

def get_benchmark_tracker(output_dir: str = "reports/benchmarks") -> BenchmarkTracker:
    """Get the global benchmark tracker instance."""
    global _benchmark_tracker
    if _benchmark_tracker is None:
        _benchmark_tracker = BenchmarkTracker(output_dir)
    return _benchmark_tracker

# Convenient decorator functions
def track_time(name: str) -> Callable:
    """Decorator to track execution time."""
    return get_benchmark_tracker().track_time(name)

def track_memory(name: str) -> Callable:
    """Decorator to track memory usage."""
    return get_benchmark_tracker().track_memory(name)

def checkpoint(name: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """Record a benchmark checkpoint."""
    get_benchmark_tracker().checkpoint(name, metrics)

def generate_report() -> str:
    """Generate a benchmark report."""
    return get_benchmark_tracker().generate_report()

def save_data() -> None:
    """Save benchmarking data to JSON."""
    get_benchmark_tracker().save_data()
