"""
Ekya micro-profiler implementation.
Profiles model performance and resource usage for scheduling decisions.
"""

import time
import numpy as np
from collections import defaultdict

class EkyaProfiler:
    def __init__(self):
        self.profiles = defaultdict(dict)
        self.current_profile = None
        self.start_time = None
        
    def start_profiling(self, task_id, resource_fraction):
        """Start profiling a task with given resource fraction"""
        self.current_profile = {
            'task_id': task_id,
            'resource_fraction': resource_fraction,
            'start_time': time.time(),
            'accuracy': [],
            'loss': [],
            'throughput': []
        }
        
    def record_metrics(self, accuracy, loss, batch_size, time_taken):
        """Record metrics for current profiling session"""
        if self.current_profile is None:
            return
            
        self.current_profile['accuracy'].append(accuracy)
        self.current_profile['loss'].append(loss)
        self.current_profile['throughput'].append(batch_size / time_taken)
        
    def end_profiling(self):
        """End current profiling session and save results"""
        if self.current_profile is None:
            return
            
        task_id = self.current_profile['task_id']
        resource_fraction = self.current_profile['resource_fraction']
        
        # Calculate average metrics
        profile_summary = {
            'avg_accuracy': np.mean(self.current_profile['accuracy']),
            'avg_loss': np.mean(self.current_profile['loss']),
            'avg_throughput': np.mean(self.current_profile['throughput']),
            'duration': time.time() - self.current_profile['start_time']
        }
        
        # Store profile
        self.profiles[task_id][resource_fraction] = profile_summary
        self.current_profile = None
        
    def get_profile(self, task_id, resource_fraction):
        """Get profile for specific task and resource fraction"""
        return self.profiles.get(task_id, {}).get(resource_fraction)
        
    def estimate_completion_time(self, task_id, resource_fraction, remaining_iterations):
        """Estimate time to complete remaining iterations based on profile"""
        profile = self.get_profile(task_id, resource_fraction)
        if profile is None:
            return None
            
        avg_throughput = profile['avg_throughput']
        if avg_throughput <= 0:
            return float('inf')
            
        return remaining_iterations / avg_throughput
        
    def estimate_accuracy(self, task_id, resource_fraction):
        """Estimate accuracy for given resource fraction"""
        profile = self.get_profile(task_id, resource_fraction)
        if profile is None:
            return None
        return profile['avg_accuracy'] 