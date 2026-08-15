"""
Ekya thief scheduler implementation.
Implements resource allocation and scheduling based on profiling data.
"""

import time
import numpy as np
from collections import defaultdict
from src.schedulers.ekya_profiler import EkyaProfiler

class EkyaScheduler:
    def __init__(self, num_gpus=1, time_slice=10, min_resource=0.1, max_resource=1.0, utility_threshold=1.5):
        self.num_gpus = num_gpus
        self.time_slice = time_slice
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.utility_threshold = utility_threshold
        self.profiler = EkyaProfiler()
        self.tasks = defaultdict(dict)
        self.current_allocations = {}
        
    def register_task(self, task_id, total_iterations, deadline=None):
        """Register a new task with the scheduler"""
        self.tasks[task_id] = {
            'total_iterations': total_iterations,
            'remaining_iterations': total_iterations,
            'deadline': deadline,
            'start_time': time.time()
        }
        
    def update_progress(self, task_id, completed_iterations):
        """Update task progress"""
        if task_id in self.tasks:
            self.tasks[task_id]['remaining_iterations'] = \
                self.tasks[task_id]['total_iterations'] - completed_iterations
                
    def _calculate_utility(self, task_id, resource_fraction):
        """Calculate utility of allocating resources to a task"""
        if task_id not in self.tasks:
            return 0
            
        task = self.tasks[task_id]
        accuracy = self.profiler.estimate_accuracy(task_id, resource_fraction) or 0
        completion_time = self.profiler.estimate_completion_time(
            task_id, 
            resource_fraction,
            task['remaining_iterations']
        ) or float('inf')
        
        if task['deadline'] and completion_time > task['deadline']:
            return 0
            
        # Utility is a combination of accuracy and completion time
        return accuracy / (1 + completion_time)
        
    def _optimize_allocation(self):
        """Optimize resource allocation across tasks"""
        allocations = {}
        remaining_resources = self.num_gpus
        
        # Sort tasks by utility
        task_utilities = []
        for task_id in self.tasks:
            max_utility = 0
            best_fraction = 0
            
            # Try different resource fractions within configured bounds
            for fraction in np.arange(self.min_resource, self.max_resource + 0.1, 0.1):
                utility = self._calculate_utility(task_id, fraction)
                if utility > max_utility:
                    max_utility = utility
                    best_fraction = fraction
                    
            task_utilities.append((task_id, max_utility, best_fraction))
            
        # Sort by utility (highest first)
        task_utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate resources
        for task_id, utility, fraction in task_utilities:
            if remaining_resources >= fraction:
                allocations[task_id] = fraction
                remaining_resources -= fraction
                
        return allocations
        
    def get_allocation(self, task_id):
        """Get current resource allocation for a task"""
        return self.current_allocations.get(task_id, 0)
        
    def update_allocations(self):
        """Update resource allocations based on current state"""
        self.current_allocations = self._optimize_allocation()
        return self.current_allocations
        
    def should_steal_resources(self, task_id):
        """Determine if task should steal resources from others"""
        if task_id not in self.tasks:
            return False
            
        current_allocation = self.get_allocation(task_id)
        task = self.tasks[task_id]
        
        # Check if task needs more resources
        completion_time = self.profiler.estimate_completion_time(
            task_id,
            current_allocation,
            task['remaining_iterations']
        )
        
        if completion_time is None:
            return False
            
        if task['deadline']:
            return completion_time > task['deadline']
            
        # If no deadline, consider stealing if utility would improve significantly
        current_utility = self._calculate_utility(task_id, current_allocation)
        potential_utility = self._calculate_utility(task_id, current_allocation + 0.1)
        
        return potential_utility > current_utility * self.utility_threshold 