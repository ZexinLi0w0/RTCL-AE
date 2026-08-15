"""
Dynamic configuration controller for runtime parameter adjustments.
"""

class DynamicConfigController:
    """
    Controller for dynamic configuration of training and evaluation.
    Allows changing parameters at runtime, such as batch sizes.
    """
    def __init__(self, shared_data, args):
        self.shared_data = shared_data
        self.args = args
        self.max_batch_size = 512  # Max batch size
        
        # Initialize configuration
        if "config" not in self.shared_data:
            self.shared_data["config"] = {
                "train_batch_size": args.training_bs,
                "eval_batch_size": args.eval_bs,
                "timeslice": args.timeslice
            }
            
        if "config_version" not in self.shared_data:
            self.shared_data["config_version"] = 0
            
        self.local_config_version = 0
        
        # Adaptive accuracy related variables
        self.accuracy_threshold = args.adaptive_accuracy_threshold
        self.batch_size_increase_factor = 2  # Batch size increase factor
        self.last_accuracy = 0.0
        self.consecutive_threshold_meets = 0
        self.required_consecutive_meets = 2  # Number of consecutive threshold meets
    
    def check_and_adapt_batch_size(self, current_accuracy):
        """
        Determine if batch size should be adjusted based on current accuracy.
        
        Args:
            current_accuracy: Current accuracy of the evaluation
        
        Returns:
            bool: Whether batch size has been changed
        """
        if self.args.global_scheduler_mode != "adaptive_accuracy":
            return False
            
        current_config = self.get_current_config()
        current_batch_size = current_config["train_batch_size"]
        
        # Check if accuracy meets threshold
        if current_accuracy >= self.accuracy_threshold:
            self.consecutive_threshold_meets += 1
            print(f"[Adaptive] Accuracy {current_accuracy:.4f} meets threshold {self.accuracy_threshold:.4f} "
                  f"({self.consecutive_threshold_meets}/{self.required_consecutive_meets})")
            
            # Check if consecutive threshold is met
            if self.consecutive_threshold_meets >= self.required_consecutive_meets:
                # Calculate new batch size
                new_batch_size = min(
                    int(current_batch_size * self.batch_size_increase_factor),
                    self.max_batch_size
                )
                
                # Check if batch size actually increases
                if new_batch_size > current_batch_size:
                    print(f"[Adaptive] Increasing batch size from {current_batch_size} to {new_batch_size}")
                    
                    # Apply new configuration
                    new_config = current_config.copy()
                    new_config["train_batch_size"] = new_batch_size
                    new_config["eval_batch_size"] = min(new_batch_size, self.max_batch_size)
                    
                    # Update configuration
                    self.update_config(new_config)
                    
                    # Reset counter
                    self.consecutive_threshold_meets = 0
                    return True
        else:
            # If threshold is not met, reset counter
            self.consecutive_threshold_meets = 0
            print(f"[Adaptive] Accuracy {current_accuracy:.4f} below threshold {self.accuracy_threshold:.4f}")
        
        self.last_accuracy = current_accuracy
        return False
    
    def update_config(self, new_config):
        """
        Update configuration with new values.
        
        Args:
            new_config: Dictionary containing new configuration values
            
        Returns:
            Boolean indicating whether any values changed
        """
        current_config = self.get_current_config()
        changed = False
        
        for key, value in new_config.items():
            if key in current_config and current_config[key] != value:
                current_config[key] = value
                changed = True
                
        if changed:
            self.shared_data["config"] = current_config
            # Increase version
            self.shared_data["config_version"] = self.shared_data["config_version"] + 1
            print(f"[ConfigController] Configuration updated: {current_config}")
        
        return changed
    
    def get_current_config(self):
        """
        Get the current configuration.
        
        Returns:
            Dictionary containing current configuration values
        """
        return self.shared_data["config"].copy()
        
    def check_for_updates(self):
        """
        Check if configuration has been updated since last check.
        
        Returns:
            Boolean indicating whether an update occurred
        """
        current_version = self.shared_data["config_version"]
        if current_version > self.local_config_version:
            self.local_config_version = current_version
            return True
        return False