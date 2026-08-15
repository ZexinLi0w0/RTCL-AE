"""
Dynamic configuration worker implementation.
"""

import time
import signal
import os
import threading
from src.utils.signal_handlers import safe_kill
from src.utils.logging_utils import log_event, log_info, log_warning, log_error
from src.utils.config_loader import load_config, validate_batch_config, get_config_value

# Constants for timing
CONFIG_POLL_INTERVAL = 0.5  # Polling interval in seconds to check for termination

def dynamic_config_worker(config_controller, train_pid, eval_pid, timeslice=10.0, config_file=None):
    """
    Worker for dynamic configuration of batch size.
    
    Args:
        config_controller: Dynamic configuration controller
        train_pid: PID of the training process
        eval_pid: PID of the evaluation process
        timeslice: Default time interval between configuration updates
        config_file: Path to the configuration file
    """
    try:
        # set default batch size sequence
        train_batch_sizes = [32, 64, 128, 256, 512]
        eval_batch_sizes = [16, 32, 64, 128, 256]
        durations = [timeslice] * len(train_batch_sizes)  # duration of each configuration
        
        # if config file is provided, load settings from the file
        if config_file and os.path.exists(config_file):
            try:
                log_event("config", f"Loading configuration from file: {config_file}")
                config = load_config(config_file)
                
                # validate config file
                validate_batch_config(config)
                
                # update batch size and duration
                batches = config.get('batches', [])
                train_batch_sizes = [batch['train_batch_size'] for batch in batches]
                eval_batch_sizes = [batch['eval_batch_size'] for batch in batches]
                durations = [batch.get('duration', timeslice) for batch in batches]
                
                # get global timeslice value (if provided)
                config_timeslice = get_config_value(config, 'timeslice', timeslice)
                
                log_event("config", f"Successfully loaded configuration with {len(batches)} batch configurations")
                log_info(f"[DynamicConfig] Loaded configuration with {len(batches)} batch settings")
                
                # print loaded settings (for debugging)
                for i, (train_bs, eval_bs, duration) in enumerate(zip(train_batch_sizes, eval_batch_sizes, durations)):
                    log_info(f"[DynamicConfig] Config {i+1}: train_batch={train_bs}, eval_batch={eval_bs}, duration={duration}s")
                
            except Exception as e:
                log_event("config", f"Error loading configuration file: {e}")
                log_error(f"[DynamicConfig] Error loading configuration file: {e}")
                log_warning(f"[DynamicConfig] Falling back to default configuration.")
        else:
            if config_file:
                log_event("config", f"Configuration file not found: {config_file}")
                log_warning(f"[DynamicConfig] Configuration file not found: {config_file}")
            log_info(f"[DynamicConfig] Using default batch size sequence")
        
        log_event("config", "Dynamic configuration worker started")
        log_info("[DynamicConfig] Dynamic configuration worker started")

        # Function to wait for specified duration with termination checks
        def wait_with_termination_check(duration):
            """Wait for specified duration with periodic termination checks."""
            wait_start_time = time.time()
            while time.time() - wait_start_time < duration:
                # check for termination signal
                if config_controller.shared_data.get("TERMINATE_SIGNAL", False):
                    log_event("config", "Termination signal received. Stopping configuration worker.")
                    log_info("[DynamicConfig] Termination signal received. Stopping configuration worker.")
                    return False  # Terminate
                
                # wait for a short period
                time.sleep(CONFIG_POLL_INTERVAL)
            return True  # Continue execution

        # change batch size sequentially
        for i in range(len(train_batch_sizes)):
            # wait for the current duration, but check for termination periodically
            current_duration = durations[i]
            log_event("config", f"Waiting {current_duration}s before applying next configuration")
            
            # If should terminate, exit
            if not wait_with_termination_check(current_duration):
                return
            
            # check for termination signal again (redundant but ensures consistent behavior)
            if config_controller.shared_data.get("TERMINATE_SIGNAL", False):
                log_event("config", "Termination signal received. Stopping configuration worker.")
                log_info("[DynamicConfig] Termination signal received. Stopping configuration worker.")
                return  # Exit function

            # create new configuration
            new_config = {
                "train_batch_size": train_batch_sizes[i],
                "eval_batch_size": eval_batch_sizes[i],
                "timeslice": timeslice
            }
            
            # update configuration
            config_controller.update_config(new_config)
            
            # print new configuration
            log_info(f"[DynamicConfig] New configuration ready: train_batch={train_batch_sizes[i]}, eval_batch={eval_batch_sizes[i]}")
            log_info(f"[DynamicConfig] Sending signal to: train_pid={train_pid}, eval_pid={eval_pid}")
            
            # send signal to training and evaluation processes
            train_sent = safe_kill(train_pid, signal.SIGUSR1)
            eval_sent = safe_kill(eval_pid, signal.SIGUSR1)
            
            log_info(f"[DynamicConfig] Signal sending result: train={train_sent}, eval={eval_sent}")
            
            log_event("config", f"New configuration applied: train batch={train_batch_sizes[i]}, eval batch={eval_batch_sizes[i]}")
        
        log_event("config", "Configuration change test completed")
        log_info("[DynamicConfig] Configuration change test completed")
        
    except Exception as e:
        log_event("config", f"Exception in dynamic_config_worker: {e}")
        log_error(f"[DynamicConfig] Exception occurred in worker: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        log_event("config", "Configuration worker shutting down")
        log_info("[DynamicConfig] Configuration worker shutting down")