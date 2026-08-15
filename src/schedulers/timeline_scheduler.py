"""
Global timeline scheduler for managing train/eval processes.
"""

import os
import sys
import signal
import time
import psutil
from src.utils.signal_handlers import safe_kill
from src.utils.logging_utils import log_info, log_warning, log_error
from src.schedulers.ekya_scheduler import EkyaScheduler

# Constants for timing
SCHEDULER_POLL_INTERVAL = 1.0    # Default polling interval in seconds
EVAL_TRANSITION_TIME = 3.0       # Time to allow for evaluation process to run
FORCED_EVAL_INTERVAL = 10.0      # Seconds between forced evaluation checks
MAX_ACCURACY_CHECKS = 10         # Maximum number of checks before forcing mode switch

def set_cpu_affinity(pid, cpu_list):
    """
    Set CPU affinity for a process using psutil.
    """
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(cpu_list)
        print(f"[EkyaScheduler] Set PID {pid} affinity to CPUs {cpu_list}")
    except Exception as e:
        print(f"[EkyaScheduler] Failed to set affinity for PID {pid}: {e}")

class GlobalTimelineScheduler:
    """
    The global timeline scheduler manages the train/eval processes with different scheduling strategies:
      - default: alternates train/eval each time_slice seconds.
      - fully_parallel: train & eval both run continuously.
      - continuous_eval: evaluation runs continuously; training is intermittent (on/off in time slices).
      - adaptive_time: prioritizes training in early experiences, then switches to alternating.
      - adaptive_accuracy: prioritizes training until reaching a target accuracy, then alternates.
      - ekya: uses Ekya's micro-profiler and thief scheduler for resource allocation.
    """
    def __init__(self, time_slice, mode="default", adaptive_params=None):
        self.time_slice = time_slice
        self.mode = mode
        
        # Initialize Ekya scheduler if in ekya mode
        if mode == "ekya":
            ekya_params = adaptive_params.get("ekya", {}) if adaptive_params else {}
            self.ekya_scheduler = EkyaScheduler(
                time_slice=time_slice,
                min_resource=ekya_params.get("min_resource", 0.1),
                max_resource=ekya_params.get("max_resource", 1.0),
                utility_threshold=ekya_params.get("utility_threshold", 1.5)
            )
        else:
            self.ekya_scheduler = None
        
        # Parameters for adaptive scheduling
        if adaptive_params is None:
            self.adaptive_params = {
                # For adaptive_time mode: percentage of experiences to prioritize training
                "priority_percent": 0.3,  
                
                # For adaptive_accuracy mode: accuracy threshold to switch scheduling
                "accuracy_threshold": 0.4  
            }
        else:
            self.adaptive_params = adaptive_params

def global_scheduler_worker(global_scheduler, train_pid, eval_pid, shared_data):
    """
    Global scheduler logic for managing train/eval processes and batch size updates
    """
    mode = global_scheduler.mode
    time_slice = global_scheduler.time_slice
    
    train_alive = True
    eval_alive = True
    
    def check_processes_alive():
        nonlocal train_alive, eval_alive
        if train_alive and not safe_kill(train_pid, 0):
            train_alive = False
            log_info("[Scheduler] Training process has terminated")
        if eval_alive and not safe_kill(eval_pid, 0):
            eval_alive = False
            log_info("[Scheduler] Evaluation process has terminated")
        return train_alive or eval_alive

    def check_and_update_config():
        """Check and signal processes for configuration updates"""
        if shared_data.get("CONFIG_UPDATE_REQUESTED", False):
            log_info("[Scheduler] Configuration update requested, signaling processes")
            if train_alive:
                safe_kill(train_pid, signal.SIGUSR1)
            if eval_alive:
                safe_kill(eval_pid, signal.SIGUSR1)
            shared_data["CONFIG_UPDATE_REQUESTED"] = False

    log_info(f"[Scheduler] Starting with mode: {mode}, time_slice: {time_slice}s")
    
    if mode == "ekya":
        log_info("[Scheduler] Running in Ekya mode with micro-profiler and thief scheduler")
        
        ekya = global_scheduler.ekya_scheduler
        ekya.register_task("train", total_iterations=shared_data.get("total_iterations", 1000))
        ekya.register_task("eval", total_iterations=shared_data.get("eval_iterations", 100))
        ekya.profiler.start_profiling("train", 0.5)
        ekya.profiler.start_profiling("eval", 0.5)
        
        # Example: total available CPU cores
        total_cores = psutil.cpu_count(logical=False) or 4
        train_cores = list(range(total_cores // 2))
        eval_cores = list(range(total_cores // 2, total_cores))
        set_cpu_affinity(train_pid, train_cores)
        set_cpu_affinity(eval_pid, eval_cores)
        
        while check_processes_alive():
            check_and_update_config()
            # Example: dynamically adjust CPU allocation every 10 seconds
            if int(time.time()) % 10 == 0:
                # Simple policy: if train is prioritized, give it more cores
                if shared_data.get("train_priority", True):
                    set_cpu_affinity(train_pid, list(range(total_cores)))
                    set_cpu_affinity(eval_pid, [])  # Pause eval
                else:
                    set_cpu_affinity(train_pid, train_cores)
                    set_cpu_affinity(eval_pid, eval_cores)
            
            # Update task progress
            if shared_data.get("train_iterations_completed"):
                ekya.update_progress("train", shared_data["train_iterations_completed"])
            if shared_data.get("eval_iterations_completed"):
                ekya.update_progress("eval", shared_data["eval_iterations_completed"])
            
            # Record metrics
            if shared_data.get("train_metrics"):
                metrics = shared_data["train_metrics"]
                ekya.profiler.record_metrics(
                    accuracy=metrics.get("accuracy", 0),
                    loss=metrics.get("loss", 0),
                    batch_size=metrics.get("batch_size", 1),
                    time_taken=metrics.get("time_taken", 1)
                )
            
            # Update resource allocations
            allocations = ekya.update_allocations()
            
            # Apply resource allocations through CPU scheduling
            train_allocation = allocations.get("train", 0)
            eval_allocation = allocations.get("eval", 0)
            
            # Implement resource allocation through process scheduling
            if train_alive and train_allocation > 0:
                safe_kill(train_pid, signal.SIGCONT)
                time.sleep(time_slice * train_allocation)
                safe_kill(train_pid, signal.SIGSTOP)
                
            if eval_alive and eval_allocation > 0:
                safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice * eval_allocation)
                safe_kill(eval_pid, signal.SIGSTOP)
            
            # Check if any task needs to steal resources
            if ekya.should_steal_resources("train"):
                log_info("[Scheduler] Train task stealing resources")
                safe_kill(eval_pid, signal.SIGSTOP)
                safe_kill(train_pid, signal.SIGCONT)
                time.sleep(time_slice)
            elif ekya.should_steal_resources("eval"):
                log_info("[Scheduler] Eval task stealing resources")
                safe_kill(train_pid, signal.SIGSTOP)
                safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice)
            
            time.sleep(SCHEDULER_POLL_INTERVAL)
            
    elif mode == "fully_parallel":
        log_info("[Scheduler] Running both train & eval continuously")
        if train_alive:
            safe_kill(train_pid, signal.SIGCONT)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGCONT)
            
        while check_processes_alive():
            check_and_update_config()
            time.sleep(time_slice)
            
    elif mode == "continuous_eval":
        log_info("[Scheduler] Running eval continuously, training in time slices")
        if train_alive:
            safe_kill(train_pid, signal.SIGSTOP)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGCONT)
            
        while check_processes_alive():
            check_and_update_config()
            
            if train_alive:
                log_info("[Scheduler] Resuming training")
                safe_kill(train_pid, signal.SIGCONT)
                time.sleep(time_slice)
                
            if train_alive and check_processes_alive():
                log_info("[Scheduler] Pausing training")
                safe_kill(train_pid, signal.SIGSTOP)
                time.sleep(time_slice)
                
    elif mode == "adaptive_time":
        log_info(f"[GlobalScheduler] Mode: adaptive_time => Prioritizing training for first {global_scheduler.adaptive_params['priority_percent']*100:.0f}% of experiences.")
        
        # Initially pause evaluation, resume training
        if train_alive:
            safe_kill(train_pid, signal.SIGCONT)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGSTOP)
            
        priority_phase = True
        last_eval_check_time = time.time()
        
        while check_processes_alive():
            current_time = time.time()
            
            # Check and log shared_data status
            if shared_data:
                current_exp = shared_data.get("current_experience", 0)
                total_exps = shared_data.get("total_experiences", 10)
                if current_exp > 0 or total_exps > 0:
                    log_info(f"[GlobalScheduler] Current status: Experience {current_exp}/{total_exps}")
            
            # Periodically force evaluation execution
            if priority_phase and current_time - last_eval_check_time > FORCED_EVAL_INTERVAL:
                log_info(f"[GlobalScheduler] Forced evaluation check during priority phase")
                if eval_alive:
                    safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(EVAL_TRANSITION_TIME)  # Give time for evaluation to run
                last_eval_check_time = current_time
                
                # Return to train priority mode
                if train_alive:
                    safe_kill(train_pid, signal.SIGCONT)
            
            # Switch to alternating phase if any of the following conditions are met:
            # 1. Experience progress exceeds threshold (e.g., 30%)
            # 2. "all_experiences_completed" is True
            # 3. No experience data found
            if priority_phase and shared_data:
                # Find current experience and total experience count
                current_exp = shared_data.get("current_experience", 0)
                total_exps = shared_data.get("total_experiences", 10)
                
                # Calculate progress percentage based on total experiences (experience index starts from 0)
                # Example: Experience 0 = 0% progress, Last experience = 100% progress
                if total_exps > 1:  # Prevent division by zero
                    progress_percent = (current_exp) / (total_exps - 1) if total_exps > 1 else 0
                    priority_percent = global_scheduler.adaptive_params["priority_percent"]
                    
                    log_info(f"[GlobalScheduler] Experience progress: {progress_percent:.2f} (current: {current_exp}, total: {total_exps})")
                    
                    # Switch phase if progress exceeds threshold
                    if progress_percent >= priority_percent or \
                       shared_data.get("all_experiences_completed", False):
                        priority_phase = False
                        log_info(f"[GlobalScheduler] Switching to alternating mode. Progress: {progress_percent:.2f}, Threshold: {priority_percent:.2f}")
                else:
                    # Handle case with only one experience or none
                    if current_exp > 0 or shared_data.get("all_experiences_completed", False):
                        priority_phase = False
                        log_info("[GlobalScheduler] Switching to alternating mode (single experience or completed)")
            
            if not priority_phase:
                # After priority phase, alternate like in fully parallel mode
                log_info("[GlobalScheduler] Fully parallel mode.")
                safe_kill(train_pid, signal.SIGCONT)
                safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice)
            else:
                # In priority mode, focus on training but periodically check evaluation
                time.sleep(SCHEDULER_POLL_INTERVAL)
                
            # Check again before switching
            if not check_processes_alive():
                break

        log_info("[GlobalScheduler] Both processes have completed. Scheduler exiting.")
        sys.exit(0)
        
    elif mode == "adaptive_accuracy":
        log_info(f"[GlobalScheduler] Mode: adaptive_accuracy => Prioritizing training until {global_scheduler.adaptive_params['accuracy_threshold']*100:.0f}% accuracy.")
        
        # Initially pause evaluation, resume training
        if train_alive:
            safe_kill(train_pid, signal.SIGCONT)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGSTOP)
            
        priority_phase = True
        last_eval_check_time = time.time()
        accuracy_check_count = 0
        
        while check_processes_alive():
            current_time = time.time()
            
            # Periodically log accuracy information
            if shared_data and "latest_accuracy" in shared_data:
                latest_accuracy = shared_data.get("latest_accuracy", 0)
                threshold = global_scheduler.adaptive_params["accuracy_threshold"]
                log_info(f"[GlobalScheduler] Current accuracy: {latest_accuracy:.4f} (threshold: {threshold:.4f})")
            
            # Logic for accuracy checking in priority phase
            if priority_phase:
                # Periodically force evaluation
                if current_time - last_eval_check_time > FORCED_EVAL_INTERVAL:
                    accuracy_check_count += 1
                    log_info(f"[GlobalScheduler] Forced evaluation check #{accuracy_check_count} during priority phase")
                    
                    # Run evaluation process
                    if eval_alive:
                        safe_kill(eval_pid, signal.SIGCONT)
                        
                    # Give sufficient time for evaluation to run
                    time.sleep(EVAL_TRANSITION_TIME)
                    last_eval_check_time = current_time
                    
                    # Check accuracy
                    if shared_data and "latest_accuracy" in shared_data:
                        latest_accuracy = shared_data.get("latest_accuracy", 0)
                        threshold = global_scheduler.adaptive_params["accuracy_threshold"]
                        log_info(f"[GlobalScheduler] Accuracy check result: {latest_accuracy:.4f} / {threshold:.4f}")
                        
                        # Check if accuracy threshold reached
                        if latest_accuracy >= threshold:
                            priority_phase = False
                            log_info(f"[GlobalScheduler] Accuracy threshold reached! {latest_accuracy:.4f} >= {threshold:.4f}")
                            log_info("[GlobalScheduler] Switching to alternating mode")
                    
                    # Resume focus on training
                    if train_alive:
                        safe_kill(train_pid, signal.SIGCONT)
                        
                    # Force mode switch if accuracy check count exceeds limit without reaching threshold
                    # This prevents infinite loops
                    if accuracy_check_count >= MAX_ACCURACY_CHECKS and priority_phase:
                        log_info("[GlobalScheduler] Maximum accuracy checks reached without hitting threshold.")
                        log_info("[GlobalScheduler] Forcing switch to alternating mode for safety.")
                        priority_phase = False
                
                # Switch modes if training process finished
                if not train_alive:
                    priority_phase = False
                    log_info("[GlobalScheduler] Training process finished. Switching to alternating mode.")
                
                # Check status at short intervals in priority mode
                time.sleep(SCHEDULER_POLL_INTERVAL)
            else:
                # Switch to fully parallel mode after threshold is reached
                log_info("[GlobalScheduler] Running in fully parallel mode.")
                if train_alive:
                    safe_kill(train_pid, signal.SIGCONT)
                if eval_alive:
                    safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice)
            
            # Recheck process status
            if not check_processes_alive():
                break

        log_info("[GlobalScheduler] Both processes have completed. Scheduler exiting.")
        sys.exit(0)

    else:
        log_info("[Scheduler] Running alternating train/eval in time slices")
        if train_alive:
            safe_kill(train_pid, signal.SIGCONT)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGSTOP)
            
        while check_processes_alive():
            check_and_update_config()
            
            if train_alive:
                log_info("[Scheduler] Training slice: resuming training, pausing evaluation")
                safe_kill(train_pid, signal.SIGCONT)
                if eval_alive:
                    safe_kill(eval_pid, signal.SIGSTOP)
                time.sleep(time_slice)
                
            if not check_processes_alive():
                break
                
            if eval_alive:
                log_info("[Scheduler] Evaluation slice: pausing training, resuming evaluation")
                if train_alive:
                    safe_kill(train_pid, signal.SIGSTOP)
                safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice)
    
    log_info("[Scheduler] All processes completed")