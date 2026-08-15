import time
import signal
import math
import os # For os.getpid() in eval_worker for shared_data key

from src.utils.signal_handlers import safe_kill
from src.utils.logging_utils import log_info, log_warning

class RECLScheduler:
    """
    RECL-SCHED: GPU µ-window time-slicing between eval and train workers.
    - No model-reuse logic from the full RECL paper.
    - Aims to keep batch-size interaction within workers intact by time-slicing GPU access.
    """
    LOGGING_INTERVAL = 5.0 # Logging interval for scheduler status

    def __init__(self, train_pid, eval_pid, args, shared_data, lock=None): # Added lock for consistency
        self.train_pid = train_pid
        self.eval_pid = eval_pid
        self.args = args # Store the full args object
        self.shared_data = shared_data
        # self.lock = lock # Not strictly used in this simple version, but good practice

        self.micro_window_sec = args.recl_micro_ms / 1000.0
        self.current_eval_weight = args.recl_eval_weight # Initial weight
        self.adapt_alpha = args.recl_adapt_alpha
        
        # New parameters for operation focus
        self.operation_focus = args.recl_operation_focus
        self.min_eval_weight = args.recl_min_eval_weight
        
        # History for SLO miss ratio calculation (last 100 observations)
        self.slo_miss_history = []
        self.max_history_len = 100

        # Initial check for necessary shared_data keys (main.py should ensure these are initialized)
        if 'eval_latency' not in self.shared_data:
            log_warning("[RECL_SCHED] 'eval_latency' not found in shared_data. Adaptive weight may not work.")
            # Potentially initialize if manager allows, but typically main does this.
            # self.shared_data['eval_latency'] = mp.Manager().dict() # Requires mp
        if 'eval_slo_ms' not in self.shared_data:
            log_warning("[RECL_SCHED] 'eval_slo_ms' not found in shared_data. Adaptive weight may not work.")
            self.shared_data['eval_slo_ms'] = args.slo_ms # Store it from args if not present

        log_info(f"[RECL_SCHED] Initialized: µ-window={self.micro_window_sec:.3f}s, eval_weight={self.current_eval_weight:.2f}, alpha={self.adapt_alpha:.2f}, SLO={self.shared_data.get('eval_slo_ms')}ms")

    def _grant_gpu(self, run_pid, stop_pid):
        """Grants GPU to run_pid by sending SIGCONT, and pauses stop_pid with SIGSTOP."""
        if safe_kill(run_pid, 0): # Check if process exists
            safe_kill(run_pid, signal.SIGCONT)
        if safe_kill(stop_pid, 0): # Check if process exists
            safe_kill(stop_pid, signal.SIGSTOP)

    def _update_miss_ratio_and_weight(self):
        """Updates the SLO miss ratio and adjusts the evaluation weight accordingly."""
        eval_pid_latency_ms = self.shared_data.get('eval_latency', {}).get(str(self.eval_pid)) # Keyed by PID as string
        slo_ms = self.shared_data.get('eval_slo_ms', self.args.slo_ms) # Default to args.slo_ms if not in shared

        if eval_pid_latency_ms is None:
            # No new latency reported for this eval_pid, don't update history or weight this cycle
            return

        # Add current observation to history
        is_miss = 1 if eval_pid_latency_ms > slo_ms else 0
        self.slo_miss_history.append(is_miss)
        if len(self.slo_miss_history) > self.max_history_len:
            self.slo_miss_history.pop(0)
        
        # Clear the reported latency for this PID so we only process it once
        # This assumes eval_worker writes and then this scheduler consumes.
        # A more robust way might involve timestamps or sequence numbers if multiple consumers.
        if str(self.eval_pid) in self.shared_data.get('eval_latency', {}):
             # This requires shared_data['eval_latency'] to be a mutable proxy (like Manager.dict)
             # For simplicity, we'll assume it's handled by the caller or main.py structure.
             # A direct del self.shared_data['eval_latency'][str(self.eval_pid)] might fail if not a proxy.
             # Or, eval_worker could set it to None after we read.
             # For now, we just use the value and it will be overwritten by eval_worker.
             pass


        if not self.slo_miss_history:
            return

        miss_ratio = sum(self.slo_miss_history) / len(self.slo_miss_history)
        
        previous_weight = self.current_eval_weight

        if self.operation_focus == "balanced":
            if miss_ratio > self.adapt_alpha:
                self.current_eval_weight = min(self.current_eval_weight + 0.1, 0.9)
            elif miss_ratio < (self.adapt_alpha / 2.0): # If significantly below, reduce eval time
                self.current_eval_weight = max(self.current_eval_weight - 0.1, 0.1)
        elif self.operation_focus == "inference_focus":
            # In inference_focus, adapt modestly and maintain high eval weight
            if miss_ratio > self.adapt_alpha:
                # Increase eval weight, up to 1.0
                self.current_eval_weight = min(self.current_eval_weight + 0.05, 1.0)
            elif miss_ratio < (self.adapt_alpha / 2.0): 
                # Decrease eval weight, but not below min_eval_weight
                self.current_eval_weight = max(self.current_eval_weight - 0.05, self.min_eval_weight)
            # Ensure it never goes below min_eval_weight if somehow initialized lower or due to other logic
            self.current_eval_weight = max(self.current_eval_weight, self.min_eval_weight)
        else: # Should not happen if choices are enforced by argparse
            log_warning(f"[RECL_SCHED] Unknown operation_focus: {self.operation_focus}. Defaulting to balanced behavior.")
            if miss_ratio > self.adapt_alpha:
                self.current_eval_weight = min(self.current_eval_weight + 0.1, 0.9)
            elif miss_ratio < (self.adapt_alpha / 2.0):
                self.current_eval_weight = max(self.current_eval_weight - 0.1, 0.1)
        
        if abs(previous_weight - self.current_eval_weight) > 1e-3 : # If changed
             log_info(f"[RECL_SCHED] SLO Miss Ratio: {miss_ratio:.2f}. Eval weight adapted from {previous_weight:.2f} to {self.current_eval_weight:.2f}")


    def run(self):
        """Main scheduler loop."""
        log_info("[RECL_SCHED] Starting scheduler loop.")
        window_index = 0
        last_log_time = time.time()

        try:
            while True:
                # Termination signal should take precedence
                if self.shared_data.get("TERMINATE_SIGNAL", False):
                    log_info("[RECL_SCHED] TERMINATE_SIGNAL received. Exiting scheduler loop immediately.")
                    break

                train_worker_alive = self.train_pid is not None and safe_kill(self.train_pid, 0)
                eval_worker_alive = self.eval_pid is not None and safe_kill(self.eval_pid, 0)
                
                # Optional: Log worker states for debugging
                # log_info(f"[RECL_SCHED_DEBUG] train_worker_alive: {train_worker_alive}, eval_worker_alive: {eval_worker_alive}")

                if not train_worker_alive and not eval_worker_alive:
                    log_info("[RECL_SCHED] Both train and eval workers are no longer alive. Exiting scheduler loop.")
                    break
                
                # 1. Determine which process gets the GPU for this µ-window
                # As per user request, though current_eval_weight should already reflect the mode.
                if self.operation_focus == "inference_focus":
                    # current_eval_weight will be high (e.g., 0.9-1.0) due to _update_miss_ratio_and_weight logic
                    eval_slots = int(round(self.current_eval_weight * 100))
                else: # balanced mode
                    eval_slots = int(round(self.current_eval_weight * 100))
                
                # Only grant GPU if the respective worker is alive
                if (window_index % 100) < eval_slots:
                    if eval_worker_alive:
                        self._grant_gpu(self.eval_pid, self.train_pid if train_worker_alive else None)
                        if time.time() - last_log_time > self.LOGGING_INTERVAL:
                            log_info(f"[RECL_SCHED] µ-window for EVAL (Weight: {self.current_eval_weight:.2f})")
                    elif train_worker_alive: # If eval is supposed to run but isn't alive, give to train if it is
                        self._grant_gpu(self.train_pid, None) # Eval worker is dead, so nothing to stop for it
                        if time.time() - last_log_time > self.LOGGING_INTERVAL:
                            log_info(f"[RECL_SCHED] µ-window for TRAIN (eval dead, Weight: {self.current_eval_weight:.2f})")
                    # If neither is alive, the loop should have broken already.
                else:
                    if train_worker_alive:
                        self._grant_gpu(self.train_pid, self.eval_pid if eval_worker_alive else None)
                        if time.time() - last_log_time > self.LOGGING_INTERVAL:
                            log_info(f"[RECL_SCHED] µ-window for TRAIN (Weight: {self.current_eval_weight:.2f})")
                    elif eval_worker_alive: # If train is supposed to run but isn't alive, give to eval if it is
                        self._grant_gpu(self.eval_pid, None) # Train worker is dead
                        if time.time() - last_log_time > self.LOGGING_INTERVAL:
                            log_info(f"[RECL_SCHED] µ-window for EVAL (train dead, Weight: {self.current_eval_weight:.2f})")
                    # If neither is alive, the loop should have broken already.
                
                if time.time() - last_log_time > self.LOGGING_INTERVAL:
                    last_log_time = time.time()

                # Sleep for the duration of the µ-window
                time.sleep(self.micro_window_sec)
                window_index += 1

                # 2. Update SLO miss ratio and adapt evaluation weight (only if eval worker is expected to be alive)
                if eval_worker_alive: # Or based on whether eval_latency was recently updated for this eval_pid
                    self._update_miss_ratio_and_weight()

        except KeyboardInterrupt:
            log_info("[RECL_SCHED] Interrupted by user.")
        finally:
            log_info("[RECL_SCHED] Scheduler loop finished. Ensuring workers are not left stopped.")
            # Resume workers only if they were found to be alive by safe_kill in the last check before exiting,
            # or more simply, just try to SIGCONT if their PIDs are known.
            # The safe_kill(pid, 0) check inside _grant_gpu already handles non-existent processes.
            # Here, we ensure they are not left in a SIGSTOP state if the scheduler exits abruptly.
            if self.train_pid is not None and safe_kill(self.train_pid, 0): # Check if alive before sending SIGCONT
                 safe_kill(self.train_pid, signal.SIGCONT)
            if self.eval_pid is not None and safe_kill(self.eval_pid, 0): # Check if alive before sending SIGCONT
                 safe_kill(self.eval_pid, signal.SIGCONT)
            log_info("[RECL_SCHED] Scheduler stopping.")

def recl_scheduler_worker(global_scheduler, train_pid, eval_pid, shared_data, lock, args_obj): # Match AdaptOCL worker signature
    """
    Worker entrypoint for RECLScheduler.
    'global_scheduler' here would be an instance of RECLScheduler,
    but it's not pre-initialized like in AdaptOCL. We create it.
    This function should be the target of the scheduler process.
    """
    # The 'global_scheduler' argument in this context is a bit of a misnomer
    # if we're creating the scheduler instance here.
    # Let's assume this function itself is the effective "run" method for the process.
    # We need args to initialize RECLScheduler properly.
    
    # If global_scheduler is already an instance (e.g. from main.py if pre-created)
    # if isinstance(global_scheduler, RECLScheduler):
    #    scheduler_instance = global_scheduler
    # else: # Create instance here
    
    # The prompt implies main.py will create RECLBatchScheduler and then a process targets sched.run
    # So, this worker function might not be needed if main.py directly calls instantiated_scheduler.run()
    # For now, let's assume main.py creates the RECLScheduler instance,
    # then creates a Process targeting *that instance's* run method.
    # This 'recl_scheduler_worker' function is more of a pattern from AdaptOCL.
    # If main.py does:
    #   sched = RECLScheduler(train_proc.pid, eval_proc.pid, args, shared)
    #   sched_proc = mp.Process(target=sched.run)
    # then this function is not directly used.
    #
    # Let's provide it just in case the main structure expects a callable like this.
    # It will need 'args' passed to it.
    
    scheduler = RECLScheduler(train_pid, eval_pid, args_obj, shared_data, lock)
    return scheduler.run() 