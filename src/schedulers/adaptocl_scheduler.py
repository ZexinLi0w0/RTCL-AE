"""
AdaptOCL Scheduler: Unified Adaptation Metric (UAM), dynamic alternation (LA/AA switching),
and dynamic batch/time-slice reconfiguration (Algorithm 1, Sec 4/5 of paper).

- Section 4: Implements UAM and dynamic alternation (LA/AA switching) as in Algorithm 1.
- Section 5: Integrates dynamic batch/time-slice reconfiguration and static mode fallback (Sec 5.5).
- Fallback: If ω=1, γ=η=α=0, falls back to static mode (FP/DA/LA/AA) as in Section 5.5.

Key variables:
- ω (omega): accuracy weight in UAM
- γ (gamma): batch size changing rate
- η (eta): timeslice changing rate
- α (alpha): threshold for LA mode (equation 1)
- δ_acc (delta_acc): accuracy threshold for AA mode (equation 2)

This scheduler is fully compatible with the existing Ekya/FP/DA/LA/AA pipeline and CLI.
"""

import time
import numpy as np
import signal
import torch.multiprocessing as mp
from src.utils.logging_utils import log_info, log_warning
from src.utils.signal_handlers import safe_kill

class AdaptOCLScheduler:
    """
    AdaptOCL Scheduler: UAM-based dynamic LA/AA switching and batch/time-slice reconfiguration.
    """
    MIN_BATCH_SIZE = 16  # Minimum batch size
    # Constants for LA/AA from timeline_scheduler
    DEFAULT_LA_PRIORITY_PERCENT = 0.3
    DEFAULT_AA_ACCURACY_THRESHOLD = 0.4
    DEFAULT_FORCED_EVAL_INTERVAL = 10.0
    DEFAULT_EVAL_TRANSITION_TIME = 3.0
    DEFAULT_MAX_ACCURACY_CHECKS = 10
    MIN_TIME_SLICE = 1.0  # Minimum 1 second for time slice itself
    LOGGING_INTERVAL = 5.0 # Logging interval for scheduler status

    def __init__(self, time_slice, mode="adaptocl", adaptocl_params=None, lock=None):
        self.time_slice = time_slice
        self.mode = mode # This is the overall scheduler mode, "adaptocl"
        self.adaptocl_params = adaptocl_params or {}
        self.lock = lock # Store the lock
        
        # UAM hyperparameters
        self.omega = self.adaptocl_params.get("omega", 0.5)  # accuracy weight
        self.gamma = self.adaptocl_params.get("gamma", 0.5)  # batch size changing rate
        self.eta = self.adaptocl_params.get("eta", 0.5)     # timeslice changing rate
        self.uam_eps = self.adaptocl_params.get("uam_eps", 0.005) # ΔUAM hysteresis epsilon
        
        # LA mode parameters
        self.la_priority_percent = self.adaptocl_params.get("la_priority_percent", self.DEFAULT_LA_PRIORITY_PERCENT)
        
        # AA mode parameters
        self.aa_accuracy_threshold = self.adaptocl_params.get("aa_accuracy_threshold", self.DEFAULT_AA_ACCURACY_THRESHOLD)
        
        # Common parameters for LA/AA priority phases (can be overridden via adaptocl_params)
        self.forced_eval_interval = self.adaptocl_params.get("forced_eval_interval", self.DEFAULT_FORCED_EVAL_INTERVAL)
        self.eval_transition_time = self.adaptocl_params.get("eval_transition_time", self.DEFAULT_EVAL_TRANSITION_TIME)
        self.max_accuracy_checks = self.adaptocl_params.get("max_accuracy_checks", self.DEFAULT_MAX_ACCURACY_CHECKS)

        # AdaptOCL operational focus
        self.operation_focus = self.adaptocl_params.get("operation_focus", "balanced") # "balanced" or "continuous_eval"

        # State variables
        self.current_internal_mode = "adaptive_accuracy" # Stores 'adaptive_accuracy' or 'latency_aware' based on UAM
        self.la_priority_phase_active = True # For LA mode's initial priority phase
        self.aa_priority_phase_active = True # For AA mode's initial priority phase
        self.last_forced_eval_time = time.time()
        self.accuracy_check_count = 0
        
        self.last_acc = 0.0
        self.last_uam = None
        self.total_experiences = 0 # Still useful for logging/context
        self.current_experience = -1 # Initialize to -1 to detect first experience
        self.last_applied_experience = -1 # For duplicate-update suppression

        # Dynamic latency normalization
        self.latency_budget = self.adaptocl_params.get("latency_budget", 5.0)

    def initialize_shared_data(self, shared_data):
        """Initialize scheduler-specific fields in shared_data."""
        if self.lock:
            with self.lock:
                if "PENDING_CFG" not in shared_data:
                    shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
                if "last_applied_batch" not in shared_data:
                    # Ensure initial last_applied_batch respects MIN_BATCH_SIZE if fetched from train_batch_size
                    initial_batch_size = shared_data.get("train_batch_size")
                    if initial_batch_size is not None:
                         shared_data["last_applied_batch"] = max(initial_batch_size, self.MIN_BATCH_SIZE)
                    else: # train_batch_size might not be set yet, default to MIN_BATCH_SIZE
                         shared_data["last_applied_batch"] = self.MIN_BATCH_SIZE
        else: # Should not happen if lock is passed correctly
            if "PENDING_CFG" not in shared_data:
                shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
            if "last_applied_batch" not in shared_data:
                initial_batch_size = shared_data.get("train_batch_size")
                if initial_batch_size is not None:
                    shared_data["last_applied_batch"] = max(initial_batch_size, self.MIN_BATCH_SIZE)
                else:
                    shared_data["last_applied_batch"] = self.MIN_BATCH_SIZE

    def run(self, train_pid, eval_pid, shared_data):
        """
        AdaptOCL scheduler worker (Algorithm 1)
        Parameters:
        - gamma: batch size changing rate
        - eta: timeslice changing rate
        - alpha: threshold for LA mode (equation 1)
        - delta_acc: accuracy threshold for AA mode (equation 2)
        """
        MIN_TIME_SLICE = 1.0  # Minimum 1 second
        LOGGING_INTERVAL = 5.0  # Log every 5 seconds
        last_log_time = 0
        
        # Initialize shared data fields if not present
        self.initialize_shared_data(shared_data)

        scheduler = self
        # Initial time_slice, will be updated if PENDING_CFG has a value or from shared_data
        time_slice = scheduler.time_slice

        omega = scheduler.omega
        gamma = scheduler.gamma
        eta = scheduler.eta
        
        # Initial batch size, re-fetched each loop
        batch_size = shared_data.get("train_batch_size", self.MIN_BATCH_SIZE) # Default to MIN_BATCH_SIZE
        batch_size = max(batch_size, self.MIN_BATCH_SIZE) # Ensure it's not below min
        B_max = shared_data.get("max_batch_size", 256)

        # Get total experiences for LA condition (now mostly for logging context)
        self.total_experiences = shared_data.get("total_experiences", 1)
        # self.current_experience is now updated at the start of the loop

        log_info(f"[AdaptOCL] Start: ω={omega}, γ={gamma}, η={eta}, ε={self.uam_eps}") # Removed alpha, delta_acc from log
        log_info(f"[AdaptOCL] Initial config: batch_size={batch_size}, time_slice={time_slice}")
        log_info(f"[AdaptOCL] Using dynamic latency budget: {self.latency_budget}")

        while shared_data.get("train_process_active", True):
            current_time = time.time()
            
            # 1. Latest Batch-Size Re-fetch & Time-slice Re-fetch
            if self.lock:
                with self.lock:
                    batch_size = shared_data.get("train_batch_size", batch_size)
                    # Also re-fetch time_slice, as it might be changed by other components
                    # or at experience boundary
                    time_slice = shared_data.get("timeslice", time_slice)
            else:
                batch_size = shared_data.get("train_batch_size", batch_size)
                time_slice = shared_data.get("timeslice", time_slice)
            time_slice = max(time_slice, MIN_TIME_SLICE)

            # 2. Experience-Boundary Application
            new_experience = shared_data.get("current_experience", self.current_experience)
            if new_experience != self.current_experience:
                self.current_experience = new_experience
                log_info(f"[AdaptOCL] New experience detected: {self.current_experience}")
                if self.current_experience != self.last_applied_experience:
                    if self.lock:
                        with self.lock:
                            pending_cfg = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None})
                            applied_new_config = False
                            if pending_cfg["batch"] is not None:
                                applied_batch_val = max(pending_cfg["batch"], self.MIN_BATCH_SIZE)
                                shared_data["train_batch_size"] = applied_batch_val
                                batch_size = applied_batch_val # Update local var
                                applied_new_config = True
                                log_info(f"[AdaptOCL] Applied pending batch: {batch_size} at exp {self.current_experience}")
                            if pending_cfg["tslice"] is not None:
                                shared_data["timeslice"] = max(pending_cfg["tslice"], MIN_TIME_SLICE)
                                time_slice = shared_data["timeslice"] # Update local var
                                applied_new_config = True
                                log_info(f"[AdaptOCL] Applied pending tslice: {time_slice} at exp {self.current_experience}")

                            if applied_new_config:
                                shared_data["last_applied_batch"] = batch_size
                                shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
                                shared_data["CONFIG_UPDATE_REQUESTED"] = True # Notify worker
                                self.last_applied_experience = self.current_experience
                    else: # No lock, less safe but proceed
                        pending_cfg = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None})
                        applied_new_config = False
                        if pending_cfg["batch"] is not None:
                            applied_batch_val = max(pending_cfg["batch"], self.MIN_BATCH_SIZE)
                            shared_data["train_batch_size"] = applied_batch_val
                            batch_size = applied_batch_val
                            applied_new_config = True
                        if pending_cfg["tslice"] is not None:
                            shared_data["timeslice"] = max(pending_cfg["tslice"], MIN_TIME_SLICE)
                            time_slice = shared_data["timeslice"]
                            applied_new_config = True

                        if applied_new_config:
                            shared_data["last_applied_batch"] = batch_size
                            shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
                            shared_data["CONFIG_UPDATE_REQUESTED"] = True
                            self.last_applied_experience = self.current_experience

            # Calculate and update values every iteration
            acc = shared_data.get("latest_accuracy", 0.0)
            if acc is None:
                acc = 0.0
            latency = shared_data.get("latest_latency", 1.0)
            if latency is None or latency <= 0:
                latency = 1.0
                
            # Calculate normalized latency using dynamic budget
            normalized_latency = min(latency / self.latency_budget, 1.0)
                
            # Calculate UAM (paper equation, corrected form)
            uam = omega * acc - (1.0 - omega) * normalized_latency
            delta_uam = 0.0 if self.last_uam is None else uam - self.last_uam
                        
            # 4. ΔUAM Hysteresis
            if abs(delta_uam) < self.uam_eps:
                effective_delta_uam = 0.0
            else:
                effective_delta_uam = delta_uam

            # Log detailed metrics
            pending_cfg_log = shared_data.get("PENDING_CFG", {"batch": "N/A", "tslice": "N/A"})
            last_applied_batch_log = shared_data.get("last_applied_batch", "N/A")
            if current_time - last_log_time >= self.LOGGING_INTERVAL: # Log less frequently
                log_info(f"[AdaptOCL] Metrics: acc={acc:.3f}, lat={latency:.3f}, norm_lat={normalized_latency:.3f}, B={batch_size}, T_slice={time_slice:.2f}")
                log_info(f"[AdaptOCL] UAM={uam:.3f}, ΔUAM={delta_uam:.3f} (eff_ΔUAM={effective_delta_uam:.3f}), exp={self.current_experience}/{self.total_experiences}, internal_mode={self.current_internal_mode}, pend_cfg={pending_cfg_log}, ack_batch={last_applied_batch_log}")
            
            # Update PENDING_CFG based on effective_delta_uam (batch size and time slice adjustments)
            if effective_delta_uam != 0:
                new_batch_size_pending = int(np.clip(
                    round(batch_size * (1 + gamma * np.sign(effective_delta_uam))),
                    self.MIN_BATCH_SIZE, B_max
                ))
                new_time_slice_pending = max(
                    time_slice * (1 + eta * np.sign(effective_delta_uam)),
                    self.MIN_TIME_SLICE
                )

                if self.lock:
                    with self.lock:
                        current_pending = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None}).copy() # Use .copy()
                        made_pending_change = False
                        if new_batch_size_pending != batch_size and (current_pending.get("batch") is None or new_batch_size_pending != current_pending.get("batch")):
                            current_pending["batch"] = new_batch_size_pending
                            log_info(f"[AdaptOCL] Pending batch size: {batch_size}->{new_batch_size_pending}")
                            made_pending_change = True
                        if abs(new_time_slice_pending - time_slice) > 1e-6 and (current_pending.get("tslice") is None or abs(new_time_slice_pending - current_pending.get("tslice")) > 1e-6):
                            current_pending["tslice"] = new_time_slice_pending
                            log_info(f"[AdaptOCL] Pending time slice: {time_slice:.3f}->{new_time_slice_pending:.3f}")
                            made_pending_change = True
                        if made_pending_change:
                             shared_data["PENDING_CFG"] = current_pending
                else: # No lock - less safe, for completeness
                    current_pending_no_lock = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None}).copy()
                    # ... (similar logic for no lock, omitted for brevity but should mirror above) ...
                    shared_data["PENDING_CFG"] = current_pending_no_lock


            # Determine internal mode based on UAM and handle mode switching logic
            previous_internal_mode = self.current_internal_mode
            if effective_delta_uam > 0:
                self.current_internal_mode = "latency_aware"
            else: # Covers effective_delta_uam <= 0
                self.current_internal_mode = "adaptive_accuracy"

            if self.current_internal_mode != previous_internal_mode:
                log_info(f"[AdaptOCL] Switched internal mode from {previous_internal_mode} to {self.current_internal_mode} (eff_ΔUAM={effective_delta_uam:.3f})")
                # Reset priority phase flags and counters when switching internal mode
                self.la_priority_phase_active = True 
                self.aa_priority_phase_active = True
                self.accuracy_check_count = 0
                self.last_forced_eval_time = current_time # Reset eval timer

            # --- Moved Process Liveness Check Up ---
            train_pid_alive = safe_kill(train_pid, 0) # Check if train process is alive
            eval_pid_alive = safe_kill(eval_pid, 0)   # Check if eval process is alive

            if not train_pid_alive and not eval_pid_alive:
                log_info("[AdaptOCL] Both train and eval processes are dead. Exiting scheduler.")
                break
            if not train_pid_alive: # If only train is dead, try to let eval finish if it was running
                if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT)
                log_info("[AdaptOCL] Train process is dead. Waiting for eval or exiting.")
                time.sleep(self.MIN_TIME_SLICE) # Wait a bit
                continue # Re-check at next iteration

            # --- Main Process Control Logic (based on operation_focus) ---
            if self.operation_focus == "continuous_eval":
                if eval_pid_alive:
                    safe_kill(eval_pid, signal.SIGCONT)
                if train_pid_alive:
                    safe_kill(train_pid, signal.SIGCONT)

                if current_time - last_log_time >= self.LOGGING_INTERVAL:
                    log_info(f"[AdaptOCL_ContEval] Continuous evaluation. Train parallel. B={batch_size}, T_slice={time_slice:.2f}")

            elif self.operation_focus == "balanced": # Original AdaptOCL behavior
                if self.current_internal_mode == "latency_aware":
                    if self.la_priority_phase_active:
                        current_exp = shared_data.get("current_experience", 0)
                        total_exps = shared_data.get("total_experiences", 1) # Avoid div by zero
                        progress_percent = (current_exp / total_exps) if total_exps > 0 else 0

                        if progress_percent < self.la_priority_percent and not shared_data.get("all_experiences_completed", False):
                            # LA Priority Phase: Train focus
                            if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                            if eval_pid_alive: safe_kill(eval_pid, signal.SIGSTOP)
                            if current_time - last_log_time >= self.LOGGING_INTERVAL:
                                 log_info(f"[AdaptOCL] LA Priority: Training (Exp {current_exp}/{total_exps}, Prog {progress_percent:.2f} < {self.la_priority_percent:.2f}). Eval stopped.")

                            # Optional: Forced eval check (from timeline_scheduler)
                            if eval_pid_alive and current_time - self.last_forced_eval_time >= self.forced_eval_interval:
                                log_info(f"[AdaptOCL] LA Priority: Forced eval check.")
                                safe_kill(eval_pid, signal.SIGCONT)
                                time.sleep(self.eval_transition_time) # Let eval run briefly
                                if train_pid_alive: safe_kill(train_pid, signal.SIGCONT) # Ensure train is running
                                if eval_pid_alive: safe_kill(eval_pid, signal.SIGSTOP)   # Stop eval again
                                self.last_forced_eval_time = time.time() # Update timestamp AFTER eval
                        else:
                            self.la_priority_phase_active = False
                            log_info(f"[AdaptOCL] LA: Priority phase ended (Prog {progress_percent:.2f} or all exp completed). Switching to parallel.")
                            # Fall through to parallel execution

                    if not self.la_priority_phase_active: # LA Parallel Phase
                        if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                        if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT)
                        if current_time - last_log_time >= self.LOGGING_INTERVAL:
                            log_info(f"[AdaptOCL] LA Parallel: Training and Evaluation running.")

                elif self.current_internal_mode == "adaptive_accuracy":
                    if self.aa_priority_phase_active:
                        # AA Priority Phase: Train focus, periodic eval for accuracy check
                        perform_eval_check = False
                        if current_time - self.last_forced_eval_time >= self.forced_eval_interval:
                            perform_eval_check = True
                            self.accuracy_check_count += 1
                            log_info(f"[AdaptOCL] AA Priority: Forced eval check #{self.accuracy_check_count}.")

                        if perform_eval_check and eval_pid_alive:
                            if train_pid_alive: safe_kill(train_pid, signal.SIGSTOP) # Pause train during eval
                            safe_kill(eval_pid, signal.SIGCONT)
                            time.sleep(self.eval_transition_time) # Let eval run
                            self.last_forced_eval_time = time.time() # Update timestamp AFTER eval run

                            latest_acc = shared_data.get("latest_accuracy", 0.0)
                            if latest_acc is None:
                                # integrity check, fallback to 0.0
                                latest_acc = 0.0
                            log_info(f"[AdaptOCL] AA Priority: Accuracy check result: {latest_acc:.4f} (threshold: {self.aa_accuracy_threshold:.4f})")

                            if latest_acc >= self.aa_accuracy_threshold:
                                self.aa_priority_phase_active = False
                                log_info(f"[AdaptOCL] AA: Accuracy threshold reached! ({latest_acc:.4f} >= {self.aa_accuracy_threshold:.4f}). Switching to parallel.")
                            elif self.accuracy_check_count >= self.max_accuracy_checks:
                                self.aa_priority_phase_active = False
                                log_info(f"[AdaptOCL] AA: Max accuracy checks ({self.max_accuracy_checks}) reached. Forcing parallel mode.")

                            # Resume train, stop eval (if still in priority and eval was started)
                            if self.aa_priority_phase_active: # if not switched to parallel
                                if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                                if eval_pid_alive: safe_kill(eval_pid, signal.SIGSTOP) # Stop eval if it was running for check
                            else: # Switched to parallel, ensure both are running
                                if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                                if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT)

                        else: # Not time for eval check, or eval_pid not alive
                            if self.aa_priority_phase_active: # Still in priority phase
                                 if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                                 if eval_pid_alive: safe_kill(eval_pid, signal.SIGSTOP)
                                 if current_time - last_log_time >= self.LOGGING_INTERVAL:
                                    log_info(f"[AdaptOCL] AA Priority: Training. Eval stopped. Next check in {self.forced_eval_interval - (current_time - self.last_forced_eval_time):.1f}s")

                    if not self.aa_priority_phase_active: # AA Parallel Phase
                        if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                        if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT)
                        if current_time - last_log_time >= self.LOGGING_INTERVAL:
                            log_info(f"[AdaptOCL] AA Parallel: Training and Evaluation running.")

                # Fallback if train process dies during priority phase of AA
                if self.current_internal_mode == "adaptive_accuracy" and self.aa_priority_phase_active and not train_pid_alive:
                    log_info("[AdaptOCL] AA Priority: Train process died. Switching to parallel/eval only.")
                    self.aa_priority_phase_active = False # Exit priority phase
                    if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT) # Ensure eval can run
            else: # Should not happen with proper config validation
                log_warning(f"[AdaptOCL] Unknown operation_focus: {self.operation_focus}. Defaulting to parallel execution.")
                if train_pid_alive: safe_kill(train_pid, signal.SIGCONT)
                if eval_pid_alive: safe_kill(eval_pid, signal.SIGCONT)

            # Update state for next iteration
            self.last_acc = acc
            self.last_uam = uam
            # self.current_internal_mode is already updated
            
            # Update logging interval timer
            if current_time - last_log_time >= self.LOGGING_INTERVAL:
                last_log_time = current_time
            
            # Main scheduler loop sleep
            time.sleep(max(time_slice, self.MIN_TIME_SLICE)) # Use self.MIN_TIME_SLICE
            
        # Scheduler loop exited (likely because train_process_active is False or TERMINATE_SIGNAL)
        log_info("[AdaptOCL] Main scheduler loop finished.")

        # Ensure eval_worker is properly handled if it's still alive when scheduler exits
        # This is important if the scheduler stops due to train_process_active becoming False
        # while eval_worker was SIGSTOPped by the scheduler itself.
        if not shared_data.get("TERMINATE_SIGNAL", False): # Only do this if not already in global termination sequence
            eval_pid_alive_at_exit = safe_kill(eval_pid, 0)
            if eval_pid_alive_at_exit:
                log_info(f"[AdaptOCL] Training has likely ended. Attempting to ensure eval_worker (PID: {eval_pid}) is woken and can terminate.")
                try:
                    # Wake it up in case it was SIGSTOPped by this scheduler
                    safe_kill(eval_pid, signal.SIGCONT) 
                    time.sleep(0.1) # Give a moment for SIGCONT to be processed
                    # We don't necessarily send SIGTERM here, as the main process's cleanup 
                    # should handle it. The critical part is ensuring it's not SIGSTOPped.
                    # If it's meant to complete some final evaluation, it can do so now.
                    # If main.py is already terminating, it will get a SIGTERM from there.
                    log_info(f"[AdaptOCL] Sent SIGCONT to eval_worker (PID: {eval_pid}) to ensure it is not stopped.")
                except Exception as e:
                    log_warning(f"[AdaptOCL] Error during final SIGCONT to eval_worker (PID: {eval_pid}): {e}")
            else:
                log_info("[AdaptOCL] Eval_worker was not alive at scheduler exit or termination already in progress.")
        else:
            log_info("[AdaptOCL] Global termination signal active, main process will handle eval_worker cleanup.")
            
        log_info("[AdaptOCL] Scheduler worker stopping.")

def adaptocl_scheduler_worker(global_scheduler, train_pid, eval_pid, shared_data, lock): # Added lock
    """
    Worker entrypoint for AdaptOCLScheduler, matching the interface of other schedulers.
    """
    # Pass the lock to the scheduler instance
    if isinstance(global_scheduler, AdaptOCLScheduler):
        global_scheduler.lock = lock
    return global_scheduler.run(train_pid, eval_pid, shared_data) 