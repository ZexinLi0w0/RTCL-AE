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
from contextlib import nullcontext
from src.utils.logging_utils import log_info, log_warning
from src.utils.signal_handlers import safe_kill, is_process_alive

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

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _locked(self):
        """Context manager for the shared-data lock (no-op if no lock was given)."""
        return self.lock if self.lock else nullcontext()

    @staticmethod
    def _signal(pid, alive, sig):
        """Send `sig` to `pid` only if the worker is known to be alive."""
        if alive:
            safe_kill(pid, sig)

    def _run_both(self, train_pid, eval_pid, train_alive, eval_alive):
        """Parallel phase: training and evaluation both run."""
        self._signal(train_pid, train_alive, signal.SIGCONT)
        self._signal(eval_pid, eval_alive, signal.SIGCONT)

    def _train_only(self, train_pid, eval_pid, train_alive, eval_alive):
        """Priority phase: training runs, evaluation is paused."""
        self._signal(train_pid, train_alive, signal.SIGCONT)
        self._signal(eval_pid, eval_alive, signal.SIGSTOP)

    def initialize_shared_data(self, shared_data):
        """Initialize scheduler-specific fields in shared_data."""
        with self._locked():
            if "PENDING_CFG" not in shared_data:
                shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
            if "last_applied_batch" not in shared_data:
                # train_batch_size might not be set yet; respect MIN_BATCH_SIZE either way
                initial_batch_size = shared_data.get("train_batch_size")
                if initial_batch_size is not None:
                    shared_data["last_applied_batch"] = max(initial_batch_size, self.MIN_BATCH_SIZE)
                else:
                    shared_data["last_applied_batch"] = self.MIN_BATCH_SIZE

    # ------------------------------------------------------------------
    # Algorithm 1 steps
    # ------------------------------------------------------------------
    def _refresh_config(self, shared_data, batch_size, time_slice):
        """Step 1: re-fetch the batch size and time slice (other components may change them)."""
        with self._locked():
            batch_size = shared_data.get("train_batch_size", batch_size)
            time_slice = shared_data.get("timeslice", time_slice)
        return batch_size, max(time_slice, self.MIN_TIME_SLICE)

    def _apply_pending_at_experience_boundary(self, shared_data, batch_size, time_slice):
        """
        Step 2: when a new experience starts, commit the staged PENDING_CFG
        (batch size / time slice) and ask the workers to pick it up.
        """
        new_experience = shared_data.get("current_experience", self.current_experience)
        if new_experience == self.current_experience:
            return batch_size, time_slice
        self.current_experience = new_experience
        log_info(f"[AdaptOCL] New experience detected: {self.current_experience}")
        if self.current_experience == self.last_applied_experience:
            return batch_size, time_slice

        with self._locked():
            pending_cfg = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None})
            applied_new_config = False
            if pending_cfg["batch"] is not None:
                batch_size = max(pending_cfg["batch"], self.MIN_BATCH_SIZE)
                shared_data["train_batch_size"] = batch_size
                applied_new_config = True
                log_info(f"[AdaptOCL] Applied pending batch: {batch_size} at exp {self.current_experience}")
            if pending_cfg["tslice"] is not None:
                shared_data["timeslice"] = max(pending_cfg["tslice"], self.MIN_TIME_SLICE)
                time_slice = shared_data["timeslice"]
                applied_new_config = True
                log_info(f"[AdaptOCL] Applied pending tslice: {time_slice} at exp {self.current_experience}")

            if applied_new_config:
                shared_data["last_applied_batch"] = batch_size
                shared_data["PENDING_CFG"] = {"batch": None, "tslice": None}
                shared_data["CONFIG_UPDATE_REQUESTED"] = True  # Notify worker
                self.last_applied_experience = self.current_experience
        return batch_size, time_slice

    def _compute_uam(self, shared_data):
        """
        Step 3: UAM = omega * acc - (1 - omega) * normalized_latency, its change
        since the last tick, and the change after epsilon hysteresis (step 4).
        """
        acc = shared_data.get("latest_accuracy", 0.0)
        if acc is None:
            acc = 0.0
        latency = shared_data.get("latest_latency", 1.0)
        if latency is None or latency <= 0:
            latency = 1.0
        normalized_latency = min(latency / self.latency_budget, 1.0)

        uam = self.omega * acc - (1.0 - self.omega) * normalized_latency
        delta_uam = 0.0 if self.last_uam is None else uam - self.last_uam
        effective_delta_uam = 0.0 if abs(delta_uam) < self.uam_eps else delta_uam
        return acc, latency, normalized_latency, uam, delta_uam, effective_delta_uam

    def _stage_pending_config(self, shared_data, batch_size, time_slice, effective_delta_uam, B_max):
        """
        Step 5: grow (dUAM > 0) or shrink (dUAM < 0) batch size by gamma and time
        slice by eta. The new values are only staged in PENDING_CFG; they are
        committed at the next experience boundary.
        """
        direction = np.sign(effective_delta_uam)
        new_batch_size = int(np.clip(round(batch_size * (1 + self.gamma * direction)), self.MIN_BATCH_SIZE, B_max))
        new_time_slice = max(time_slice * (1 + self.eta * direction), self.MIN_TIME_SLICE)

        with self._locked():
            pending = shared_data.get("PENDING_CFG", {"batch": None, "tslice": None}).copy()
            changed = False
            if new_batch_size != batch_size and (pending.get("batch") is None or new_batch_size != pending.get("batch")):
                pending["batch"] = new_batch_size
                log_info(f"[AdaptOCL] Pending batch size: {batch_size}->{new_batch_size}")
                changed = True
            if abs(new_time_slice - time_slice) > 1e-6 and (pending.get("tslice") is None or abs(new_time_slice - pending.get("tslice")) > 1e-6):
                pending["tslice"] = new_time_slice
                log_info(f"[AdaptOCL] Pending time slice: {time_slice:.3f}->{new_time_slice:.3f}")
                changed = True
            if changed:
                shared_data["PENDING_CFG"] = pending

    def _update_internal_mode(self, effective_delta_uam, current_time):
        """Step 6: dUAM > 0 selects latency-aware (LA) mode, otherwise accuracy-aware (AA)."""
        previous_mode = self.current_internal_mode
        self.current_internal_mode = "latency_aware" if effective_delta_uam > 0 else "adaptive_accuracy"
        if self.current_internal_mode != previous_mode:
            log_info(f"[AdaptOCL] Switched internal mode from {previous_mode} to {self.current_internal_mode} (eff_ΔUAM={effective_delta_uam:.3f})")
            # A mode switch restarts the priority phase of the new mode
            self.la_priority_phase_active = True
            self.aa_priority_phase_active = True
            self.accuracy_check_count = 0
            self.last_forced_eval_time = current_time

    # ------------------------------------------------------------------
    # Worker control (step 7)
    # ------------------------------------------------------------------
    def _control_workers(self, shared_data, train_pid, eval_pid, train_alive, eval_alive,
                         batch_size, time_slice, current_time, should_log):
        """Step 7: pause/resume train and eval according to operation_focus and the internal mode."""
        if self.operation_focus == "continuous_eval":
            self._signal(eval_pid, eval_alive, signal.SIGCONT)
            self._signal(train_pid, train_alive, signal.SIGCONT)
            if should_log:
                log_info(f"[AdaptOCL_ContEval] Continuous evaluation. Train parallel. B={batch_size}, T_slice={time_slice:.2f}")
        elif self.operation_focus == "balanced":
            if self.current_internal_mode == "latency_aware":
                self._control_latency_aware(shared_data, train_pid, eval_pid, train_alive, eval_alive, current_time, should_log)
            elif self.current_internal_mode == "adaptive_accuracy":
                self._control_adaptive_accuracy(shared_data, train_pid, eval_pid, train_alive, eval_alive, current_time, should_log)
        else:  # Should not happen with proper config validation
            log_warning(f"[AdaptOCL] Unknown operation_focus: {self.operation_focus}. Defaulting to parallel execution.")
            self._run_both(train_pid, eval_pid, train_alive, eval_alive)

    def _control_latency_aware(self, shared_data, train_pid, eval_pid, train_alive, eval_alive, current_time, should_log):
        """
        LA mode: train-only until `la_priority_percent` of the experiences are done
        (with a short forced eval every `forced_eval_interval`), then parallel.
        """
        if self.la_priority_phase_active:
            current_exp = shared_data.get("current_experience", 0)
            total_exps = shared_data.get("total_experiences", 1)
            progress_percent = (current_exp / total_exps) if total_exps > 0 else 0

            if progress_percent < self.la_priority_percent and not shared_data.get("all_experiences_completed", False):
                self._train_only(train_pid, eval_pid, train_alive, eval_alive)
                if should_log:
                    log_info(f"[AdaptOCL] LA Priority: Training (Exp {current_exp}/{total_exps}, Prog {progress_percent:.2f} < {self.la_priority_percent:.2f}). Eval stopped.")
                if eval_alive and current_time - self.last_forced_eval_time >= self.forced_eval_interval:
                    log_info(f"[AdaptOCL] LA Priority: Forced eval check.")
                    safe_kill(eval_pid, signal.SIGCONT)
                    time.sleep(self.eval_transition_time)  # Let eval run briefly
                    self._train_only(train_pid, eval_pid, train_alive, eval_alive)
                    self.last_forced_eval_time = time.time()  # Update timestamp AFTER eval
            else:
                self.la_priority_phase_active = False
                log_info(f"[AdaptOCL] LA: Priority phase ended (Prog {progress_percent:.2f} or all exp completed). Switching to parallel.")

        if not self.la_priority_phase_active:
            self._run_both(train_pid, eval_pid, train_alive, eval_alive)
            if should_log:
                log_info(f"[AdaptOCL] LA Parallel: Training and Evaluation running.")

    def _control_adaptive_accuracy(self, shared_data, train_pid, eval_pid, train_alive, eval_alive, current_time, should_log):
        """
        AA mode: train-only, with a forced accuracy check every `forced_eval_interval`;
        switch to parallel once accuracy reaches `aa_accuracy_threshold` or after
        `max_accuracy_checks` checks.
        """
        if self.aa_priority_phase_active:
            perform_eval_check = False
            if current_time - self.last_forced_eval_time >= self.forced_eval_interval:
                perform_eval_check = True
                self.accuracy_check_count += 1
                log_info(f"[AdaptOCL] AA Priority: Forced eval check #{self.accuracy_check_count}.")

            if perform_eval_check and eval_alive:
                self._forced_accuracy_check(shared_data, train_pid, eval_pid, train_alive, eval_alive)
            else:
                self._train_only(train_pid, eval_pid, train_alive, eval_alive)
                if should_log:
                    log_info(f"[AdaptOCL] AA Priority: Training. Eval stopped. Next check in {self.forced_eval_interval - (current_time - self.last_forced_eval_time):.1f}s")

        if not self.aa_priority_phase_active:
            self._run_both(train_pid, eval_pid, train_alive, eval_alive)
            if should_log:
                log_info(f"[AdaptOCL] AA Parallel: Training and Evaluation running.")

    def _forced_accuracy_check(self, shared_data, train_pid, eval_pid, train_alive, eval_alive):
        """Pause training, let eval run for `eval_transition_time`, and compare accuracy to the AA threshold."""
        self._signal(train_pid, train_alive, signal.SIGSTOP)  # Pause train during eval
        safe_kill(eval_pid, signal.SIGCONT)
        time.sleep(self.eval_transition_time)
        self.last_forced_eval_time = time.time()  # Update timestamp AFTER eval run

        latest_acc = shared_data.get("latest_accuracy", 0.0)
        if latest_acc is None:
            latest_acc = 0.0  # integrity check
        log_info(f"[AdaptOCL] AA Priority: Accuracy check result: {latest_acc:.4f} (threshold: {self.aa_accuracy_threshold:.4f})")

        if latest_acc >= self.aa_accuracy_threshold:
            self.aa_priority_phase_active = False
            log_info(f"[AdaptOCL] AA: Accuracy threshold reached! ({latest_acc:.4f} >= {self.aa_accuracy_threshold:.4f}). Switching to parallel.")
        elif self.accuracy_check_count >= self.max_accuracy_checks:
            self.aa_priority_phase_active = False
            log_info(f"[AdaptOCL] AA: Max accuracy checks ({self.max_accuracy_checks}) reached. Forcing parallel mode.")

        if self.aa_priority_phase_active:
            self._train_only(train_pid, eval_pid, train_alive, eval_alive)
        else:
            self._run_both(train_pid, eval_pid, train_alive, eval_alive)

    def _release_eval_worker(self, shared_data, eval_pid):
        """
        After the loop ends, make sure the eval worker is not left SIGSTOPped by
        this scheduler so it can finish its last evaluation and exit. Under global
        termination, main.py handles cleanup instead.
        """
        if shared_data.get("TERMINATE_SIGNAL", False):
            log_info("[AdaptOCL] Global termination signal active, main process will handle eval_worker cleanup.")
            return
        if not is_process_alive(eval_pid):
            log_info("[AdaptOCL] Eval_worker was not alive at scheduler exit or termination already in progress.")
            return
        log_info(f"[AdaptOCL] Training has likely ended. Attempting to ensure eval_worker (PID: {eval_pid}) is woken and can terminate.")
        safe_kill(eval_pid, signal.SIGCONT)
        time.sleep(0.1)  # Give a moment for SIGCONT to be processed
        log_info(f"[AdaptOCL] Sent SIGCONT to eval_worker (PID: {eval_pid}) to ensure it is not stopped.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, train_pid, eval_pid, shared_data):
        """
        AdaptOCL scheduler worker (Algorithm 1). Each tick (one time slice):
          1. re-fetch batch size / time slice,
          2. commit staged config at experience boundaries,
          3-4. compute UAM and its hysteresis-filtered change,
          5. stage the next batch size / time slice,
          6. pick LA or AA internal mode,
          7. pause/resume the train and eval workers accordingly.
        The loop exits when training finishes or both workers are gone.
        """
        self.initialize_shared_data(shared_data)

        time_slice = self.time_slice
        batch_size = max(shared_data.get("train_batch_size", self.MIN_BATCH_SIZE), self.MIN_BATCH_SIZE)
        B_max = shared_data.get("max_batch_size", 256)
        self.total_experiences = shared_data.get("total_experiences", 1)

        log_info(f"[AdaptOCL] Start: ω={self.omega}, γ={self.gamma}, η={self.eta}, ε={self.uam_eps}")
        log_info(f"[AdaptOCL] Initial config: batch_size={batch_size}, time_slice={time_slice}")
        log_info(f"[AdaptOCL] Using dynamic latency budget: {self.latency_budget}")

        last_log_time = 0
        while shared_data.get("train_process_active", True):
            current_time = time.time()
            should_log = current_time - last_log_time >= self.LOGGING_INTERVAL

            batch_size, time_slice = self._refresh_config(shared_data, batch_size, time_slice)
            batch_size, time_slice = self._apply_pending_at_experience_boundary(shared_data, batch_size, time_slice)
            acc, latency, normalized_latency, uam, delta_uam, effective_delta_uam = self._compute_uam(shared_data)

            if should_log:
                pending_cfg_log = shared_data.get("PENDING_CFG", {"batch": "N/A", "tslice": "N/A"})
                last_applied_batch_log = shared_data.get("last_applied_batch", "N/A")
                log_info(f"[AdaptOCL] Metrics: acc={acc:.3f}, lat={latency:.3f}, norm_lat={normalized_latency:.3f}, B={batch_size}, T_slice={time_slice:.2f}")
                log_info(f"[AdaptOCL] UAM={uam:.3f}, ΔUAM={delta_uam:.3f} (eff_ΔUAM={effective_delta_uam:.3f}), exp={self.current_experience}/{self.total_experiences}, internal_mode={self.current_internal_mode}, pend_cfg={pending_cfg_log}, ack_batch={last_applied_batch_log}")

            if effective_delta_uam != 0:
                self._stage_pending_config(shared_data, batch_size, time_slice, effective_delta_uam, B_max)
            self._update_internal_mode(effective_delta_uam, current_time)

            train_alive = is_process_alive(train_pid)
            eval_alive = is_process_alive(eval_pid)
            if not train_alive and not eval_alive:
                log_info("[AdaptOCL] Both train and eval processes are dead. Exiting scheduler.")
                break
            if not train_alive:  # Let eval finish if it is still running
                self._signal(eval_pid, eval_alive, signal.SIGCONT)
                log_info("[AdaptOCL] Train process is dead. Waiting for eval or exiting.")
                time.sleep(self.MIN_TIME_SLICE)
                continue

            self._control_workers(shared_data, train_pid, eval_pid, train_alive, eval_alive,
                                  batch_size, time_slice, current_time, should_log)

            self.last_acc = acc
            self.last_uam = uam
            if should_log:
                last_log_time = current_time
            time.sleep(max(time_slice, self.MIN_TIME_SLICE))

        log_info("[AdaptOCL] Main scheduler loop finished.")
        self._release_eval_worker(shared_data, eval_pid)
        log_info("[AdaptOCL] Scheduler worker stopping.")

def adaptocl_scheduler_worker(global_scheduler, train_pid, eval_pid, shared_data, lock): # Added lock
    """
    Worker entrypoint for AdaptOCLScheduler, matching the interface of other schedulers.
    """
    # Pass the lock to the scheduler instance
    if isinstance(global_scheduler, AdaptOCLScheduler):
        global_scheduler.lock = lock
    return global_scheduler.run(train_pid, eval_pid, shared_data) 