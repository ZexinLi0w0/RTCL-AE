"""
Global timeline scheduler for managing train/eval processes.
"""

import signal
import time
import psutil
from src.utils.signal_handlers import safe_kill, is_process_alive
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

class _Workers:
    """
    PIDs and liveness of the train/eval workers as seen by the scheduler, plus
    the config-update relay that every mode performs each tick.
    """
    def __init__(self, train_pid, eval_pid, shared_data):
        self.train_pid = train_pid
        self.eval_pid = eval_pid
        self.shared_data = shared_data
        self.train_alive = True
        self.eval_alive = True

    def check_alive(self):
        """Refresh liveness flags; returns True while at least one worker is running."""
        if self.train_alive and not is_process_alive(self.train_pid):
            self.train_alive = False
            log_info("[Scheduler] Training process has terminated")
        if self.eval_alive and not is_process_alive(self.eval_pid):
            self.eval_alive = False
            log_info("[Scheduler] Evaluation process has terminated")
        return self.train_alive or self.eval_alive

    def relay_config_update(self):
        """Forward a pending CONFIG_UPDATE_REQUESTED to the workers via SIGUSR1."""
        if self.shared_data.get("CONFIG_UPDATE_REQUESTED", False):
            log_info("[Scheduler] Configuration update requested, signaling processes")
            self.signal_train(signal.SIGUSR1)
            self.signal_eval(signal.SIGUSR1)
            self.shared_data["CONFIG_UPDATE_REQUESTED"] = False

    def signal_train(self, sig):
        if self.train_alive:
            safe_kill(self.train_pid, sig)

    def signal_eval(self, sig):
        if self.eval_alive:
            safe_kill(self.eval_pid, sig)

def _run_ekya(global_scheduler, w, shared_data, time_slice):
    """Ekya: micro-profiler + thief scheduler, applied through CPU affinity and time sharing."""
    log_info("[Scheduler] Running in Ekya mode with micro-profiler and thief scheduler")

    ekya = global_scheduler.ekya_scheduler
    ekya.register_task("train", total_iterations=shared_data.get("total_iterations", 1000))
    ekya.register_task("eval", total_iterations=shared_data.get("eval_iterations", 100))
    ekya.profiler.start_profiling("train", 0.5)
    ekya.profiler.start_profiling("eval", 0.5)

    total_cores = psutil.cpu_count(logical=False) or 4
    train_cores = list(range(total_cores // 2))
    eval_cores = list(range(total_cores // 2, total_cores))
    set_cpu_affinity(w.train_pid, train_cores)
    set_cpu_affinity(w.eval_pid, eval_cores)

    while w.check_alive():
        w.relay_config_update()
        # Re-balance CPU allocation every 10 seconds
        if int(time.time()) % 10 == 0:
            if shared_data.get("train_priority", True):
                set_cpu_affinity(w.train_pid, list(range(total_cores)))
                set_cpu_affinity(w.eval_pid, [])  # Pause eval
            else:
                set_cpu_affinity(w.train_pid, train_cores)
                set_cpu_affinity(w.eval_pid, eval_cores)

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

        # Apply resource allocations through process time sharing
        allocations = ekya.update_allocations()
        train_allocation = allocations.get("train", 0)
        eval_allocation = allocations.get("eval", 0)
        if w.train_alive and train_allocation > 0:
            safe_kill(w.train_pid, signal.SIGCONT)
            time.sleep(time_slice * train_allocation)
            safe_kill(w.train_pid, signal.SIGSTOP)
        if w.eval_alive and eval_allocation > 0:
            safe_kill(w.eval_pid, signal.SIGCONT)
            time.sleep(time_slice * eval_allocation)
            safe_kill(w.eval_pid, signal.SIGSTOP)

        # Check if any task needs to steal resources
        if ekya.should_steal_resources("train"):
            log_info("[Scheduler] Train task stealing resources")
            safe_kill(w.eval_pid, signal.SIGSTOP)
            safe_kill(w.train_pid, signal.SIGCONT)
            time.sleep(time_slice)
        elif ekya.should_steal_resources("eval"):
            log_info("[Scheduler] Eval task stealing resources")
            safe_kill(w.train_pid, signal.SIGSTOP)
            safe_kill(w.eval_pid, signal.SIGCONT)
            time.sleep(time_slice)

        time.sleep(SCHEDULER_POLL_INTERVAL)

def _run_fully_parallel(w, time_slice):
    """AOCL_basic: train and eval both run continuously."""
    log_info("[Scheduler] Running both train & eval continuously")
    w.signal_train(signal.SIGCONT)
    w.signal_eval(signal.SIGCONT)
    while w.check_alive():
        w.relay_config_update()
        time.sleep(time_slice)

def _run_continuous_eval(w, time_slice):
    """Eval runs continuously; training is switched on/off every time slice."""
    log_info("[Scheduler] Running eval continuously, training in time slices")
    w.signal_train(signal.SIGSTOP)
    w.signal_eval(signal.SIGCONT)
    while w.check_alive():
        w.relay_config_update()
        if w.train_alive:
            log_info("[Scheduler] Resuming training")
            safe_kill(w.train_pid, signal.SIGCONT)
            time.sleep(time_slice)
        if w.train_alive and w.check_alive():
            log_info("[Scheduler] Pausing training")
            safe_kill(w.train_pid, signal.SIGSTOP)
            time.sleep(time_slice)

def _adaptive_time_progress_reached(global_scheduler, shared_data):
    """True once experience progress passes priority_percent (or training is complete)."""
    current_exp = shared_data.get("current_experience", 0)
    total_exps = shared_data.get("total_experiences", 10)
    if total_exps > 1:
        # Experience 0 = 0% progress, last experience = 100% progress
        progress_percent = current_exp / (total_exps - 1)
        priority_percent = global_scheduler.adaptive_params["priority_percent"]
        log_info(f"[GlobalScheduler] Experience progress: {progress_percent:.2f} (current: {current_exp}, total: {total_exps})")
        if progress_percent >= priority_percent or shared_data.get("all_experiences_completed", False):
            log_info(f"[GlobalScheduler] Switching to alternating mode. Progress: {progress_percent:.2f}, Threshold: {priority_percent:.2f}")
            return True
        return False
    # Only one experience (or none)
    if current_exp > 0 or shared_data.get("all_experiences_completed", False):
        log_info("[GlobalScheduler] Switching to alternating mode (single experience or completed)")
        return True
    return False

def _run_adaptive_time(global_scheduler, w, shared_data, time_slice):
    """TA: train-only (with periodic forced eval) for the first priority_percent of experiences, then parallel."""
    log_info(f"[GlobalScheduler] Mode: adaptive_time => Prioritizing training for first {global_scheduler.adaptive_params['priority_percent']*100:.0f}% of experiences.")
    w.signal_train(signal.SIGCONT)
    w.signal_eval(signal.SIGSTOP)

    priority_phase = True
    last_eval_check_time = time.time()
    while w.check_alive():
        current_time = time.time()

        if shared_data:
            current_exp = shared_data.get("current_experience", 0)
            total_exps = shared_data.get("total_experiences", 10)
            if current_exp > 0 or total_exps > 0:
                log_info(f"[GlobalScheduler] Current status: Experience {current_exp}/{total_exps}")

        # Periodically force evaluation execution
        if priority_phase and current_time - last_eval_check_time > FORCED_EVAL_INTERVAL:
            log_info(f"[GlobalScheduler] Forced evaluation check during priority phase")
            w.signal_eval(signal.SIGCONT)
            time.sleep(EVAL_TRANSITION_TIME)
            last_eval_check_time = current_time
            w.signal_train(signal.SIGCONT)  # Return to train priority mode

        if priority_phase and shared_data and _adaptive_time_progress_reached(global_scheduler, shared_data):
            priority_phase = False

        if not priority_phase:
            log_info("[GlobalScheduler] Fully parallel mode.")
            w.signal_train(signal.SIGCONT)
            w.signal_eval(signal.SIGCONT)
            time.sleep(time_slice)
        else:
            time.sleep(SCHEDULER_POLL_INTERVAL)

        if not w.check_alive():
            break
    log_info("[GlobalScheduler] Both processes have completed. Scheduler exiting.")

def _run_adaptive_accuracy(global_scheduler, w, shared_data, time_slice):
    """AA: train-only with periodic accuracy checks until accuracy_threshold (or MAX_ACCURACY_CHECKS), then parallel."""
    threshold = global_scheduler.adaptive_params["accuracy_threshold"]
    log_info(f"[GlobalScheduler] Mode: adaptive_accuracy => Prioritizing training until {threshold*100:.0f}% accuracy.")
    w.signal_train(signal.SIGCONT)
    w.signal_eval(signal.SIGSTOP)

    priority_phase = True
    last_eval_check_time = time.time()
    accuracy_check_count = 0
    while w.check_alive():
        current_time = time.time()

        if shared_data and "latest_accuracy" in shared_data:
            log_info(f"[GlobalScheduler] Current accuracy: {shared_data.get('latest_accuracy', 0):.4f} (threshold: {threshold:.4f})")

        if priority_phase:
            if current_time - last_eval_check_time > FORCED_EVAL_INTERVAL:
                accuracy_check_count += 1
                log_info(f"[GlobalScheduler] Forced evaluation check #{accuracy_check_count} during priority phase")
                w.signal_eval(signal.SIGCONT)
                time.sleep(EVAL_TRANSITION_TIME)
                last_eval_check_time = current_time

                if shared_data and "latest_accuracy" in shared_data:
                    latest_accuracy = shared_data.get("latest_accuracy", 0)
                    log_info(f"[GlobalScheduler] Accuracy check result: {latest_accuracy:.4f} / {threshold:.4f}")
                    if latest_accuracy >= threshold:
                        priority_phase = False
                        log_info(f"[GlobalScheduler] Accuracy threshold reached! {latest_accuracy:.4f} >= {threshold:.4f}")
                        log_info("[GlobalScheduler] Switching to alternating mode")

                w.signal_train(signal.SIGCONT)  # Resume focus on training

                # Bound the priority phase even if the threshold is never reached
                if accuracy_check_count >= MAX_ACCURACY_CHECKS and priority_phase:
                    log_info("[GlobalScheduler] Maximum accuracy checks reached without hitting threshold.")
                    log_info("[GlobalScheduler] Forcing switch to alternating mode for safety.")
                    priority_phase = False

            if not w.train_alive:
                priority_phase = False
                log_info("[GlobalScheduler] Training process finished. Switching to alternating mode.")
            time.sleep(SCHEDULER_POLL_INTERVAL)
        else:
            log_info("[GlobalScheduler] Running in fully parallel mode.")
            w.signal_train(signal.SIGCONT)
            w.signal_eval(signal.SIGCONT)
            time.sleep(time_slice)

        if not w.check_alive():
            break
    log_info("[GlobalScheduler] Both processes have completed. Scheduler exiting.")

def _run_default_alternation(w, time_slice):
    """DA: strict round-robin, one time slice of training then one of evaluation."""
    log_info("[Scheduler] Running alternating train/eval in time slices")
    w.signal_train(signal.SIGCONT)
    w.signal_eval(signal.SIGSTOP)
    while w.check_alive():
        w.relay_config_update()
        if w.train_alive:
            log_info("[Scheduler] Training slice: resuming training, pausing evaluation")
            safe_kill(w.train_pid, signal.SIGCONT)
            w.signal_eval(signal.SIGSTOP)
            time.sleep(time_slice)
        if not w.check_alive():
            break
        if w.eval_alive:
            log_info("[Scheduler] Evaluation slice: pausing training, resuming evaluation")
            w.signal_train(signal.SIGSTOP)
            safe_kill(w.eval_pid, signal.SIGCONT)
            time.sleep(time_slice)

def global_scheduler_worker(global_scheduler, train_pid, eval_pid, shared_data):
    """
    Global scheduler process: dispatches to the policy selected by
    global_scheduler.mode and returns once both workers have exited.
    """
    mode = global_scheduler.mode
    time_slice = global_scheduler.time_slice
    w = _Workers(train_pid, eval_pid, shared_data)
    log_info(f"[Scheduler] Starting with mode: {mode}, time_slice: {time_slice}s")

    if mode == "ekya":
        _run_ekya(global_scheduler, w, shared_data, time_slice)
    elif mode == "fully_parallel":
        _run_fully_parallel(w, time_slice)
    elif mode == "continuous_eval":
        _run_continuous_eval(w, time_slice)
    elif mode == "adaptive_time":
        _run_adaptive_time(global_scheduler, w, shared_data, time_slice)
        return
    elif mode == "adaptive_accuracy":
        _run_adaptive_accuracy(global_scheduler, w, shared_data, time_slice)
        return
    else:
        _run_default_alternation(w, time_slice)
    log_info("[Scheduler] All processes completed")
