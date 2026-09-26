"""
Signal handlers for process communication and termination.
"""

import signal
import os
import psutil
from src.globals import TERMINATE_SIGNAL

def signal_handler(signum, frame):
    """
    Handler for termination signals (SIGINT, SIGTERM).
    Sets the global termination flag to True.
    """
    global TERMINATE_SIGNAL
    if signum in [signal.SIGINT, signal.SIGTERM]:
        print("\n[Main] Termination signal received. Gracefully shutting down...")
        TERMINATE_SIGNAL = True

def request_config_update_handler(signum, frame, shared_data=None):
    """
    Handler for configuration update signals (SIGUSR1).
    Sets the configuration update flag in shared_data.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        shared_data: Shared data dictionary between processes
    """
    if shared_data is not None:
        shared_data["CONFIG_UPDATE_REQUESTED"] = True
        print("[Signal] Configuration update requested")
    else:
        from src.globals import CONFIG_UPDATE_REQUESTED
        CONFIG_UPDATE_REQUESTED = True
        print("[Signal] Configuration update requested")

def is_process_alive(pid):
    """
    Return True if `pid` refers to a running (non-zombie) process.
    Never logs: a worker that has already exited is an expected state at shutdown.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Exists but owned by someone else.
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def safe_kill(pid, sig):
    """
    Given a pid, send a signal and handle any exceptions.
    Returns True if the signal was sent successfully, False otherwise.
    `sig == 0` is a liveness probe and is equivalent to `is_process_alive(pid)`.
    """
    if sig == 0:
        return is_process_alive(pid)
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        # The target already exited (normal once train/eval finish); nothing to signal.
        return False
    except PermissionError:
        print(f"[SafeKill] PID {pid} cannot send signal.")
        return False
    except Exception as e:
        print(f"[SafeKill] Signal sending error: {e}")
        return False
