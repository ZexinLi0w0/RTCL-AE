"""
Global variables used across the application.
This file contains all global variables used throughout the application.
"""

import queue
import threading

# hardware
hardware = "Nano"  # "Nano" or "Orin"

# Process control variables
TERMINATE_SIGNAL = False
CONFIG_UPDATE_REQUESTED = False

# Timestamp event queue
timestamp_queue = queue.Queue()
timestamp_lock = threading.Lock()

# Data loader related variables
default_training_batch_size = 16
default_eval_batch_size = 16
default_timeslice = 1.0

# Process state related variables (default values)
train_process_active = True
all_experiences_completed = False

# Model scheduler related variables
using_shared_memory = False
current_buffer = "A"  # "A" or "B" (for double buffering)

# Logging related constants
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING" 
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_DEBUG = "DEBUG"

def reset_globals():
    """
    Resets all global variables to their default values.
    Useful when starting a new process in a multiprocessing environment.
    """
    global TERMINATE_SIGNAL, CONFIG_UPDATE_REQUESTED
    global train_process_active, all_experiences_completed
    
    TERMINATE_SIGNAL = False
    CONFIG_UPDATE_REQUESTED = False
    train_process_active = True
    all_experiences_completed = False
    
    # Clear the queue
    while not timestamp_queue.empty():
        try:
            timestamp_queue.get_nowait()
        except:
            pass