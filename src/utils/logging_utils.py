"""
Logging utilities for the application.
"""

import logging
import sys
import os
import time
import queue
import csv
import datetime
from src.globals import timestamp_queue, timestamp_lock, LOG_LEVEL_INFO

# Default logger configuration
log_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set default log level
logger.addHandler(console_handler)

def log_warning(message):
    """Log a warning message."""
    logger.warning(message)

def log_error(message):
    """Log an error message."""
    logger.error(message)

def setup_logger(name, log_level=logging.INFO):
    """Set up a logger with the specified name and level."""
    new_logger = logging.getLogger(name)
    new_logger.setLevel(log_level)
    # Avoid adding duplicate handlers
    if not new_logger.hasHandlers():
        new_logger.addHandler(console_handler)
    # Prevent propagation to parent logger (configure as needed)
    # new_logger.propagate = False
    return new_logger

def get_timestamp():
    """
    Get current timestamp in a consistent format
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def debug_print(message, args):
    """Print debug messages if debug flag is set."""
    if hasattr(args, 'debug') and args.debug:
        logger.debug(f"[Debug] {message}")  # Use logger.debug

def log_info(message):
    """Log an info message using the root logger."""
    logger.info(message)

def log_event(event_type, message, batch_idx=None, batch_size=None):
    """
    Log an event with timestamp information
    """
    timestamp = get_timestamp()
    event = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "time_seconds": time.time()
    }
    
    with timestamp_lock:
        timestamp_queue.put(event)
    
    log_info(f"[{timestamp}][{event_type}] {message}")

def timestamp_logger_worker(filename="timeslice_events.csv"):
    """Worker to write timestamped events to a CSV file."""
    csv_file = filename
    file_exists = os.path.exists(csv_file)
    try:
        with open(csv_file, 'a', newline='') as f:
            fieldnames = ["timestamp", "event_type", "message", "batch_idx", "batch_size", "time_seconds"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            while True:
                try:
                    event = timestamp_queue.get(timeout=1.0)
                    writer.writerow(event)
                    f.flush()
                    timestamp_queue.task_done()
                except queue.Empty:
                    # Check for termination condition if needed
                    continue
                except KeyboardInterrupt:
                    log_info("Timestamp logger worker interrupted.")
                    break
                except Exception as e:
                    log_error(f"Error in timestamp logger worker: {e}")
                    time.sleep(1)  # Avoid busy-looping on error
    except Exception as e:
        log_error(f"Could not open or write to timestamp log file {csv_file}: {e}")

def set_debug_level(args):
    """Set the appropriate logging level based on the debug flag."""
    if hasattr(args, 'debug') and args.debug:
        logger.setLevel(logging.DEBUG)
        log_info("Debug level logging enabled.")
    else:
        logger.setLevel(logging.INFO)

def log_accuracy(accuracy, message=None, experience_id=None):
    """
    Log accuracy information.
    
    Args:
        accuracy: The accuracy value to log (float)
        message: Optional message to include with the log
        experience_id: Optional experience ID for the accuracy
    """
    if experience_id is not None:
        if message:
            log_info(f"[Accuracy] Experience {experience_id}: {accuracy:.4f} - {message}")
        else:
            log_info(f"[Accuracy] Experience {experience_id}: {accuracy:.4f}")
    else:
        if message:
            log_info(f"[Accuracy] {accuracy:.4f} - {message}")
        else:
            log_info(f"[Accuracy] {accuracy:.4f}")
