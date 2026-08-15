"""
Memory monitoring worker implementation.
"""

import time
import psutil
import os
import platform
from src.globals import TERMINATE_SIGNAL
from src.utils.logging_utils import log_event

# Constants for timing and monitoring configuration
MONITORING_INTERVAL = 5.0  # Seconds between monitoring checks
DEFAULT_MAX_DURATION = 600  # Default monitoring duration in seconds (10 minutes)
MEMORY_SIZE_MB = 1024 * 1024  # 1 MB in bytes
MEMORY_SIZE_GB = 1024 * 1024 * 1024  # 1 GB in bytes

def get_cpu_memory_usage():
    """
    Returns the resident memory (RSS) used by the current process in bytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_system_memory_usage():
    """
    Returns system-wide memory usage.
    """
    virtual_memory = psutil.virtual_memory()
    return {
        "total": virtual_memory.total,
        "available": virtual_memory.available,
        "used": virtual_memory.used,
        "percent": virtual_memory.percent
    }

def memory_monitor_worker(shared_data=None, max_duration=DEFAULT_MAX_DURATION, interval=MONITORING_INTERVAL):
    """
    Monitors CPU memory, GPU usage, and RAM usage on a Jetson device.
    If jtop is not installed, falls back to psutil monitoring.
    
    Args:
        shared_data: Shared data dictionary for communication between processes
        max_duration: Maximum monitoring duration in seconds (default: 10 minutes)
        interval: Monitoring interval in seconds (default: 5 seconds)
    """
    # log the start of the memory monitoring
    log_event("monitor", "Memory monitoring started")
    print("[MemoryMonitor] Starting memory monitoring...")
    
    # set the cycle count and max cycles
    cycle_count = 0
    max_cycles = int(max_duration / interval)  # Convert duration to cycles
    use_jtop = False
    jetson = None
    
    try:
        # check if the device is a Jetson device (based on platform information)
        is_jetson = "aarch64" in platform.machine() and "tegra" in platform.platform().lower()
        
        if is_jetson:
            try:
                from jtop import jtop
                jetson = jtop()
                use_jtop = jetson.open()
                if use_jtop:
                    print("[MemoryMonitor] Using jtop for Jetson device monitoring")
                else:
                    print("[MemoryMonitor] Could not open jtop session. Falling back to psutil")
            except ImportError:
                print("[MemoryMonitor] `jtop` module not found. Install it with: pip install jetson-stats")
                print("[MemoryMonitor] Falling back to psutil for basic monitoring")
        else:
            print("[MemoryMonitor] Not a Jetson device. Using psutil for basic monitoring")
    
        # main monitoring loop
        while cycle_count < max_cycles:
            # check for termination signal (both global variable and shared_data)
            if TERMINATE_SIGNAL or (shared_data and shared_data.get("TERMINATE_SIGNAL", False)):
                print("[MemoryMonitor] Termination signal received. Exiting monitor.")
                break
            
            # increment the cycle count
            cycle_count += 1
            
            # check the basic CPU memory usage (in all cases)
            cpu_mem = get_cpu_memory_usage()
            system_mem = get_system_memory_usage()
            
            # print the basic information
            print(f"[MemoryMonitor] CPU Memory usage: {cpu_mem / MEMORY_SIZE_MB:.2f} MB")
            print(f"[MemoryMonitor] System Memory: {system_mem['used'] / MEMORY_SIZE_GB:.2f} GB / {system_mem['total'] / MEMORY_SIZE_GB:.2f} GB ({system_mem['percent']}%)")
            
            # collect additional information through jtop (if the device is a Jetson and jtop is available)
            if use_jtop and jetson:
                try:
                    # GPU and RAM usage
                    stats = jetson.stats
                    
                    # GPU usage (percentage)
                    gpu_usage = stats.get("GPU", None)
                    
                    # RAM and SWAP usage
                    ram_fraction = stats.get("RAM", None)
                    swap_fraction = stats.get("SWAP", None)
                    
                    # print the information
                    if gpu_usage is not None:
                        print(f"[MemoryMonitor] GPU usage: {gpu_usage}%")
                    if ram_fraction is not None:
                        print(f"[MemoryMonitor] RAM usage: {ram_fraction * 100:.2f}%")
                    if swap_fraction is not None:
                        print(f"[MemoryMonitor] Swap usage: {swap_fraction * 100:.2f}%")
                except Exception as e:
                    print(f"[MemoryMonitor] Error reading from jtop: {e}")
            
            print("-" * 40)
            time.sleep(interval)
        
        print(f"[MemoryMonitor] Monitoring completed after {cycle_count * interval} seconds")
        
    except KeyboardInterrupt:
        print("[MemoryMonitor] Monitoring stopped by user")
    except Exception as e:
        print(f"[MemoryMonitor] Error in monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the jtop session if it is open
        if use_jtop and jetson:
            try:
                jetson.close()
                print("[MemoryMonitor] Closed jtop session")
            except:
                pass
        log_event("monitor", "Memory monitoring completed")
        print("[MemoryMonitor] Memory monitoring process completed")