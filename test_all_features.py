#!/usr/bin/env python3
"""
Comprehensive test script for the refactored Avalanche multiprocessing code.
Tests all major features with split_cifar10 benchmark.
"""

import os
import time
import subprocess
import argparse
import glob
import csv
import matplotlib.pyplot as plt

def run_test(config_name, args_string, timeout=300):
    """
    Run a single test with the given configuration.
    
    Args:
        config_name: Name of the test configuration for logging
        args_string: Arguments to pass to main.py
        timeout: Maximum runtime in seconds before terminating
    
    Returns:
        Tuple of (success_bool, output_log, error_log, execution_time)
    """
    # Always use --enable_double_buffer for all tests
    if "--enable_double_buffer" not in args_string:
        args_string += " --enable_double_buffer"
    # Always use --algorithm replay for all tests
    if "--algorithm" not in args_string:
        args_string += " --algorithm replay"
        
    print(f"\n{'='*80}")
    print(f"RUNNING TEST: {config_name}")
    print(f"COMMAND: python3 main.py {args_string}")
    print(f"{'='*80}")
    
    start_time = time.time()
    process = subprocess.Popen(
        f"python3 main.py {args_string}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        success = process.returncode == 0
        execution_time = time.time() - start_time
        
        print(f"{'='*40} RESULTS {'='*40}")
        print(f"Status: {'SUCCESS' if success else 'FAILURE'}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Return code: {process.returncode}")
        
        if not success:
            print("\nError output:")
            print(stderr[:500] + ("..." if len(stderr) > 500 else ""))
        
        # Save output to log file
        log_dir = "test_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        with open(f"{log_dir}/{config_name}_{timestamp}.stdout.log", "w") as f:
            f.write(stdout)
        
        with open(f"{log_dir}/{config_name}_{timestamp}.stderr.log", "w") as f:
            f.write(stderr)
        
        return (success, stdout, stderr, execution_time)
        
    except subprocess.TimeoutExpired:
        process.kill()
        execution_time = time.time() - start_time
        print(f"Test TIMED OUT after {timeout} seconds")
        return (False, "", "Timeout expired", execution_time)
    except Exception as e:
        process.kill()
        execution_time = time.time() - start_time
        print(f"Error running test: {e}")
        return (False, "", str(e), execution_time)

def analyze_batch_size_changes(stdout):
    """
    Analyze the output to check if batch size changes are working.
    
    Returns:
        Tuple of (changes_detected, num_changes)
    """
    changes = [line for line in stdout.split('\n') if "batch size change" in line.lower()]
    return (len(changes) > 0, len(changes))

def analyze_checkpoint_saving(stdout):
    """
    Analyze the output to check if model checkpoints are being saved.
    
    Returns:
        Tuple of (checkpoints_detected, num_checkpoints)
    """
    saves = [line for line in stdout.split('\n') if "model state" in line.lower() and "saved" in line.lower()]
    return (len(saves) > 0, len(saves))

def check_csv_files():
    """
    Check if CSV log files were generated.
    
    Returns:
        Tuple of (csv_found, num_files)
    """
    csv_files = glob.glob("*.csv")
    return (len(csv_files) > 0, len(csv_files))

def check_timeslice_events():
    """
    Check if timeslice_events.csv contains valid events.
    
    Returns:
        Tuple of (events_detected, num_events)
    """
    if not os.path.exists("timeslice_events.csv"):
        return (False, 0)
    
    try:
        events = []
        with open("timeslice_events.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
        
        return (len(events) > 0, len(events))
    except Exception:
        return (False, 0)

def plot_batch_size_changes():
    """
    Plot batch size changes from timeslice_events.csv if available.
    """
    if not os.path.exists("timeslice_events.csv"):
        return False
    
    try:
        # Extract batch size changes
        timestamps = []
        batch_sizes = []
        
        with open("timeslice_events.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "batch size" in row.get("message", "").lower() and row.get("batch_size"):
                    try:
                        batch_size = int(row["batch_size"])
                        timestamps.append(float(row.get("time_seconds", 0)))
                        batch_sizes.append(batch_size)
                    except (ValueError, TypeError):
                        pass
        
        if not batch_sizes:
            return False
            
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, batch_sizes, 'o-', linewidth=2)
        plt.title('Batch Size Changes During Execution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Batch Size')
        plt.grid(True)
        
        # Save plot
        plot_file = "batch_size_changes.png"
        plt.savefig(plot_file)
        print(f"Batch size change plot saved to {plot_file}")
        return True
    except Exception as e:
        print(f"Error plotting batch size changes: {e}")
        return False

def test_adaptocl_mode():
    """Test AdaptOCL mode: checks dynamic batch size and scheduling."""
    import subprocess
    import sys
    args = [
        sys.executable, "main.py",
        "--benchmark", "split_cifar10",
        "--global_scheduler_mode", "adaptocl",
        "--epoch", "1",
        "--timeslice", "10",
        "--uam_omega", "0.5",
        "--uam_gamma", "0.5",
        "--uam_eta", "0.5",
        "--uam_alpha", "0.5",
        "--uam_delta_acc", "0.01",
        "--algorithm", "replay",
        "--debug"
    ]
    print("[TEST] Running AdaptOCL mode test...")
    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"AdaptOCL mode failed: {result.stderr}"
    # Check for dynamic batch size log
    assert "Dynamic batch size applied" in result.stdout or "AdaptOCL batch size applied" in result.stdout, "No dynamic batch size log found in AdaptOCL mode!"
    print("[TEST] AdaptOCL mode test passed.")

def main():
    parser = argparse.ArgumentParser(description="Test all features of the refactored Avalanche code.")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout for each test in seconds")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    args = parser.parse_args()
    
    # Define the test configurations
    tests = [
        # Basic scheduler mode tests
        {
            "name": "default_mode",
            "args": "--benchmark split_cifar10 --global_scheduler_mode default --epoch 1",
            "timeout": args.timeout
        },
        {
            "name": "fully_parallel_mode",
            "args": "--benchmark split_cifar10 --global_scheduler_mode fully_parallel --epoch 1",
            "timeout": args.timeout
        },
        {
            "name": "continuous_eval_mode",
            "args": "--benchmark split_cifar10 --global_scheduler_mode continuous_eval --epoch 1",
            "timeout": args.timeout
        },
        # Ekya scheduler mode tests
        {
            "name": "ekya_basic",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 1 --timeslice 10",
            "timeout": args.timeout
        },
        {
            "name": "ekya_aggressive_batch",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 1 --timeslice 10 --training_bs 16",
            "timeout": args.timeout
        },
        {
            "name": "ekya_large_memory",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 1 --timeslice 10 --mem_size 10000",
            "timeout": args.timeout
        },
        {
            "name": "ekya_dynamic_reconfig",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 1 --timeslice 10 --enable_dynamic_reconfiguration --reconfiguration_interval 5.0",
            "timeout": args.timeout
        },
        {
            "name": "ekya_with_memory_monitor",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 1 --timeslice 10 --enable_memory_monitor",
            "timeout": args.timeout
        },
        # Adaptive scheduler mode tests
        {
            "name": "adaptive_time_mode_30percent",
            "args": "--benchmark split_cifar10 --global_scheduler_mode adaptive_time --adaptive_priority_percent 0.3 --epoch 1",
            "timeout": args.timeout * 1.5
        },
        {
            "name": "adaptive_accuracy_mode_40percent",
            "args": "--benchmark split_cifar10 --global_scheduler_mode adaptive_accuracy --adaptive_accuracy_threshold 0.4 --epoch 1",
            "timeout": args.timeout * 1.5
        },
        # Slow/Comprehensive tests
        {
            "name": "ekya_longer_training",
            "args": "--benchmark split_cifar10 --global_scheduler_mode ekya --epoch 3 --timeslice 10",
            "timeout": args.timeout * 3,
            "slow": True
        },
        {
            "name": "fully_parallel_longer_training",
            "args": "--benchmark split_cifar10 --global_scheduler_mode fully_parallel --epoch 3",
            "timeout": args.timeout * 3,
            "slow": True
        },
    ]
    
    # Run all tests
    results = []
    
    for test in tests:
        if args.skip_slow and test.get("slow", False):
            print(f"\nSkipping slow test: {test['name']}")
            continue
            
        result = run_test(
            test["name"], 
            test["args"], 
            timeout=test.get("timeout", args.timeout)
        )
        
        results.append({
            "name": test["name"],
            "success": result[0],
            "execution_time": result[3],
            "stdout": result[1],
            "stderr": result[2]
        })
    
    # Check for batch size changes in dynamic reconfiguration tests
    for result in results:
        if "dynamic_reconfiguration" in result["name"] or "ekya_dynamic_reconfig" in result["name"]:
            if result["success"]:
                batch_changes = analyze_batch_size_changes(result["stdout"])
                result["batch_size_changes"] = batch_changes[0]
                result["num_batch_changes"] = batch_changes[1]
    
    # Check for timeslice events CSV file after dynamic reconfig tests
    timeslice_events = check_timeslice_events()
    if timeslice_events[0]:
        print(f"\nFound timeslice_events.csv with {timeslice_events[1]} events")
        # Plot batch size changes
        plot_batch_size_changes()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    print("\nTest Results:")
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['name']} ({result['execution_time']:.2f}s)")
        
        # Print batch size change results if applicable
        if "batch_size_changes" in result:
            if result["batch_size_changes"]:
                print(f"  - Detected {result['num_batch_changes']} batch size changes ✅")
            else:
                print(f"  - No batch size changes detected ❌")
    
    print("\nDetailed logs are saved in the test_logs directory.")

    # Add AdaptOCL mode test to the test suite
    test_adaptocl_mode()

if __name__ == "__main__":
    main()