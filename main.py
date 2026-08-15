#!/usr/bin/env python3
"""
Main entry point for the Avalanche training and evaluation system.
This script orchestrates the training and evaluation processes
using multiprocessing.
"""

import argparse
import signal
import time
import os
import torch
import torch.multiprocessing as mp

# IMPORT ORDER FIX: avalanche.benchmarks.classic MUST be imported before any src.workers.*
# because workers indirectly trigger avalanche.training, which depends on avalanche.benchmarks.classic
# being already loaded (circular import in modified avalanche fork).
from avalanche.benchmarks.classic import EndlessCLSim, PermutedMNIST, SplitCIFAR10, SplitCIFAR100, CORe50, SoftRobot

from src.config import parse_arguments
from src.globals import TERMINATE_SIGNAL, reset_globals, train_process_active, all_experiences_completed, hardware
from src.schedulers.model_scheduler import ModelScheduler
from src.schedulers.timeline_scheduler import GlobalTimelineScheduler, global_scheduler_worker
from src.schedulers.adaptocl_scheduler import AdaptOCLScheduler, adaptocl_scheduler_worker
from src.schedulers.recl_scheduler import RECLScheduler, recl_scheduler_worker
from src.workers.train_worker import train_worker
from src.workers.eval_worker import eval_worker
from src.workers.config_worker import dynamic_config_worker
from src.workers.monitor_worker import memory_monitor_worker
from src.utils.signal_handlers import signal_handler
from src.utils.analytics import analyze_batch_size_changes
from src.utils.logging_utils import setup_logger, timestamp_logger_worker

# Set up logger
logger = setup_logger("main")

def transform_target(x):
    """Transform target for semseg task"""
    return torch.from_numpy(x).long()

def create_benchmark(args):
    """Create benchmark based on arguments"""
    if args.benchmark == "endless":
        target_transform = None
        if args.semseg:
            target_transform = transform_target
        return EndlessCLSim(
            scenario=args.scenario,
            sequence_order=None,
            task_order=None,
            semseg=args.semseg,
            dataset_root=args.dataset_root,
            target_transform=target_transform,
        )
    elif args.benchmark == "split_cifar10":
        # large_model_resnet50 branch: ImageNet-style backbones need 224x224 inputs
        from src.models.model_init import _IMAGENET_BACKBONES
        if args.model in _IMAGENET_BACKBONES:
            from torchvision import transforms
            train_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
            ])
            eval_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
            ])
            return SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False,
                                train_transform=train_tf, eval_transform=eval_tf)
        return SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "split_cifar100":
        return SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(100)), shuffle=False, return_task_id=False)
    elif args.benchmark == "core50":
        benchmark = CORe50(scenario=args.scenario_core50, mini=True, object_lvl=False) # due to memory limitations, we use mini
        # Force benchmark to load data immediately instead of lazy loading
        _ = benchmark.train_stream
        _ = benchmark.test_stream
        return benchmark
    elif args.benchmark == "perm_mnist":
        return PermutedMNIST(n_experiences=3)
    elif args.benchmark == "soft_robot":
        if args.scenario_soft_robot == "ic":
            return SoftRobot(
                n_experiences=12,
                scenario="ic",
                dataset_root=args.dataset_root,
            )
        elif args.scenario_soft_robot == "il":
            return SoftRobot(
                n_experiences=5,
                scenario="il",
                dataset_root=args.dataset_root,
            )
        else:
            raise ValueError("Invalid scenario for soft robot benchmark")
    else:
        raise ValueError("Invalid benchmark name")

def main():
    global TERMINATE_SIGNAL
    # Initialize global variables
    reset_globals()
    
    # Register signal handlers in main process
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Use spawn method for multiprocessing (more stable than fork)
    mp.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    args = parse_arguments()

    if args.semseg:
        import avalanche.evaluation.metrics.accuracy as _acc_mod
        _acc_mod.is_semseg_acc = True  # enable per-pixel accuracy in the custom Avalanche fork

    # hardcode setting for avoiding OOM in GSS_greedy
    # if args.algorithm == "gss_greedy" and hardware == "Nano":
    #     args.mem_size = 100
    #     print("[INFO] Hardcode mem_size to 100 for GSS_greedy on Nano")

    # Set device (CPU or GPU)
    device = torch.device(f"cuda:{args.cuda}" if args.cuda != -1 else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create benchmark
    benchmark = create_benchmark(args)
    
    # If download only mode, exit after downloading datasets
    if args.download_only:
        logger.info("Download only mode. Exiting.")
        exit(0)
    
    # Create model path with timestamp for uniqueness
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"shared_model_{args.benchmark}_{args.model}_{args.algorithm}_{timestamp}.pth"
    
    # Create shared resources
    lock = mp.Lock()
    manager = mp.Manager()
    shared_data = manager.dict()
    
    # Initialize essential shared data fields
    shared_data["train_process_active"] = train_process_active
    shared_data["all_experiences_completed"] = all_experiences_completed
    shared_data["TERMINATE_SIGNAL"] = TERMINATE_SIGNAL
    shared_data["CONFIG_UPDATE_REQUESTED"] = False
    
    # RECL_SCHED specific shared_data initialization
    if args.global_scheduler_mode == "recl_sched":
        shared_data['eval_latency'] = manager.dict() # For RECL_SCHED: eval workers report latency here
        shared_data['eval_slo_ms'] = args.slo_ms     # For RECL_SCHED: SLO in ms from args
        logger.info(f"[INFO] RECL_SCHED mode: Initialized shared_data for eval_latency and eval_slo_ms ({args.slo_ms}ms).")

    # Initialize the model scheduler
    scheduler = ModelScheduler(shared_data, use_double_buffer=args.enable_double_buffer)
    
    # Initialize configuration controller if dynamic reconfiguration is enabled
    config_controller = None
    if args.enable_dynamic_reconfiguration:
        from src.schedulers.dynamic_config import DynamicConfigController
        config_controller = DynamicConfigController(shared_data, args)
        
        # Determine initial training batch size
        initial_train_bs = args.training_bs
        if args.global_scheduler_mode == "adaptocl":
            initial_train_bs = 64
            logger.info(f"[INFO] Setting initial training batch size to {initial_train_bs} for AdaptOCL scheduler.")
        
        # Set initial configuration through the controller
        config_controller.update_config({
            "train_batch_size": initial_train_bs,
            "eval_batch_size": args.eval_bs,
            "timeslice": args.timeslice
        })
    else: # Dynamic reconfiguration is OFF
        # Set initial train_batch_size in shared_data directly, as config_controller is None.
        # This ensures AdaptOCLScheduler and other components read the intended initial value.
        initial_train_bs_no_reconfig = args.training_bs
        if args.global_scheduler_mode == "adaptocl":
            initial_train_bs_no_reconfig = 64
            logger.info(f"[INFO] Dynamic Reconfiguration is OFF. Setting initial shared_data.train_batch_size to {initial_train_bs_no_reconfig} for AdaptOCL scheduler.")
        else:
            logger.info(f"[INFO] Dynamic Reconfiguration is OFF. Setting initial shared_data.train_batch_size to {initial_train_bs_no_reconfig} from args.")
        shared_data["train_batch_size"] = initial_train_bs_no_reconfig

        # Set other relevant initial configurations in shared_data if not set by a controller.
        # AdaptOCLScheduler reads 'timeslice' from shared_data.
        if "timeslice" not in shared_data:
            shared_data["timeslice"] = args.timeslice
            logger.info(f"[INFO] Dynamic Reconfiguration is OFF. Setting initial shared_data.timeslice to {args.timeslice} from args.")
        if "eval_batch_size" not in shared_data: # For completeness, though not directly used by AdaptOCL scheduler loop
            shared_data["eval_batch_size"] = args.eval_bs
            logger.info(f"[INFO] Dynamic Reconfiguration is OFF. Setting initial shared_data.eval_batch_size to {args.eval_bs} from args.")
    
    # Configure adaptive scheduling parameters
    adaptive_params = {
        "priority_percent": args.adaptive_priority_percent,
        "accuracy_threshold": args.adaptive_accuracy_threshold
    }

    # Prepare AdaptOCL parameters
    adaptocl_params = {
        "omega": args.uam_omega,
        "gamma": args.uam_gamma,
        "eta": args.uam_eta,
        "alpha": args.uam_alpha,
        "delta_acc": args.uam_delta_acc
    }

    # Create the global timeline scheduler
    if args.global_scheduler_mode == "adaptocl":
        global_scheduler = AdaptOCLScheduler(
            time_slice=args.timeslice,
            mode="adaptocl",
            adaptocl_params=adaptocl_params,
            lock=lock
        )
    elif args.global_scheduler_mode == "recl_sched":
        # RECLScheduler instance is created inside recl_scheduler_worker
        global_scheduler = None # Placeholder, as the worker creates the instance
        logger.info("[INFO] RECL_SCHED mode selected. Scheduler instance will be created by the worker.")
    else:
        global_scheduler = GlobalTimelineScheduler(
            time_slice=args.timeslice, 
            mode=args.global_scheduler_mode,
            adaptive_params=adaptive_params
        )
    
    # Start logging thread if dynamic reconfiguration is enabled
    if args.enable_dynamic_reconfiguration:
        import threading
        logger_thread = threading.Thread(target=timestamp_logger_worker)
        logger_thread.daemon = True
        logger_thread.start()
    
    # List to keep track of all processes
    processes = []
    
    try:
        # Start memory monitor if enabled
        mem_monitor_proc = None
        if args.enable_memory_monitor:
            mem_monitor_proc = mp.Process(
                target=memory_monitor_worker,
                args=(shared_data,),
                name="MemoryMonitor"
            )
            mem_monitor_proc.start()
            processes.append(mem_monitor_proc)
            logger.info("Memory monitor process started.")
        
        # Create and start training process
        train_proc = mp.Process(
            target=train_worker, 
            args=(args, device, scheduler, lock, model_path, shared_data, config_controller),
            name="TrainWorker"
        )
        
        # Create and start evaluation process
        eval_proc = mp.Process(
            target=eval_worker, 
            args=(args, device, scheduler, lock, model_path, shared_data, config_controller),
            name="EvalWorker"
        )
        
        train_proc.start()
        eval_proc.start()
        processes.extend([train_proc, eval_proc])
        
        # Wait a short time to ensure child processes have started
        time.sleep(5)
        train_pid = train_proc.pid
        eval_pid = eval_proc.pid
        
        # Start dynamic configuration worker if enabled
        dynamic_config_proc = None
        if args.enable_dynamic_reconfiguration:
            dynamic_config_proc = mp.Process(
                target=dynamic_config_worker, 
                args=(config_controller, train_pid, eval_pid, args.reconfiguration_interval),
                name="DynamicConfigWorker"
            )
            dynamic_config_proc.start()
            processes.append(dynamic_config_proc)
            logger.info("Dynamic configuration worker started.")
        
        # Start global scheduler process
        if args.global_scheduler_mode == "adaptocl":
            gs_proc = mp.Process(
                target=adaptocl_scheduler_worker,
                args=(global_scheduler, train_pid, eval_pid, shared_data, lock),
                name="AdaptOCLScheduler"
            )
        elif args.global_scheduler_mode == "recl_sched":
            gs_proc = mp.Process(
                target=recl_scheduler_worker,
                # Pass args_obj for RECLScheduler initialization within the worker
                args=(global_scheduler, train_pid, eval_pid, shared_data, lock, args), 
                name="RECLScheduler"
            )
        else:
            gs_proc = mp.Process(
                target=global_scheduler_worker, 
                args=(global_scheduler, train_pid, eval_pid, shared_data),
                name="GlobalScheduler"
            )
        gs_proc.start()
        processes.append(gs_proc)
        
        # Monitor processes with a timeout
        max_runtime = getattr(args, 'max_runtime', 36000)
        start_time = time.time()
        
        # Main monitoring loop
        while any(p.is_alive() for p in processes):
            # Check for timeout or termination signal
            if TERMINATE_SIGNAL or (time.time() - start_time > max_runtime):
                if not TERMINATE_SIGNAL:
                    logger.info(f"Maximum runtime of {max_runtime}s reached. Initiating graceful shutdown...")
                    TERMINATE_SIGNAL = True
                
                # Update shared data to notify all processes
                shared_data["TERMINATE_SIGNAL"] = True
                
                # Gracefully terminate all processes
                for p in processes:
                    if p.is_alive():
                        logger.info(f"Sending termination signal to process {p.name} (PID: {p.pid})")
                        os.kill(p.pid, signal.SIGTERM)
                
                # Give processes time to clean up
                timeout = time.time() + 10  # 10 seconds timeout
                while any(p.is_alive() for p in processes) and time.time() < timeout:
                    time.sleep(0.5)
                
                # Force terminate if needed
                for p in processes:
                    if p.is_alive():
                        logger.warning(f"Force terminating process {p.name}")
                        p.terminate()
                        p.join(1)
                
                break
            
            time.sleep(1)  # Check every second
        
        # Join processes with a timeout to prevent hanging
        for p in processes:
            p.join(timeout=5)
        
        # Run batch size change analysis if requested
        if args.enable_dynamic_reconfiguration:
            logger.info("Batch size change analysis running...")
            analyze_batch_size_changes()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Initiating graceful shutdown...")
        TERMINATE_SIGNAL = True
        
        # Gracefully terminate all processes
        for p in processes:
            if p and p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
                p.join(timeout=5)
                
        logger.info("Cleanup complete after keyboard interrupt.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # Terminate any remaining processes
        for p in processes:
            if p and p.is_alive():
                p.terminate()
                p.join(timeout=1)
    
    finally:
        # Ensure all processes are properly terminated
        for p in processes:
            if p and p.is_alive():
                logger.warning(f"Force terminating process {p.name}")
                p.terminate()
                try:
                    p.join(timeout=1)
                except:
                    pass
    
    logger.info("All processes have completed. Program exiting.")

if __name__ == "__main__":
    main()