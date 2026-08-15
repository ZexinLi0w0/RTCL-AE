import tinyimagenet
import numpy as np
import torch
import torch.multiprocessing as mp
import sys
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
from avalanche.benchmarks.classic import EndlessCLSim, PermutedMNIST, SplitCIFAR10, SplitCIFAR100, SplitMNIST, SplitFMNIST, SplitCUB200, CORe50
from avalanche.training import Naive, Replay, EWC, GEM, AGEM, GSS_greedy, AR1, MIR, SCR
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, GEMPlugin, EWCPlugin
import torch.optim.lr_scheduler
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    cpu_usage_metrics,
    gpu_usage_metrics,
    disk_usage_metrics,
    ram_usage_metrics,
    timing_metrics,
    MAC_metrics,
    StreamAccuracy,
    StreamForgetting,
)
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.models import pytorchcv_wrapper
import argparse
import dill
import random
import time
import os
import signal
import psutil
import queue
import csv
import datetime
import threading

# variable for dynamic batch size change
CONFIG_UPDATE_REQUESTED = False

# signal handler
def request_config_update_handler(signum, frame):
    global CONFIG_UPDATE_REQUESTED
    CONFIG_UPDATE_REQUESTED = True
    print("[Signal] Configuration update requested")
    
def get_cpu_memory_usage():
    """
    Returns the resident memory (RSS) used by the current process in bytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def memory_monitor_worker():
    """
    Monitors CPU memory, GPU usage, and RAM usage on a Jetson device.
    If jtop is not installed or jtop is unavailable, it prints a warning and exits.
    """
    try:
        from jtop import jtop
    except ImportError:
        print("[MemoryMonitor] `jtop` module not found. Install it with: python -m pip install jetson-stats")
        return

    with jtop() as jetson:
        print("[MemoryMonitor] Monitoring CPU memory, GPU usage, and RAM usage on your Jetson device.")
        print("[MemoryMonitor] Press Ctrl+C or terminate the process to exit.")
        while jetson.ok():
            # Get CPU memory usage.
            cpu_mem = get_cpu_memory_usage()

            # Retrieve system stats via jtop.
            stats = jetson.stats

            # Get GPU usage from the stats (percentage).
            gpu_usage = stats.get("GPU", None)

            # Get RAM and SWAP usage (as fractions, convert to percentages).
            ram_fraction = stats.get("RAM", None)
            swap_fraction = stats.get("SWAP", None)

            # Display the current metrics.
            print(f"[MemoryMonitor] CPU Memory usage: {cpu_mem} bytes")
            if gpu_usage is not None:
                print(f"[MemoryMonitor] GPU usage: {gpu_usage}%")
            else:
                print("[MemoryMonitor] GPU usage data not available.")
            if ram_fraction is not None:
                print(f"[MemoryMonitor] RAM usage: {ram_fraction * 100:.2f}%")
            else:
                print("[MemoryMonitor] RAM usage data not available.")
            if swap_fraction is not None:
                print(f"[MemoryMonitor] Swap usage: {swap_fraction * 100:.2f}%")

            print("-" * 40)
            time.sleep(5)
            
# ------------------
#  Reproducibility
# ------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------
#  ModelScheduler Class (Double Buffer Scheduler)
# ------------------
class ModelScheduler:
    """
    Memory-based lock-free double-buffer scheduler.
    Uses PyTorch's shared memory features to efficiently share model states between processes.
    """
    def __init__(self, shared_data, use_double_buffer=False):
        self.shared_data = shared_data
        self.use_double_buffer = use_double_buffer
        
        # Flag to track if we're using shared memory tensors
        if "using_shared_memory" not in self.shared_data:
            self.shared_data["using_shared_memory"] = use_double_buffer
            
        if use_double_buffer:
            # Double-buffer with shared memory approach
            if "latest" not in self.shared_data:
                self.shared_data["latest"] = "A"
            
            # Initialize the buffer indices as empty
            if "bufA_keys" not in self.shared_data:
                self.shared_data["bufA_keys"] = []
            if "bufB_keys" not in self.shared_data:
                self.shared_data["bufB_keys"] = []
        else:
            # For single buffer approach
            if "single_buf" not in self.shared_data:
                self.shared_data["single_buf"] = None

    def write_state_dict(self, model_state_dict):
        """
        Writes the model state to the shared memory.
        """
        if self.use_double_buffer:
            # Select the inactive buffer
            current_latest = self.shared_data["latest"]  # "A" or "B"
            next_buf = "B" if current_latest == "A" else "A"
            
            # Get all keys from the inactive buffer
            old_keys = list(self.shared_data.get(f"buf{next_buf}_keys", []))
            
            # Clean up existing tensors in the inactive buffer
            for key in old_keys:
                buffer_key = f"buf{next_buf}_{key}"
                if buffer_key in self.shared_data:
                    del self.shared_data[buffer_key]
            
            # New key list
            new_keys = []
            
            # Copy each tensor of the model state to the shared memory
            for key, tensor in model_state_dict.items():
                # If the key is too long, replace it with a hash value to save memory
                short_key = str(hash(key))
                new_keys.append(short_key)
                
                # Move the tensor to CPU and save it to the shared memory
                cpu_tensor = tensor.cpu()
                buffer_key = f"buf{next_buf}_{short_key}"
                
                # Large tensors are compressed (convert float32 to float16 to save memory)
                if cpu_tensor.element_size() * cpu_tensor.nelement() > 1024*1024:  # if larger than 1MB
                    if cpu_tensor.dtype == torch.float32:
                        cpu_tensor = cpu_tensor.half()  # compress to float16 to save memory
                
                # Save the tensor to the shared memory
                self.shared_data[buffer_key] = cpu_tensor
                
            # Save the new key list
            self.shared_data[f"buf{next_buf}_keys"] = new_keys
            
            # Save the key-original name mapping
            key_mapping = {str(hash(k)): k for k in model_state_dict.keys()}
            self.shared_data[f"buf{next_buf}_mapping"] = key_mapping
            
            # Update the active buffer (perform last)
            self.shared_data["latest"] = next_buf
            print(f"[ModelScheduler] Model state updated to memory buffer {next_buf}")
        else:
            # Use single buffer as usual
            self.shared_data["single_buf"] = {"model_state_dict": model_state_dict}

    def read_state_dict(self):
        """
        Reads the model state from the shared memory.
        """
        if self.use_double_buffer:
            # Read from the active buffer
            current_buf = self.shared_data["latest"]
            keys = self.shared_data.get(f"buf{current_buf}_keys", [])
            
            # If there are no keys, the model hasn't been saved yet
            if not keys:
                return None
                
            # Get the key-original name mapping
            key_mapping = self.shared_data.get(f"buf{current_buf}_mapping", {})
            
            # Collect each tensor to restore the state_dict
            state_dict = {}
            for short_key in keys:
                buffer_key = f"buf{current_buf}_{short_key}"
                if buffer_key in self.shared_data:
                    tensor = self.shared_data[buffer_key]
                    
                    # If needed, decompress (half -> float32)
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()  # restore to float32
                        
                    # Restore the original key name
                    original_key = key_mapping.get(short_key, short_key)
                    state_dict[original_key] = tensor
            
            # Return the state_dict in the usual format
            return {"model_state_dict": state_dict}
        else:
            # Use single buffer as usual
            return self.shared_data.get("single_buf", None)

# ------------------
#  GlobalTimelineScheduler Class with OS Signal-Based Pause/Resume
# ------------------
class GlobalTimelineScheduler:
    """
    The global timeline scheduler will manage the train/eval processes based on the selected mode:
      - default: alternates train/eval each time_slice seconds.
      - fully_parallel: train & eval both run continuously.
      - continuous_eval: evaluation runs continuously; training is intermittent (on/off in time slices).
    """
    def __init__(self, time_slice, mode="default"):
        self.time_slice = time_slice
        self.mode = mode

def safe_kill(pid, sig):
    """
    Given a pid, send a signal.
    If the signal fails, ignore the exception.
    """
    try:
        os.kill(pid, sig)
        return True
    except Exception:
        return False

def global_scheduler_worker(global_scheduler, train_pid, eval_pid):
    """
    The global scheduler logic that uses OS signals to pause/resume processes
    depending on the selected mode. Runs until both train and eval processes complete.
    """
    mode = global_scheduler.mode
    time_slice = global_scheduler.time_slice
    
    train_alive = True
    eval_alive = True
    
    # Helper to check if processes are still alive
    def check_processes_alive():
        nonlocal train_alive, eval_alive
        if train_alive and not safe_kill(train_pid, 0):  # Signal 0 doesn't send a signal but checks if process exists
            train_alive = False
            print("[GlobalScheduler] Training process has terminated.")
        if eval_alive and not safe_kill(eval_pid, 0):
            eval_alive = False
            print("[GlobalScheduler] Evaluation process has terminated.")
        return train_alive or eval_alive  # Return True if at least one is still alive

    if mode == "fully_parallel":
        print("[GlobalScheduler] Mode: fully_parallel => Running both train & eval continuously.")
        # Resume both processes and let them run until completion
        if train_alive:
            if not safe_kill(train_pid, signal.SIGCONT):
                train_alive = False
        if eval_alive:
            if not safe_kill(eval_pid, signal.SIGCONT):
                eval_alive = False

        # Just keep checking if processes are alive
        while check_processes_alive():
            time.sleep(0.1)  # Check every second to prevent busy waiting

        print("[GlobalScheduler] Both processes have completed. Scheduler exiting.")
        sys.exit(0)

    elif mode == "continuous_eval":
        print("[GlobalScheduler] Mode: continuous_eval => eval runs continuously, training toggles in time slices.")
        # Start training paused, eval resumed (continuous)
        if train_alive:
            safe_kill(train_pid, signal.SIGSTOP)
        if eval_alive:
            safe_kill(eval_pid, signal.SIGCONT)

        while check_processes_alive():
            # Toggle training ON for time_slice
            if train_alive:
                print("[GlobalScheduler] Training slice: Resuming training.")
                safe_kill(train_pid, signal.SIGCONT)
                time.sleep(time_slice)

            # Toggle training OFF for time_slice
            if train_alive and check_processes_alive():
                print("[GlobalScheduler] Training paused.")
                safe_kill(train_pid, signal.SIGSTOP)
                time.sleep(time_slice)

        print("[GlobalScheduler] Both processes have completed. Scheduler exiting.")
        sys.exit(0)

    else:
        print("[GlobalScheduler] Mode: default => Alternating train/eval in time slices.")
        # Initial state: training resumed, evaluation paused
        print("[GlobalScheduler] Initial state: training resumed, evaluation paused")
        if train_alive:
            if not safe_kill(train_pid, signal.SIGCONT):
                train_alive = False
        if eval_alive:
            if not safe_kill(eval_pid, signal.SIGSTOP):
                eval_alive = False

        # Alternate train/eval in time slices until both complete
        while check_processes_alive():
            if train_alive:
                print("[GlobalScheduler] Training slice: Resuming training, Pausing evaluation.")
                safe_kill(train_pid, signal.SIGCONT)
                if eval_alive:
                    safe_kill(eval_pid, signal.SIGSTOP)
                time.sleep(time_slice)
            
            # Check again before switching
            if not check_processes_alive():
                break

            if eval_alive:
                print("[GlobalScheduler] Evaluation slice: Pausing training, Resuming evaluation.")
                if train_alive:
                    safe_kill(train_pid, signal.SIGSTOP)
                safe_kill(eval_pid, signal.SIGCONT)
                time.sleep(time_slice)

        print("[GlobalScheduler] Both processes have completed. Scheduler exiting.")
        sys.exit(0)

# ------------------
#  Train Worker Function
# ------------------
def train_worker(args, device, scheduler, lock, model_path, shared_data, config_controller=None):
    # register signal handler
    signal.signal(signal.SIGUSR1, request_config_update_handler)
    
    # Set train worker status as active
    shared_data["train_process_active"] = True
    
    # Initialize model based on provided arguments
    if args.model.startswith("resnet"):
        if args.benchmark == "perm_mnist":
            dataset_name = "mnist"
        elif args.benchmark == "split_cifar10":
            dataset_name = "cifar10"
        elif args.benchmark == "split_cifar100":
            dataset_name = "cifar100"
        else:
            dataset_name = "cifar10" # default resnet
        model = pytorchcv_wrapper.resnet(dataset_name, depth=int(args.model[6:]), pretrained=False)
    else:
        raise ValueError("Model not supported")

    # CL Benchmark Creation
    if args.benchmark == "endless":
        target_transform = None

        # modify output layer to match the number of classes in the benchmark
        model.output = torch.nn.Linear(5184, 5)
        if args.semseg:
            # Remove final_pool to retain spatial resolution
            model.final_pool = nn.Identity()
            # Modify output layer for segmentation: remove fixed upsampling
            model.output = nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, padding=1),  # Increase feature depth
                nn.ReLU(),
                nn.Conv2d(512, 8, kernel_size=1)  # Final segmentation output (8 classes)
            )
            # Override the forward function to upsample dynamically
            def _seg_forward(x):
                input_size = x.shape[-2:]  # Get original input spatial dimensions (e.g., 135x240)
                x = model.features(x)  # Extract features (downsampled)
                x = model.final_pool(x)  # Identity here, so retains current feature map size
                x = model.output(x)  # Apply segmentation head (produces [B, num_classes, H_feat, W_feat])
                # Upsample to the original input image size
                x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

                return x
            model.forward = _seg_forward

            # torch.from_numpy(target).long()
            target_transform = lambda x: torch.from_numpy(x).long()

        benchmark = EndlessCLSim(
            scenario=args.scenario,  # choice from ["Classes", "Illumination", "Weather"]
            sequence_order=None,
            task_order=None,
            semseg=args.semseg,
            dataset_root=args.dataset_root,
            target_transform=target_transform,
        )
    elif args.benchmark == "split_cifar10":
        benchmark = SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "split_cifar100":
        benchmark = SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "core50":
        benchmark = CORe50(scenario=args.scenario_core50, mini=True, object_lvl=False)
        # choice from ["ni", "nc", "nic"]
        # ni - new instances, nc - new classes, nic - new instances and classes
        # mini - True for 32x32, False for 128x128
        # object_lvl – True for a 50-way classification at the object level. False if you want to use the categories as classes.
    elif args.benchmark == "perm_mnist":
        # not tested. We don't need to use this benchmark
        benchmark = PermutedMNIST(n_experiences=3)
    else:
        raise ValueError("Invalid benchmark name")

    if args.benchmark == "endless":
        scenario = args.scenario
        input_size = [3, 64, 64]
    elif args.benchmark == "core50":
        # mini version of core50, resolution 32x32
        scenario = args.scenario_core50
        input_size = [3, 32, 32]
    else:
        scenario = 'na'
        input_size = [3, 32, 32]

    # Will be removed later
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Logging / plugins
    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger("train")
    loggers = [interactive_logger, csv_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        ram_usage_metrics(every=True, minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        MAC_metrics(experience=True),
        loggers=loggers,
    )
    
    training_plugins = []
    # add replay plugin if algorithm is not replay
    if args.algorithm != "replay":
        training_plugins.append(ReplayPlugin(mem_size=args.mem_size))

    if args.optimization == "gem":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
    elif args.optimization == "ewc":
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "both":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "none":
        pass
    else:
        raise ValueError("Invalid optimization name")

    # Choose the CL strategy
    if args.algorithm == "naive":
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
        )
    elif args.algorithm == "replay":
        cl_strategy = Replay(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            plugins=training_plugins,
        )
    elif args.algorithm == "gem":
        cl_strategy = GEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm == "ewc":
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            ewc_lambda=0.5,
            plugins=training_plugins,
        )
    elif args.algorithm == "gss_greedy":
        cl_strategy = GSS_greedy(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            mem_strength=1,
            input_size=[3, 32, 32],
            plugins=training_plugins,
        )
    elif args.algorithm == "agem":
        cl_strategy = AGEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm in ["mir", "scr", "ar1"]:
        raise NotImplementedError("Algorithm not implemented")
    else:
        raise ValueError("Invalid algorithm name")

    print("Training process started.")
    for experience in train_stream:
        print(f"Training for experience: {experience.current_experience}")
        #cl_strategy.train(experience)
        interruptible_train(cl_strategy, experience, config_controller)
        
        # Save model using either double-buffer or single-buffer approach
        if args.enable_double_buffer:
            try:
                # Save model state to double-buffer
                scheduler.write_state_dict(cl_strategy.model.state_dict())
                print("[Train] Model state updated in double-buffer")
                scheduler.write_state_dict(cl_strategy.model.cpu().state_dict())
                cl_strategy.model = cl_strategy.model.to(device)
            except Exception as e:
                print(f"[Train] Error updating double-buffer: {e}")
        else:
            with lock:
                torch.save({
                    'model_state_dict': cl_strategy.model.state_dict(),
                }, model_path)
                print(f"[Train] Model state_dict saved to {model_path}")

    # Mark train worker as inactive when training is completed
    shared_data["train_process_active"] = False
    print("[Train] Training process completed.")


def eval_worker(args, device, scheduler, lock, model_path, shared_data, config_controller=None):
    """
    Evaluation process:
      - Periodically reads the 'latest' state_dict from the shared path
        and performs evaluation if available.
      - Terminates if training process is no longer active.
    """

    # register signal handler
    signal.signal(signal.SIGUSR1, request_config_update_handler)
    
    if args.model.startswith("resnet"):
        if args.benchmark == "perm_mnist":
            dataset_name = "mnist"
        elif args.benchmark == "split_cifar10":
            dataset_name = "cifar10"
        elif args.benchmark == "split_cifar100":
            dataset_name = "cifar100"
        else:
            dataset_name = "cifar10" # default resnet
        model = pytorchcv_wrapper.resnet(dataset_name, depth=int(args.model[6:]), pretrained=False)
    else:
        raise ValueError("Model not supported")

    if args.benchmark == "endless":
        target_transform = None

        # modify output layer to match the number of classes in the benchmark
        model.output = torch.nn.Linear(5184, 5)
        if args.semseg:
            # Remove final_pool to retain spatial resolution
            model.final_pool = nn.Identity()
            # Modify output layer for segmentation: remove fixed upsampling
            model.output = nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, padding=1),  # Increase feature depth
                nn.ReLU(),
                nn.Conv2d(512, 8, kernel_size=1)  # Final segmentation output (8 classes)
            )
            # Override the forward function to upsample dynamically
            def _seg_forward(x):
                input_size = x.shape[-2:]  # Get original input spatial dimensions (e.g., 135x240)
                x = model.features(x)  # Extract features (downsampled)
                x = model.final_pool(x)  # Identity here, so retains current feature map size
                x = model.output(x)  # Apply segmentation head (produces [B, num_classes, H_feat, W_feat])
                # Upsample to the original input image size
                x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

                return x
            model.forward = _seg_forward

            # torch.from_numpy(target).long()
            target_transform = lambda x: torch.from_numpy(x).long()

        benchmark = EndlessCLSim(
            scenario=args.scenario,  # choice from ["Classes", "Illumination", "Weather"]
            sequence_order=None,
            task_order=None,
            semseg=args.semseg,
            dataset_root=args.dataset_root,
            target_transform=target_transform,
        )
    elif args.benchmark == "split_cifar10":
        benchmark = SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "split_cifar100":
        benchmark = SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "core50":
        benchmark = CORe50(scenario=args.scenario_core50, mini=True, object_lvl=False)
        # choice from ["ni", "nc", "nic"]
        # ni - new instances, nc - new classes, nic - new instances and classes
        # mini - True for 32x32, False for 128x128
        # object_lvl – True for a 50-way classification at the object level. False if you want to use the categories as classes.
    elif args.benchmark == "perm_mnist":
        # not tested. We don't need to use this benchmark
        benchmark = PermutedMNIST(n_experiences=3)
    else:
        raise ValueError("Invalid benchmark name")

    if args.benchmark == "endless":
        scenario = args.scenario
        input_size = [3, 64, 64]
    elif args.benchmark == "core50":
        # mini version of core50, resolution 32x32
        scenario = args.scenario_core50
        input_size = [3, 32, 32]
    else:
        scenario = 'na'
        input_size = [3, 32, 32]

    # Will be removed later
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger("train")
    loggers = [interactive_logger, csv_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        ram_usage_metrics(every=True, minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        MAC_metrics(experience=True),
        loggers=loggers,
    )

    training_plugins = []
    if args.algorithm != "replay":
        training_plugins.append(ReplayPlugin(mem_size=args.mem_size))

    if args.optimization == "gem":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
    elif args.optimization == "ewc":
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "both":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "none":
        pass
    else:
        raise ValueError("Invalid optimization name")

    if args.algorithm == "naive":
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
        )
    elif args.algorithm == "replay":
        cl_strategy = Replay(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            plugins=training_plugins,
        )
    elif args.algorithm == "gem":
        cl_strategy = GEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm == "ewc":
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            ewc_lambda=0.5,
            plugins=training_plugins,
        )
    elif args.algorithm == "gss_greedy":
        cl_strategy = GSS_greedy(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            mem_strength=1,
            input_size=input_size,
            plugins=training_plugins,
        )
    elif args.algorithm == "agem":
        cl_strategy = AGEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm in ["mir", "scr", "ar1"]:
        raise NotImplementedError("Algorithm not implemented")
    else:
        raise ValueError("Invalid algorithm name")

    test_stream = benchmark.test_stream

    print("Starting evaluation process...")
    results = []
    begin_time = time.time()
    count = 0
    for _ in range(2000):
        # Check if training process is still active
        if not shared_data.get("train_process_active", True):
            print("[Eval] Training process has terminated. Stopping evaluation process.")
            break
            
        if args.enable_double_buffer:
            try:
                checkpoint = scheduler.read_state_dict()
                if checkpoint is not None and checkpoint['model_state_dict'] is not None:
                    cl_strategy.model.load_state_dict(checkpoint['model_state_dict'])
                    cl_strategy.model = cl_strategy.model.to(device)  # 
                    print("[Eval] Model loaded from double-buffer")
                else:
                    print("[Eval] Model state not found in double-buffer")
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"[Eval] Error loading model from double-buffer: {e}")
                time.sleep(1)
                continue
        else:
            if os.path.exists(model_path):
                with lock:
                    checkpoint = torch.load(model_path, map_location=device)
                    cl_strategy.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[Eval] Model loaded from {model_path}")
            else:
                print(f"[Eval] Model not found in {model_path}")
                time.sleep(1)
                continue
        # Use interruptible evaluation
        eval_results = interruptible_eval(cl_strategy, test_stream, config_controller)
        
        if eval_results:
            results.append(eval_results)
            print("[Eval] Evaluation completed")
        else:
            print("[Eval] Evaluation failed")
            continue                

        count += 1
        print("Evaluation count: ", count)


    print("Evaluation completed")

    print("[Eval] Evaluation process completed.")

# ------------------
#  custom DataLoader for dynamic batch size
# ------------------
class DynamicBatchSizeDataLoader:
    """
    Custom DataLoader class for dynamic batch size
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.dataloader = self._create_dataloader()
        self.iterator = None
        
    def _create_dataloader(self):
        """Create internal PyTorch DataLoader"""
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
    
    def set_batch_size(self, new_batch_size):
        """Change batch size"""
        if new_batch_size == self.batch_size:
            return False  # no change

        print(f"[DynamicDataLoader] Batch size changed: {self.batch_size} -> {new_batch_size}")
        self.batch_size = new_batch_size
        
        # if existing iterator, close it
        if self.iterator is not None:
            try:
                self.dataloader._iterator.close()
            except:
                pass
            self.iterator = None
        
        # create new DataLoader
        self.dataloader = self._create_dataloader()
        return True  # batch size changed
        
    def __iter__(self):
        """Return DataLoader iterator"""
        self.iterator = iter(self.dataloader)
        return self.iterator
    
    def __len__(self):
        """Return DataLoader length"""
        return len(self.dataloader)
# ------------------
#  dynamic configuration controller
# ------------------
class DynamicConfigController:
    """
    Controller for dynamic configuration of training and evaluation
    """
    def __init__(self, shared_data):
        self.shared_data = shared_data
        
        # if initial configuration is not set, initialize with default values
        if "config" not in self.shared_data:
            self.shared_data["config"] = {
                "train_batch_size": 16,  # use args's value as default
                "eval_batch_size": 16,   # use args's value as default
                "timeslice": 1.0
            }
            
        if "config_version" not in self.shared_data:
            self.shared_data["config_version"] = 0
            
        self.local_config_version = 0
    
    def update_config(self, new_config):
        current_config = self.get_current_config()
        changed = False
        
        for key, value in new_config.items():
            if key in current_config and current_config[key] != value:
                current_config[key] = value
                changed = True
                
        if changed:
            self.shared_data["config"] = current_config
            # increase version
            self.shared_data["config_version"] = self.shared_data["config_version"] + 1
            print(f"[ConfigController] Configuration updated: {current_config}")
        
        return changed
    
    def get_current_config(self):
        return self.shared_data["config"].copy()
        
    def check_for_updates(self):
        current_version = self.shared_data["config_version"]
        if current_version > self.local_config_version:
            self.local_config_version = current_version
            return True
        return False

# ------------------
#  interruptible train function
# ------------------
def interruptible_train(cl_strategy, experience, config_controller):
    """
    Interruptible training loop by batch unit
    """
    global CONFIG_UPDATE_REQUESTED
    
    try:
        dataset = experience.dataset
        device = cl_strategy.device
        
        # create dynamic batch size DataLoader
        dynamic_loader = DynamicBatchSizeDataLoader(
            dataset, 
            batch_size=cl_strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        
        criterion = torch.nn.CrossEntropyLoss().to(device)
        cl_strategy.model = cl_strategy.model.to(device)
        
        print(f"[Train] Experience {experience.current_experience} training started, batch size: {cl_strategy.train_mb_size}")
        
        for epoch in range(cl_strategy.train_epochs):
            print(f"[Train] Epoch {epoch+1}/{cl_strategy.train_epochs}")
            
            for i, batch in enumerate(dynamic_loader):
                # check for configuration update
                if CONFIG_UPDATE_REQUESTED:
                    CONFIG_UPDATE_REQUESTED = False
                    
                    if config_controller:
                        current_config = config_controller.get_current_config()
                        if current_config["train_batch_size"] != cl_strategy.train_mb_size:
                            # update batch size
                            new_batch_size = current_config["train_batch_size"]
                            print(f"[Train] Batch size changed: {cl_strategy.train_mb_size} -> {new_batch_size}")
                            cl_strategy.train_mb_size = new_batch_size
                            
                            # update DataLoader
                            dynamic_loader.set_batch_size(new_batch_size)
                
                try:
                    # process batch
                    x, y, task_id = batch
                    
                    # move to device
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    # forward
                    cl_strategy.model.train()
                    outputs = cl_strategy.model(x)
                    
                    # calculate loss
                    loss = criterion(outputs, y)
                    
                    # backward
                    cl_strategy.optimizer.zero_grad()
                    loss.backward()
                    cl_strategy.optimizer.step()
                    
                    if i % 10 == 0:
                        print(f"[Train] Batch {i}, Loss: {loss.item():.4f}, Batch size: {cl_strategy.train_mb_size}")
                
                except Exception as e:
                    print(f"[Train] Error occurred while processing batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"[Train] Experience {experience.current_experience} training completed")
        return True
    
    except Exception as e:
        print(f"[Train] Exception occurred in interruptible_train: {e}")
        import traceback
        traceback.print_exc()
        return False

# ------------------
#  interruptible eval function
# ------------------
def interruptible_eval(cl_strategy, test_stream, config_controller):
    """
    Interruptible evaluation loop
    """
    global CONFIG_UPDATE_REQUESTED
    
    try:
        device = cl_strategy.device
        cl_strategy.model = cl_strategy.model.to(device)
        
        results = {}
        
        for experience in test_stream:
            # check for configuration update
            if CONFIG_UPDATE_REQUESTED:
                CONFIG_UPDATE_REQUESTED = False
                
                if config_controller:
                    current_config = config_controller.get_current_config()
                    if current_config["eval_batch_size"] != cl_strategy.eval_mb_size:
                        # update batch size
                        new_batch_size = current_config["eval_batch_size"]
                        print(f"[Eval] Batch size changed: {cl_strategy.eval_mb_size} -> {new_batch_size}")
                        cl_strategy.eval_mb_size = new_batch_size
            
            dataset = experience.dataset
            
            # create dynamic batch size DataLoader
            dynamic_loader = DynamicBatchSizeDataLoader(
                dataset, 
                batch_size=cl_strategy.eval_mb_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )
            
            print(f"[Eval] Experience {experience.current_experience} evaluation started, batch size: {cl_strategy.eval_mb_size}")
            
            cl_strategy.model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in dynamic_loader:
                    x, y, task_id = batch
                    
                    # move to device
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    # forward
                    outputs = cl_strategy.model(x)
                    
                    # save predictions and targets
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(y.cpu())

                    # check for configuration update (batch-wise)
                    if CONFIG_UPDATE_REQUESTED:
                        CONFIG_UPDATE_REQUESTED = False
                        
                        if config_controller:
                            current_config = config_controller.get_current_config()
                            if current_config["eval_batch_size"] != cl_strategy.eval_mb_size:
                                # update batch size
                                new_batch_size = current_config["eval_batch_size"]
                                print(f"[Eval] Batch size changed: {cl_strategy.eval_mb_size} -> {new_batch_size}")
                                cl_strategy.eval_mb_size = new_batch_size
                                
                                # update DataLoader
                                dynamic_loader.set_batch_size(new_batch_size)
            
            # calculate accuracy
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            accuracy = (all_preds == all_targets).float().mean().item()
            
            results[f"Top1_Acc_Stream/eval_phase/test_stream/Task{experience.current_experience:03d}"] = accuracy
            print(f"[Eval] Experience {experience.current_experience} accuracy: {accuracy:.4f}")
        
        return results
    
    except Exception as e:
        print(f"[Eval] Exception occurred in interruptible_eval: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ------------------
#  dynamic configuration worker
# ------------------
def dynamic_config_worker(config_controller, train_pid, eval_pid, timeslice=10.0):
    """
    Worker for dynamic configuration of batch size
    """
    try:
        # batch size sequence for testing
        train_batch_sizes = [32, 64, 128, 256, 512]
        eval_batch_sizes = [16, 32, 64, 128, 256]
        
        print("[DynamicConfig] Dynamic configuration worker started")

        # change batch size sequentially
        for i in range(len(train_batch_sizes)):
            time.sleep(timeslice)  # wait for a while

            new_config = {
                "train_batch_size": train_batch_sizes[i],
                "eval_batch_size": eval_batch_sizes[i]
            }
            
            # update configuration
            config_controller.update_config(new_config)
            
            # send signal to training and evaluation processes
            safe_kill(train_pid, signal.SIGUSR1)
            safe_kill(eval_pid, signal.SIGUSR1)
            
            print(f"[DynamicConfig] New configuration applied: train batch={train_batch_sizes[i]}, eval batch={eval_batch_sizes[i]}")
        
        print("[DynamicConfig] Configuration change test completed")
    
    except Exception as e:
        print(f"[DynamicConfig] Exception occurred in worker: {e}")
        import traceback
        traceback.print_exc()

# ------------------
#  logging and analysis functions
# ------------------
# queue for timestamp events
timestamp_queue = queue.Queue()

def log_event(event_type, message, batch_idx=None, batch_size=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    event = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "time_seconds": time.time()
    }
    timestamp_queue.put(event)
    print(f"[{timestamp}] [{event_type}] {message}")

def timestamp_logger_worker():
    csv_file = "timeslice_events.csv"
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ["timestamp", "event_type", "message", "batch_idx", "batch_size", "time_seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            try:
                event = timestamp_queue.get(timeout=1.0)
                writer.writerow(event)
                f.flush()  # write immediately to file
                timestamp_queue.task_done()
            except queue.Empty:
                # wait for 1 second if no events
                continue
            except KeyboardInterrupt:
                break

def analyze_batch_size_changes():
    """
    Analyze batch size change history and time
    """
    csv_file = "timeslice_events.csv"
    
    # load data
    events = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['time_seconds'] = float(row['time_seconds'])
            if row['batch_idx'] and row['batch_idx'] != '':
                row['batch_idx'] = int(row['batch_idx'])
            if row['batch_size'] and row['batch_size'] != '':
                row['batch_size'] = int(row['batch_size'])
            events.append(row)
    
    # extract batch size change events
    batch_size_changes = [e for e in events if 'batch size change' in e['message']]
    
    print("\n===== Batch size change analysis =====")
    print(f"Total batch size change events: {len(batch_size_changes)}")
    
    if batch_size_changes:
        # 변경 내역 시각화
        print("\nBatch size change history:")
        for i, change in enumerate(batch_size_changes):
            print(f"  {i+1}. {change['timestamp']} - {change['message']}")
        
        # 변경 간격 계산
        if len(batch_size_changes) > 1:
            intervals = []
            for i in range(1, len(batch_size_changes)):
                interval = batch_size_changes[i]['time_seconds'] - batch_size_changes[i-1]['time_seconds']
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals)
            print(f"\nAverage batch size change interval: {avg_interval:.2f} seconds")
    
    return batch_size_changes

# ------------------
#  Main Function
# ------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    mp.reductions.ForkingPickler = dill.Pickler

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Classes",
        choices=["Classes", "Illumination", "Weather"],
        help="Select scenario: Classes, Illumination, Weather (for EndlessCLSim).",
    )
    parser.add_argument(
        "--scenario_core50",
        type=str,
        default="ni",
        choices=["ni", "nc", "nic"],
        help="Select scenario: ni, nc, nic (for core50).",
    )
    parser.add_argument("--semseg", action="store_true", default=False)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="endless",
                        choices=["endless", "split_cifar10", "split_cifar100", "core50", "perm_mnist"])

    parser.add_argument("--training_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--mem_size", type=int, default=50000)
    parser.add_argument("--algorithm", type=str, default="naive", choices=["naive", "replay", "gem", "ewc",
                                                                           "gss_greedy", "agem", "mir", "scr",
                                                                           "ar1"])
    parser.add_argument("--model", type=str, default="resnet20",
                        choices=["simple_mlp", "resnet20", "resnet56", "resnet110", "resnet1001"])
    parser.add_argument("--optimization", type=str, default="none",
                        choices=["none", "gem", "ewc", "both"])
    parser.add_argument("--download_only", action="store_true", default=False)

    # NEW arguments for controlling the global scheduler
    parser.add_argument("--global_scheduler_mode", type=str, default="default",
                        choices=["default", "fully_parallel", "continuous_eval"],
                        help="Choose how train/eval processes are scheduled.")
    parser.add_argument("--timeslice", type=float, default=1.0,
                        help="Time slice (in seconds) for toggling processes in 'default' or 'continuous_eval' modes.")

    parser.add_argument("--enable_memory_monitor", action="store_true", default=False,
                        help="Enable Jetson-based memory monitor (disabled by default).")

    parser.add_argument("--enable_double_buffer", action="store_true", default=False,
                        help="Enable lock-free double buffering for model state sharing")
    parser.add_argument("--enable_dynamic_reconfiguration", action="store_true", default=False,
                        help="Enable dynamic reconfiguration of batch size")
    parser.add_argument("--reconfiguration_interval", type=float, default=10.0,
                        help="Batch size reconfiguration interval (seconds)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if args.cuda != -1 else "cpu")

    if args.download_only:
        print("Download only mode. Exiting.")
        exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"shared_model_{args.benchmark}_{args.model}_{args.algorithm}_{timestamp}.pth"
    lock = mp.Lock()

    manager = mp.Manager()
    shared_data = manager.dict()
    scheduler = ModelScheduler(shared_data, use_double_buffer=args.enable_double_buffer)
 
    config_controller = None
    if args.enable_dynamic_reconfiguration:
        config_controller = DynamicConfigController(shared_data)
        # initial configuration
        config_controller.update_config({
            "train_batch_size": args.training_bs,
            "eval_batch_size": args.eval_bs,
            "timeslice": args.timeslice
        })
        
    # Create the global timeline scheduler with user-defined time slice + mode
    global_scheduler = GlobalTimelineScheduler(time_slice=args.timeslice, mode=args.global_scheduler_mode)

    if args.enable_dynamic_reconfiguration:
        logger_thread = threading.Thread(target=timestamp_logger_worker)
        logger_thread.daemon = True
        logger_thread.start()
        
    mem_monitor_proc = None
    if args.enable_memory_monitor:
        mem_monitor_proc = mp.Process(target=memory_monitor_worker)
        mem_monitor_proc.start()
        
    # Initialize the shared_data with train process status
    shared_data["train_process_active"] = True
    
    train_proc = mp.Process(target=train_worker, 
                           args=(args, device, scheduler, lock, model_path, shared_data, config_controller))
    eval_proc = mp.Process(target=eval_worker, 
                          args=(args, device, scheduler, lock, model_path, shared_data, config_controller))

    train_proc.start()
    eval_proc.start()

    # Wait a short time to ensure child processes have started
    time.sleep(5)
    train_pid = train_proc.pid
    eval_pid = eval_proc.pid

    dynamic_config_proc = None
    if args.enable_dynamic_reconfiguration:
        dynamic_config_proc = mp.Process(
            target=dynamic_config_worker, 
            args=(config_controller, train_pid, eval_pid, args.reconfiguration_interval)
        )
        dynamic_config_proc.start()
        
    # Launch the global scheduler process with no run_duration
    gs_proc = mp.Process(
        target=global_scheduler_worker, 
        args=(global_scheduler, train_pid, eval_pid)
    )
    gs_proc.start()
    
    train_proc.join()
    eval_proc.join()
    gs_proc.join()
    
    if dynamic_config_proc:
        dynamic_config_proc.terminate()
        dynamic_config_proc.join()
        
        # run batch size change analysis
        print("\nBatch size change analysis running...")
        analyze_batch_size_changes()
        
    if mem_monitor_proc is not None:
        mem_monitor_proc.terminate()
        mem_monitor_proc.join()
    
    print("All processes have completed.")
