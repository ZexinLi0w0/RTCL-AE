"""
Evaluation worker implementation that handles interruption for model evaluation.
"""

import torch
import signal
import time
import os
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import traceback
import numpy as np
from collections import defaultdict
import gc
import psutil
from tqdm.auto import tqdm

from avalanche.training import Naive, Replay, EWC, GEM, AGEM, GSS_greedy
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, GEMPlugin, EWCPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
    MAC_metrics,
)
from avalanche.logging import InteractiveLogger, CSVLogger, TensorboardLogger, WandBLogger

from src.globals import TERMINATE_SIGNAL, CONFIG_UPDATE_REQUESTED
from src.utils.signal_handlers import request_config_update_handler, signal_handler
from src.models.model_init import initialize_model
from src.utils.logging_utils import log_event, debug_print, log_info, log_error, log_warning, setup_logger, timestamp_logger_worker, log_accuracy
from src.data.data_loader import DynamicBatchSizeDataLoader
# Import functions from train_worker early to avoid circular imports
from src.workers.train_worker import (
    create_benchmark, 
    create_evaluation_plugin, 
    create_training_plugins, 
    create_cl_strategy
)

# External dependencies
import sys
import random
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

# Constants for timing
EVAL_INTERVAL_SECONDS = 5.0      # Only evaluate every 5 seconds
POLL_INTERVAL_SECONDS = 0.5      # Sleep interval to check for evaluation
MODEL_LOAD_RETRY_SECONDS = 1.0   # Wait time when model loading fails

# CIFAR10 has 10 classes
# CIFAR100 has 100 classes
# CORe50 has 50 classes
threshold_by_num_classes = {
    10: 0.3,   # For CIFAR10, lower accuracy threshold of 30%
    50: 0.2,   # For CORe50, accuracy threshold of 20%
    100: 0.1   # For CIFAR100, low accuracy threshold of 10%
}

# For interruptible evaluation
global last_eval_accuracy
last_eval_accuracy = 0.0  # Initialize with zero

# Set the number of classes based on benchmark
# CIFAR10 has 10 classes
CIFAR10_NUM_CLASSES = 10
# CORe50 has 50 classes
# - ALERT: we use 10 classes for CORe50 classification instead of 50 classes for recognition
CORE50_NUM_CLASSES = 50

class SharedDataLoggerPlugin(object):
    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.last_time = None
    def before_eval_iteration(self, strategy, **kwargs):
        self.last_time = time.time()
    def after_eval_iteration(self, strategy, **kwargs):
        elapsed = time.time() - self.last_time if self.last_time else 0.0
        mb_acc = None
        if hasattr(strategy, 'mb_output') and hasattr(strategy, 'mb_y'):
            preds = strategy.mb_output.argmax(dim=1)
            correct = (preds == strategy.mb_y).sum().item()
            total = strategy.mb_y.size(0)
            mb_acc = correct / total if total > 0 else 0.0
        self.shared_data["latest_accuracy"] = mb_acc
        self.shared_data["latest_latency"] = elapsed
        if strategy.clock.eval_iterations % 10 == 0:
            log_info(f"[Eval][MiniBatch][Plugin] acc={mb_acc}, latency={elapsed:.3f}")

def request_config_update_handler_eval(signum, frame, shared_data):
    """Signal handler for configuration update requests in eval worker."""
    shared_data["CONFIG_UPDATE_REQUESTED"] = True
    log_info("[Eval] Configuration update requested")

def eval_worker(args, device, scheduler, lock, model_path, shared_data, config_controller=None):
    """
    Evaluation process:
      - Periodically reads the 'latest' state_dict from the shared path
        and performs evaluation if available.
      - Terminates if training process is no longer active.
    """
    if getattr(args, "semseg", False):
        import avalanche.evaluation.metrics.accuracy as _acc_mod
        _acc_mod.is_semseg_acc = True  # spawned process: re-set per-pixel accuracy flag
    # Register signal handler
    handler = lambda signum, frame: request_config_update_handler(signum, frame, shared_data)
    signal.signal(signal.SIGUSR1, handler)
    
    try:
        # Initialize model
        model = initialize_model(args, args.benchmark)
        
        # Create benchmark
        benchmark = create_benchmark(args)
        test_stream = benchmark.test_stream
        
        # Initialize CONFIG_UPDATE_REQUESTED flag in shared_data if not present
        if "CONFIG_UPDATE_REQUESTED" not in shared_data:
            shared_data["CONFIG_UPDATE_REQUESTED"] = False
        
        # Manage evaluation counter as a class variable
        class EvalCounter:
            def __init__(self):
                self.count = 0
            
            def increment(self):
                self.count += 1
                return self.count
        
        eval_counter = EvalCounter()
        
        # Initialize benchmark statistics
        shared_data["benchmark_stats"] = {
            "start_time": time.time(),
            "eval_latencies": [],
            "streaming_accuracies": [],
            "total_evaluations": 0,
            "experience_accuracies": defaultdict(list)  # Track accuracy for each experience
        }
        
        # Configure output layer based on benchmark
        if args.benchmark == "split_cifar10":
            num_classes = 10
            log_info(f"[Eval] Setting up model for CIFAR10 with {num_classes} classes")
            model.output = torch.nn.Linear(model.output.in_features, num_classes)
        elif args.benchmark == "core50":
            num_classes = 10 # Classification can be performed at object level (50 classes) or at category level (10 classes), we use category level.
            log_info(f"[Eval] Setting up model for CORe50 with {num_classes} classes")
            model.output = torch.nn.Linear(model.output.in_features, num_classes)
        
        # Logging to verify output layer configuration
        log_info(f"[Eval] Model output layer: {model.output}")

        if args.semseg:
            log_info(f"[Eval] Segmentation output") # disable this for avoiding error in semantic segmentation case study
        else:
            log_info(f"[Eval] Output features: {model.output.out_features}")

        # Log benchmark information
        log_info(f"[Eval] Benchmark: {args.benchmark}")
        log_info(f"[Eval] Test stream length: {len(test_stream)}")
        
        # Prepare for evaluation
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Set up logging
        interactive_logger = InteractiveLogger()
        csv_logger = CSVLogger("eval")
        loggers = [interactive_logger, csv_logger]
        
        # Create evaluation plugin
        eval_plugin = create_evaluation_plugin(loggers)
        
        # Set up training plugins
        training_plugins = create_training_plugins(args)
        # SharedDataLoggerPlugin 추가
        training_plugins.append(SharedDataLoggerPlugin(shared_data))

        # Create continual learning strategy
        cl_strategy = create_cl_strategy(args, model, optimizer, criterion, device, eval_plugin, training_plugins)
        
        # Evaluation loop
        log_info("Starting evaluation process...")
        results = []
        last_eval_time = time.time()
        waiting_for_final_eval = False

        # Main evaluation loop
        while shared_data.get("train_process_active", True) or waiting_for_final_eval:
            if TERMINATE_SIGNAL:
                log_info("[Eval] Termination requested. Stopping evaluation...")
                break

            current_time = time.time()
            if current_time - last_eval_time < EVAL_INTERVAL_SECONDS:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # max_exp_to_eval = shared_data.get("current_experience", -1) # No longer tracking train_worker's progress for choosing experiences

            # Load latest model state
            try:
                if args.enable_double_buffer:
                    raw_loaded_dict = scheduler.read_state_dict()
                    if raw_loaded_dict and 'model_state_dict' in raw_loaded_dict:
                        state_dict = raw_loaded_dict['model_state_dict']
                    elif raw_loaded_dict:
                        state_dict = raw_loaded_dict
                    else:
                        log_warning("[Eval] Failed to read model state from scheduler (None received).")
                        time.sleep(MODEL_LOAD_RETRY_SECONDS)
                        continue
                else:
                    with lock:
                        checkpoint = torch.load(model_path)
                        state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict)
                model.to(device)
            except Exception as e:
                log_warning(f"[Eval] Failed to load model state: {e}")
                time.sleep(MODEL_LOAD_RETRY_SECONDS)
                continue

            current_eval_num = eval_counter.increment()

            # if max_exp_to_eval == -1 and len(test_stream) > 0: # Logic for waiting for first experience no longer needed if evaluating all experiences always
            #      log_info("[Eval] Waiting for train_worker to start the first experience (current_experience is -1).")
            # else:
            log_info(f"[Eval] Starting evaluation cycle #{current_eval_num}. Evaluating ALL experiences in the test stream.")
            
            # active_test_experiences = [exp for exp in test_stream if exp.current_experience <= max_exp_to_eval] # No longer filtering experiences
            
            if not test_stream: # Check if test_stream itself is empty
                log_info("[Eval] Test stream is empty. No experiences to evaluate.")
                last_eval_time = time.time()
                continue

            # Run evaluation for all active experiences together
            eval_start_time = time.time()
            # Pass the full test_stream to interruptible_eval
            eval_result_dict = interruptible_eval(cl_strategy, test_stream, config_controller, shared_data, current_eval_num, args)
            eval_latency = time.time() - eval_start_time # This is the latency for the whole batch of experiences
            
            if eval_result_dict:
                results.append(eval_result_dict) # Store the full result dictionary for this cycle

                # Store overall streaming accuracy for this evaluation cycle
                if 'streaming_accuracy' in eval_result_dict and eval_result_dict['streaming_accuracy'] is not None:
                    shared_data["benchmark_stats"].setdefault("streaming_accuracies", []).append(eval_result_dict['streaming_accuracy'])

                # Store per-experience accuracies from this cycle
                if 'experience_accuracies' in eval_result_dict:
                    for exp_id, acc in eval_result_dict['experience_accuracies'].items():
                        shared_data["benchmark_stats"].setdefault("experience_accuracies", {}).setdefault(exp_id, []).append(acc)
                        # Assuming total_evaluations counts each experience-level evaluation successfully processed
                        shared_data["benchmark_stats"]["total_evaluations"] = shared_data["benchmark_stats"].get("total_evaluations", 0) + 1
                
                # Store overall latency for this evaluation cycle if returned by interruptible_eval
                # interruptible_eval already calculates its internal eval_latency, which is fine.
                # The eval_latency calculated above is for the call to interruptible_eval.
                shared_data["benchmark_stats"].setdefault("eval_latencies", []).append(eval_latency)
            else:
                log_warning("[Eval] interruptible_eval returned None or empty result.")
        
            last_eval_time = time.time()
            
            # Check if training is complete and we need one final evaluation
            if not shared_data.get("train_process_active", True):
                if not waiting_for_final_eval:
                    waiting_for_final_eval = True
                    log_info("[Eval] Training complete. Performing final evaluation...")
                else:
                    break
                    
    except Exception as e:
        log_error(f"[Eval] Error in evaluation process: {e}")
        traceback.print_exc()
    finally:
        log_info("[Eval] Evaluation process completed.")

def interruptible_eval(cl_strategy, experiences_to_evaluate, config_controller, shared_data, current_eval_num, args):
    """
    Interruptible evaluation loop.
    Now modified to evaluate all passed experiences from their beginning.
    """
    try:
        eval_start_time = time.time()
        device = cl_strategy.device
        cl_strategy.model = cl_strategy.model.to(device)
        model = cl_strategy.model
        model.eval()
        debug_print(f"[Eval] Model output layer: {cl_strategy.model.output}", args)

        if args.semseg:
            log_info(f"[Eval] Segmentation model") # disable this class print for avoiding error in semantic segmentation case study
        else:
            debug_print(f"[Eval] Number of classes: {cl_strategy.model.output.out_features}", args)
        
        streaming_accuracy = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        semseg_overall_correct = 0
        semseg_overall_total = 0
        experience_accuracies = {} # Stores accuracy for each experience in this eval cycle
        experience_details = {}    # Stores more details if needed

        # Iterate over all experiences passed to this function
        for exp_idx, experience in enumerate(experiences_to_evaluate):
            dataset = experience.dataset # Full dataset for the current experience
            exp_id = experience.current_experience
            
            # Log that we are processing this experience fully from the start
            log_info(f"[Eval] Processing Exp {exp_id} from start (batch size {cl_strategy.eval_mb_size})")

            # Create DataLoader for the full dataset of the current experience
            # The previous logic for remaining_dataset using last_index is removed
            # to ensure each experience is evaluated fully from its beginning in every call.
            dynamic_loader = DynamicBatchSizeDataLoader(
                dataset, # Use the full dataset for this experience
                batch_size=cl_strategy.eval_mb_size,
                shuffle=False, # Evaluation data should not be shuffled
                num_workers=0, # Default num_workers
                drop_last=False # Do not drop the last batch
            )
            
            cl_strategy.model.eval() # Ensure model is in eval mode for each experience
            exp_correct = 0
            exp_total = 0
            current_exp_preds = [] # Predictions for the current experience
            current_exp_targets = [] # Targets for the current experience
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dynamic_loader):
                    batch_start_time = time.time()
                    x, y, task_id = batch # Assuming task_id is part of the batch, may not be used
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    outputs = cl_strategy.model(x)
                    preds = torch.argmax(outputs, dim=1)

                    if args.semseg:
                        # For semantic segmentation, we need to reshape the predictions
                        preds = preds.view(preds.size(0), -1)
                        y = y.view(y.size(0), -1)

                        # Calculate accuracy for semantic segmentation
                        correct = (preds == y).sum().item()
                        exp_correct += correct
                        exp_total += y.numel()  # Total number of pixels
                    else:
                        current_exp_preds.extend(preds.cpu().numpy())
                        current_exp_targets.extend(y.cpu().numpy())

                        correct_in_batch = (preds == y).sum().item()
                        exp_correct += correct_in_batch
                        exp_total += y.size(0)

                    batch_elapsed_time = time.time() - batch_start_time
                    
                    # Update shared_data with latest batch-level accuracy and latency for this experience
                    # This provides fine-grained updates to the scheduler if needed
                    if exp_total > 0:
                        shared_data["latest_accuracy"] = exp_correct / exp_total 
                    else:
                        shared_data["latest_accuracy"] = 0.0
                    shared_data["latest_latency"] = batch_elapsed_time
                    
                    if batch_idx % 10 == 0: # Log progress every 10 batches for this experience
                        log_info(f"[Eval][Exp {exp_id}][MiniBatch {batch_idx}] acc_so_far={(exp_correct/exp_total if exp_total > 0 else 0.0):.4f}, batch_lat={batch_elapsed_time:.3f}s")
            
            # After processing all batches for the current experience
            if exp_total > 0:
                exp_accuracy_current = exp_correct / exp_total
                experience_accuracies[exp_id] = exp_accuracy_current # Store accuracy for this experience
                
                # For overall streaming accuracy calculation (across all experiences in this eval call)
                all_preds.extend(current_exp_preds)
                all_targets.extend(current_exp_targets)
                # total_samples will be len(all_targets) at the end

                if args.semseg:
                    class_accuracies = 0.0 # bypass this for avoiding error in semantic segmentation case study
                    semseg_overall_correct += exp_correct
                    semseg_overall_total += exp_total
                else:
                    # Per-class accuracy calculation (optional, can be kept or removed)
                    class_accuracies = {}
                    for cls_label in np.unique(current_exp_targets):
                        cls_indices = [i for i, t in enumerate(current_exp_targets) if t == cls_label]
                        if cls_indices:
                            cls_preds_for_class = [current_exp_preds[i] for i in cls_indices]
                            cls_targets_for_class = [current_exp_targets[i] for i in cls_indices]
                            cls_correct_count = sum(1 for p, t in zip(cls_preds_for_class, cls_targets_for_class) if p == t)
                            class_accuracies[int(cls_label)] = cls_correct_count / len(cls_indices)
                
                experience_details[exp_id] = {
                    'accuracy': exp_accuracy_current,
                    'samples': exp_total,
                    'class_accuracies': class_accuracies
                }
                log_info(f"[Eval] Experience {exp_id} evaluation complete: Accuracy: {exp_accuracy_current:.4f} ({exp_correct}/{exp_total})")
            else:
                log_warning(f"[Eval] Experience {exp_id} had no samples processed or all batches were empty.")
                experience_accuracies[exp_id] = 0.0 # Record 0 accuracy if no samples

        # After processing all experiences in experiences_to_evaluate
        overall_eval_latency = time.time() - eval_start_time
        

        if args.semseg:
            # bypass this integrity test
            pass
        else:
            if not all_preds or not all_targets:
                log_warning("[Eval] No predictions or targets collected across all experiences in this evaluation cycle.")
                # Return a structure indicating no results, or an empty one, to avoid breaking the caller
                return {
                    'streaming_accuracy': 0.0,
                    'experience_accuracies': experience_accuracies, # Might contain accuracies for exps with 0 samples
                    'experience_details': experience_details,
                    'eval_latency': overall_eval_latency,
                    'total_samples': 0
                }

        # Calculate overall streaming accuracy across all processed experiences in this call
        if args.semseg:
            overall_correct = semseg_overall_correct
            overall_total_samples = semseg_overall_total
        else:
            overall_correct = sum((p == t) for p, t in zip(all_preds, all_targets))
            overall_total_samples = len(all_targets)
        streaming_accuracy = overall_correct / overall_total_samples if overall_total_samples > 0 else 0.0

        # Update shared_data with the overall streaming accuracy and latency for this evaluation cycle
        shared_data["latest_accuracy"] = streaming_accuracy # This is the overall accuracy from this eval cycle
        shared_data["latest_latency"] = overall_eval_latency # Latency for the whole eval cycle

        # For RECL-SCHED: Report latency in ms, keyed by PID
        if "eval_latency" in shared_data and isinstance(shared_data["eval_latency"], dict):
            try:
                current_pid_str = str(os.getpid())
                latency_ms = overall_eval_latency * 1000.0
                shared_data["eval_latency"][current_pid_str] = latency_ms
                log_info(f"[Eval][RECL_SCHED] Reported latency for PID {current_pid_str}: {latency_ms:.2f} ms")
            except Exception as e:
                log_warning(f"[Eval][RECL_SCHED] Failed to report latency to shared_data['eval_latency']: {e}")
        log_info(f"[Eval] Overall shared_data update: acc={streaming_accuracy:.4f}, latency={overall_eval_latency:.3f}s")
        
        # BENCHMARK PROGRESS REPORT (now reflects all experiences evaluated in this call)
        log_info("\n" + "="*50)
        log_info(f"           BENCHMARK PROGRESS REPORT (Cycle #{current_eval_num})           ")
        log_info("="*50)
        log_info(f"Overall Streaming Accuracy: {streaming_accuracy:.4f} ({overall_correct}/{overall_total_samples})")
        log_info(f"Overall Evaluation Latency: {overall_eval_latency:.2f}s")
        log_info(f"Total Samples in this cycle: {overall_total_samples}")
        log_info("-"*50)
        log_info("Experience Accuracies for this cycle:")
        for exp_id_rep, acc_rep in sorted(experience_accuracies.items()):
            log_info(f"  - Experience {exp_id_rep}: {acc_rep:.4f}")
        log_info("="*50 + "\n")
        
        return {
            'streaming_accuracy': streaming_accuracy,
            'experience_accuracies': experience_accuracies, # Accuracies for experiences evaluated in this call
            'experience_details': experience_details,
            'eval_latency': overall_eval_latency,
            'total_samples': overall_total_samples
        }

    except Exception as e:
        log_error(f"[Eval] Exception occurred in interruptible_eval: {e}")
        traceback.print_exc()
        return None # Or a dict with error info

def _get_num_classes(benchmark):
    # CIFAR10 has 10 classes
    if benchmark.lower() == 'cifar10':
        return 10
    # CORe50 has 50 classes
    elif benchmark.lower() == 'core50':
        return 50
    else:
        return 10  # Default