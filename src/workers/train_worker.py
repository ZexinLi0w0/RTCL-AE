"""
Training worker implementation.
"""

import torch
import signal
import time
import os
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from avalanche.training import Naive, Replay, EWC, GEM, AGEM, GSS_greedy
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, GEMPlugin, EWCPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
    MAC_metrics,
)
from avalanche.logging import InteractiveLogger, CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from src.globals import TERMINATE_SIGNAL, CONFIG_UPDATE_REQUESTED
from src.utils.signal_handlers import request_config_update_handler, signal_handler
from src.models.model_init import initialize_model
from src.utils.logging_utils import log_event, debug_print, log_info, log_warning, log_error
from src.data.data_loader import DynamicBatchSizeDataLoader

def request_config_update_handler(signum, frame, shared_data):
    """Signal handler to request a configuration update."""
    log_info("[Signal] Configuration update requested via signal.")
    if shared_data:
        shared_data['CONFIG_UPDATE_REQUESTED'] = True
    else:
        log_warning("[Signal] shared_data not available in handler.")

def train_worker(args, device, scheduler, lock, model_path, shared_data, config_controller=None):
    """
    Training worker process.
    Uses standard Replay strategy and handles dynamic config updates via signal.
    """
    if getattr(args, "semseg", False):
        import avalanche.evaluation.metrics.accuracy as _acc_mod
        _acc_mod.is_semseg_acc = True  # spawned process: re-set per-pixel accuracy flag

    # Register signal handler within the process
    handler = lambda signum, frame: request_config_update_handler(signum, frame, shared_data)
    signal.signal(signal.SIGUSR1, handler)
    log_info(f"[TrainWorker] Registered SIGUSR1 handler.")

    # Set train worker status as active
    shared_data["train_process_active"] = True
    
    try:
        # Initialize model
        model = initialize_model(args, args.benchmark)
        
        # Create benchmark
        benchmark = create_benchmark(args)
        train_stream = benchmark.train_stream
        
        # Store total number of experiences in shared data
        shared_data["total_experiences"] = len(train_stream)
        shared_data["current_experience"] = 0  # Set initial current experience value
        
        # Prepare for training
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Set up logging
        interactive_logger = InteractiveLogger()
        csv_logger = CSVLogger("train")
        loggers = [interactive_logger, csv_logger]
        
        # Create evaluation plugin
        eval_plugin = create_evaluation_plugin(loggers)
        
        # Set up training plugins
        training_plugins = create_training_plugins(args)
        # SharedDataLoggerPlugin 추가
        training_plugins.append(SharedDataLoggerPlugin(shared_data))
        # Create continual learning strategy
        cl_strategy = create_cl_strategy(args, model, optimizer, criterion, device, eval_plugin, training_plugins, config_controller, shared_data)
        
        # Training loop
        log_info("Training process started.")
        exp_counter = 0
        for experience in train_stream:
            exp_id = experience.current_experience
            log_info(f"-->> Starting training for Experience {exp_id} <<--")
            shared_data["current_experience"] = exp_id
            # --- Dynamically apply batch size from shared_data for all modes ---
            if "train_batch_size" in shared_data:
                new_bs = shared_data["train_batch_size"]
                if cl_strategy.train_mb_size != new_bs:
                    cl_strategy.train_mb_size = new_bs
                    log_info(f"[Train] Dynamic batch size applied: {cl_strategy.train_mb_size}")
            if "timeslice" in shared_data:
                args.timeslice = shared_data["timeslice"]
                log_info(f"[Train] AdaptOCL timeslice applied: {args.timeslice}")
            if "lr" in shared_data:
                for param_group in cl_strategy.optimizer.param_groups:
                    param_group['lr'] = shared_data["lr"]
                log_info(f"[Train] AdaptOCL learning rate applied: {shared_data['lr']}")
            if "epoch" in shared_data:
                cl_strategy.train_epochs = shared_data["epoch"]
                log_info(f"[Train] AdaptOCL epoch applied: {cl_strategy.train_epochs}")

            # --- DataLoader caching per experience and batch size ---
            dataset = experience.dataset
            batch_size = cl_strategy.train_mb_size
            if not hasattr(cl_strategy, "_train_loader_cache"):
                cl_strategy._train_loader_cache = {}
            cache_key = (exp_id, batch_size)
            if cache_key in cl_strategy._train_loader_cache:
                train_loader = cl_strategy._train_loader_cache[cache_key]
                log_info(f"[Train] Reusing DataLoader for Exp {exp_id} (batch size {batch_size})")
            else:
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                cl_strategy._train_loader_cache[cache_key] = train_loader
                log_info(f"[Train] DataLoader recreated for Exp {exp_id} (batch size {batch_size})")

            # Pass train_loader to interruptible_train if possible (or set as attribute)
            cl_strategy._current_train_loader = train_loader

            # Check for termination signal
            if shared_data.get("TERMINATE_SIGNAL", False):
                log_info("[Train] Termination requested. Stopping training...")
                break
            # Check for initial stop signal
            if shared_data.get("early_stopping_triggered", False):
                log_info("[Train] Initial stop signal detected. Training process completed.")
                break
            # Run direct training loop (using DynamicBatchSizeDataLoader)
            start_time = time.time()
            success = interruptible_train(cl_strategy, experience, config_controller, shared_data, args)
            elapsed = time.time() - start_time
            # accuracy 기록 (마지막 에폭/배치 accuracy 사용 시도)
            last_acc = None
            if hasattr(cl_strategy, 'last_accuracy'):
                last_acc = cl_strategy.last_accuracy
            elif hasattr(cl_strategy, 'eval_plugin') and hasattr(cl_strategy.eval_plugin, 'last_accuracy'):
                last_acc = cl_strategy.eval_plugin.last_accuracy
            shared_data["latest_accuracy"] = last_acc
            shared_data["latest_latency"] = elapsed
            log_info(f"[Train] shared_data update: acc={shared_data.get('latest_accuracy')}, latency={elapsed:.3f}")
            if not success:
                log_error(f"Training failed for experience {exp_id}. Stopping training process.")
                break
            # Save model state
            if args.enable_double_buffer:
                try:
                    scheduler.write_state_dict(cl_strategy.model.state_dict())
                    log_info("[Train] Model state updated in double-buffer")
                    scheduler.write_state_dict(cl_strategy.model.cpu().state_dict())
                    cl_strategy.model = cl_strategy.model.to(device)
                except Exception as e:
                    log_error(f"[Train] Error updating double-buffer: {e}")
            else:
                with lock:
                    torch.save({
                        'model_state_dict': cl_strategy.model.state_dict(),
                    }, model_path)
                    log_info(f"[Train] Model state_dict saved to {model_path}")
            shared_data[f"experience_{exp_id}_completed"] = True
            exp_counter += 1
            if shared_data.get("TERMINATE_SIGNAL", False):
                log_info("[TrainWorker] Termination signal received. Stopping training.")
                break
        shared_data["all_experiences_completed"] = True
        
    except Exception as e:
        log_error(f"[Train] Error in training process: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Mark train worker as inactive
        shared_data["train_process_active"] = False
        log_info("[Train] Training process completed or terminated.")

class SharedDataLoggerPlugin(object):
    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.last_time = None
    def before_training_iteration(self, strategy, **kwargs):
        self.last_time = time.time()
    def after_training_iteration(self, strategy, **kwargs):
        elapsed = time.time() - self.last_time if self.last_time else 0.0
        mb_acc = None
        if hasattr(strategy, 'mb_output') and hasattr(strategy, 'mb_y'):
            preds = strategy.mb_output.argmax(dim=1)
            correct = (preds == strategy.mb_y).sum().item()
            total = strategy.mb_y.size(0)
            mb_acc = correct / total if total > 0 else 0.0
        self.shared_data["latest_accuracy"] = mb_acc
        self.shared_data["latest_latency"] = elapsed
        if strategy.clock.train_iterations % 10 == 0:
            log_info(f"[Train][MiniBatch][Plugin] acc={mb_acc}, latency={elapsed:.3f}")

def interruptible_train(cl_strategy, experience, config_controller, shared_data, args):
    """
    Interruptible training wrapper calling the strategy's public train method.
    """
    def log_memory_state(stage="Before"):
        """Helper function to log memory state."""
        if hasattr(cl_strategy, 'storage_policy') and hasattr(cl_strategy.storage_policy, 'buffer'):
            buffer_size = len(cl_strategy.storage_policy.buffer)
            debug_print(f"[Memory - {stage} Train] Exp {experience.current_experience} Buffer Size: {buffer_size}", args)
            if buffer_size > 0:
                unique_labels = set()
                for _, y, *_ in cl_strategy.storage_policy.buffer:
                    cls = y.item() if torch.is_tensor(y) else int(y)
                    unique_labels.add(cls)
                debug_print(f"[Memory - {stage} Train] Exp {experience.current_experience} Classes: {sorted(list(unique_labels))}", args)
        else:
            debug_print(f"[Memory - {stage} Train] Exp {experience.current_experience} No storage policy/buffer found.", args)

    try:
        log_info(f"Training Experience {experience.current_experience} started")
        # Get current classes before training
        current_classes = set()
        if hasattr(experience, 'classes_in_this_experience'):
             current_classes = set(experience.classes_in_this_experience)
        elif hasattr(experience.dataset, 'targets'):
             try:
                  targets = experience.dataset.targets
                  current_classes = set(targets.unique().tolist()) if isinstance(targets, torch.Tensor) else set(targets)
             except Exception:
                  pass
        debug_print(f"Current classes: {sorted(list(current_classes))}", args)
        debug_print(f"Dataset size: {len(experience.dataset)}", args)

        # Log memory state BEFORE training the experience
        log_memory_state(stage="Before")

        # Avalanche 공식 워크플로우로 복원
        cl_strategy.train(experience)

        # Log memory state AFTER training the experience (memory update happens inside train)
        log_memory_state(stage="After")

        log_info(f"Training Experience {experience.current_experience} completed")
        return True

    except Exception as e:
        log_info(f"Error during training experience {experience.current_experience}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_benchmark(args):
    """
    Create a benchmark based on the provided arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Benchmark instance
    """
    from avalanche.benchmarks.classic import (
        EndlessCLSim, PermutedMNIST, SplitCIFAR10, SplitCIFAR100, SplitMNIST, SplitFMNIST, SplitCUB200, CORe50, SoftRobot
    )
    
    if args.benchmark == "endless":
        target_transform = None
        
        if args.semseg:
            target_transform = lambda x: torch.from_numpy(x).long()
        
        return EndlessCLSim(
            scenario=args.scenario,
            sequence_order=None,
            task_order=None,
            semseg=args.semseg,
            dataset_root=args.dataset_root,
            target_transform=target_transform,
        )
    elif args.benchmark == "split_cifar10":
        # large_model_resnet50 branch: ImageNet backbones need 224x224
        from src.models.model_init import _IMAGENET_BACKBONES
        if args.model in _IMAGENET_BACKBONES:
            from torchvision import transforms
            _tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
            ])
            return SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False,
                                train_transform=_tf, eval_transform=_tf)
        return SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "split_cifar100":
        from src.models.model_init import _IMAGENET_BACKBONES
        if args.model in _IMAGENET_BACKBONES:
            from torchvision import transforms
            _tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])
            return SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False,
                                 train_transform=_tf, eval_transform=_tf)
        return SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "core50":
        return CORe50(scenario=args.scenario_core50, mini=True, object_lvl=False) # due to memory limitations, we use mini
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
        raise ValueError(f"Invalid benchmark name: {args.benchmark}")

def create_evaluation_plugin(loggers):
    """
    Create an evaluation plugin with the specified loggers.
    Removed ram_usage_metrics due to threading issues in the current setup.
    
    Args:
        loggers: List of loggers
        
    Returns:
        Evaluation plugin
    """
    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # ram_usage_metrics(every=True, minibatch=True, epoch=True, experience=True, stream=True), # Disabled due to threading errors
        timing_metrics(epoch=True, experience=True, stream=True),
        MAC_metrics(experience=True),
        loggers=loggers,
    )

def create_training_plugins(args):
    """
    Create training plugins based on the provided arguments.
    """
    training_plugins = []

    # Add replay plugin - Needed by Replay strategy family (including CustomReplay)
    # to manage the experience buffer (storage_policy).
    if args.algorithm == "replay": # Or check if strategy requires replay memory
        log_info("[Plugin] Adding ReplayPlugin for memory management.")
        training_plugins.append(ReplayPlugin(mem_size=args.mem_size))

    # Add optimization plugins
    if args.optimization == "gem":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
    elif args.optimization == "ewc":
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "both":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))

    return training_plugins

def create_cl_strategy(args, model, optimizer, criterion, device, eval_plugin, training_plugins, config_controller=None, shared_data=None):
    """
    Create the Continual Learning strategy based on the provided arguments.
    Uses standard Avalanche strategies.
    """
    # Import Ekya strategy here to avoid circular dependency if Ekya also imports from train_worker
    from src.workers.ekya_strategy import Ekya

    if args.global_scheduler_mode == "ekya":
        log_info(f"[TrainWorker] Creating Ekya strategy with focus: {args.ekya_focus}")
        # Note: Ekya strategy is based on Replay, so ReplayPlugin is managed internally by Ekya strategy if needed.
        # We pass relevant ekya parameters from args.
        return Ekya(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=args.mem_size, # Ekya inherits from Replay, needs mem_size
            train_mb_size=args.training_bs, # Initial batch size
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            plugins=training_plugins, # Base plugins, EkyaPlugin is added inside Ekya constructor
            # Ekya specific parameters from args
            resource_fraction=args.ekya_min_resource, # Or another specific arg like ekya_resource_fraction
            inference_priority=0.6, # Default or from args if available
            enable_profiling=True, # Default or from args if available
            initial_lr=args.lr, # Pass the general lr as initial_lr for Ekya
            ekya_focus=args.ekya_focus
        )

    if args.algorithm == "naive":
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
        )
    elif args.algorithm == "replay":
        debug_print("Creating standard Replay strategy:", args)
        debug_print(f"Memory size: {args.mem_size}", args)
        debug_print(f"Initial Batch size: {args.training_bs}", args)
        # Use the standard Replay strategy
        return Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=args.mem_size,
            train_mb_size=args.training_bs, # Initial batch size
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            plugins=training_plugins # Pass the plugins list which includes ReplayPlugin
        )
    elif args.algorithm == "gem":
        return GEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm == "ewc":
        return EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            ewc_lambda=0.5,
            plugins=training_plugins,
        )
    elif args.algorithm == "gss_greedy":
        # Set input size based on benchmark
        if args.benchmark == "endless":
            input_size = [3, 64, 64]
        elif args.benchmark == "core50":
            input_size = [3, 32, 32]  # Due to memory limit, we use 32x32 instead of 128x128 for core50
        else:
            input_size = [3, 32, 32]  # Default for CIFAR and other datasets
            
        return GSS_greedy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
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
        return AGEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    return cl_strategy