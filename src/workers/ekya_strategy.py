"""
Ekya strategy implementation for resource-aware continual learning.
Based on the paper: "Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers"
"""

import torch
import time
import math
import numpy as np
import random
from collections import defaultdict
from typing import Optional, List, Union, Sequence, Callable, Dict, Any
from torch.utils.data import DataLoader, Subset
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
from avalanche.training import Replay
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

# -------------------------------------------------------------
# Micro‑profiler ------------------------------------------------
# -------------------------------------------------------------

class MicroProfiler:
    """Estimate (accuracy, epoch_time) for multiple retraining configs
    on a *small* subset of the current experience. Inspired by Ekya §4.3.
    """

    def __init__(self, sample_frac: float = 0.1, probe_epochs: int = 5):
        self.sample_frac = sample_frac
        self.probe_epochs = probe_epochs

    @staticmethod
    def _fit_learning_curve(epoch_acc: List[float]):
        """Very small NNLS‑style fitting:  A * (1 - e^{-k * x})."""
        import numpy as np
        xs = np.arange(1, len(epoch_acc) + 1)
        ys = np.array(epoch_acc)
        # crude estimate of asymptote using last accuracy
        A = ys[-1]
        if A < 1e-4:
            return lambda e: 0.0
        # simple k from half‑life
        half_idx = max(1, len(epoch_acc) // 2) - 1
        k = -math.log(1 - ys[half_idx] / A) / (half_idx + 1)
        return lambda e: float(A * (1 - math.exp(-k * e)))

    def profile(self, model, train_dataset, loss_fn, optim_cls, configs: List[Dict[str, Any]], device="cuda"):
        """Return dict config_id -> {acc_fn, epoch_time, data_len}
        config dict must contain keys: id, batch_size, lr, epochs (target)
        
        Args:
            model: original model (copy will be created)
            train_dataset: training dataset
            loss_fn: loss function
            optim_cls: optimizer class
            configs: list of configs to test (batch size, lr, etc.)
            device: training device
        """
        profiles = {}
        if len(train_dataset) == 0:
            return profiles

        # more secure way to handle return values of data loader
        try:
            # define safe sampling function
            def safe_get_batch(loader):
                try:
                    for data in loader:
                        return data
                except Exception as e:
                    print(f"[MicroProfiler] Error in test loader: {e}")
                    return None
                
            # create test loader and get first batch
            test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            first_batch = safe_get_batch(test_loader)
            
            if first_batch is None:
                print("[MicroProfiler] Warning: Failed to get test batch")
                return profiles
                
            print(f"[MicroProfiler] First batch type: {type(first_batch)}, length: {len(first_batch) if hasattr(first_batch, '__len__') else 'not iterable'}")
            
            # check type of return value of data loader
            if isinstance(first_batch, tuple):
                print(f"[MicroProfiler] Batch is tuple with {len(first_batch)} elements")
            elif isinstance(first_batch, list):
                print(f"[MicroProfiler] Batch is list with {len(first_batch)} elements")
                if len(first_batch) >= 2:
                    print(f"[MicroProfiler] List contains input and label")
            
            # Test criterion function call method
            try:
                # first check call signature
                import inspect
                if hasattr(loss_fn, '__call__'):
                    sig = inspect.signature(loss_fn)
                    print(f"[MicroProfiler] Criterion signature: {sig}")
                    
                    # check number of required parameters
                    param_count = len(sig.parameters)
                    print(f"[MicroProfiler] Criterion requires {param_count} parameters")
                else:
                    print(f"[MicroProfiler] Criterion type: {type(loss_fn)}")
            except Exception as e:
                print(f"[MicroProfiler] Error inspecting criterion: {e}")
            
        except Exception as e:
            print(f"[MicroProfiler] Error checking dataset format: {e}")
            # continue if error occurs
        
        print("[MicroProfiler] Preparing subset for profiling...")
        # sample subset indices
        subset_len = max(1, int(len(train_dataset) * self.sample_frac))
        subset_indices = random.sample(range(len(train_dataset)), subset_len)
        subset_ds = Subset(train_dataset, subset_indices)

        import copy
        for cfg in configs:
            cfg_id = cfg["id"]
            batch_size = cfg.get("batch_size", 32)
            lr = cfg.get("lr", 1e-3)

            # deep copy model
            try:
                model_copy = copy.deepcopy(model)
                model_copy = model_copy.to(device)
                model_copy.train()
            except Exception as e:
                print(f"[Ekya] Warning: Failed to copy model: {e}")
                # skip this config if copy fails
                continue
                
            optimizer = optim_cls(model_copy.parameters(), lr=lr)
            loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=True)
            print(f"[MicroProfiler] Starting profiling for config {cfg_id} with batch size {batch_size}")

            epoch_acc = []
            epoch_times = []
            for epoch in range(self.probe_epochs):
                correct = 0
                total = 0
                start = time.time()
                
                for batch in loader:
                    # more flexible handling of dataset return structure
                    try:
                        # handle various batch formats
                        if isinstance(batch, tuple):
                            # tuple format (default format)
                            if len(batch) == 2:
                                x, y = batch
                            elif len(batch) >= 3:
                                x, y = batch[0], batch[1]  # use first two elements
                            else:
                                print(f"[MicroProfiler] Unexpected tuple batch structure with {len(batch)} elements")
                                continue
                        elif isinstance(batch, list):
                            # list format (CIFAR10, etc.)
                            if len(batch) >= 2:
                                x, y = batch[0], batch[1]
                            else:
                                print(f"[MicroProfiler] Unexpected list batch structure with {len(batch)} elements")
                                continue
                        elif hasattr(batch, 'x') and hasattr(batch, 'y'):
                            # object format
                            x, y = batch.x, batch.y
                        else:
                            print(f"[MicroProfiler] Unsupported batch type: {type(batch)}")
                            continue
                            
                        # check data type and dimension
                        if not isinstance(x, torch.Tensor):
                            x = torch.tensor(x)
                        if not isinstance(y, torch.Tensor):
                            y = torch.tensor(y)
                            
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        out = model_copy(x)
                        
                        # try various loss function call methods
                        try:
                            # first way: standard way (output, target)
                            loss = loss_fn(out, y)
                        except TypeError:
                            try:
                                # second way: one argument only (some SupervisedProblem)
                                loss = loss_fn((out, y))
                            except TypeError:
                                try:
                                    # third way: dictionary format (some SupervisedProblem)
                                    loss = loss_fn({"predictions": out, "targets": y})
                                except TypeError:
                                    # fourth way: custom loss function
                                    print(f"[MicroProfiler] Using custom loss function for config {cfg_id}")
                                    loss = torch.nn.functional.cross_entropy(out, y)
                        
                        loss.backward()
                        optimizer.step()
                        preds = out.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                    except Exception as e:
                        print(f"[MicroProfiler] Error processing batch: {e}")
                        # skip batch if error occurs
                        continue
                        
                epoch_times.append(time.time() - start)
                if total > 0:
                    epoch_acc.append(correct / total)
                    print(f"[MicroProfiler] Epoch {epoch} - Accuracy: {correct/total:.4f}, Time: {epoch_times[-1]:.4f}s")
                else:
                    epoch_acc.append(0.0)
                    print(f"[MicroProfiler] Epoch {epoch} - No valid samples processed")
                    
            # check if at least one batch was successfully processed
            if not epoch_acc:
                print(f"[MicroProfiler] No valid accuracy data for config {cfg_id}")
                continue
                
            # fit curve + store avg time per epoch
            acc_fn = self._fit_learning_curve(epoch_acc)
            profiles[cfg_id] = {
                "acc_fn": acc_fn,
                "epoch_time": sum(epoch_times) / max(1, len(epoch_times)),
                "subset_size": subset_len,
            }
            print(f"[MicroProfiler] Profiling completed for config {cfg_id} - Avg epoch time: {profiles[cfg_id]['epoch_time']:.4f}s")
            del model_copy
            torch.cuda.empty_cache()
        
        if not profiles:
            print("[MicroProfiler] Warning: No valid profiles generated")
        else:
            print(f"[MicroProfiler] Successfully profiled {len(profiles)} configurations")
            
        return profiles

# -------------------------------------------------------------
# Thief Scheduler  --------------------------------------------
# -------------------------------------------------------------

class ThiefScheduler:
    """Simplified version of Algorithm 1 in the Ekya paper.
    Assumes each *stream* has - current_accuracy, - list of cfg_ids, - micro‑profile table.
    Returns gpu_fraction per stream and chosen cfg.
    """

    def __init__(self, delta: float = 0.1):
        self.delta = delta  # gpu quantum

    def _estimate_window_accuracy(self, base_acc: float, cfg_profile: Dict[str, Any], gpu_alloc: float, window_secs: float, target_epochs: int):
        # scaled training time with alloc (assume linear)
        t_full = cfg_profile["epoch_time"] * target_epochs
        t_alloc = t_full / max(1e-4, gpu_alloc)  # if 0 alloc => inf
        if t_alloc >= window_secs:
            return base_acc  # no training finishes in window
        # portion of window before new model available
        t_inf_phase = t_alloc
        t_new_phase = window_secs - t_inf_phase
        new_acc = cfg_profile["acc_fn"](target_epochs)
        return (base_acc * t_inf_phase + new_acc * t_new_phase) / window_secs

    def schedule(self, streams: List[Dict[str, Any]], profiles: Dict[str, Dict[str, Any]], window_secs: float = 200.0) -> Dict[int, Dict[str, Any]]:
        """streams: each dict {id, cur_acc, target_epochs, cfg_ids}
        returns decision dict: id -> {gpu_frac, cfg_id, exp_acc}
        """
        # input validation
        if not streams:
            print("[ThiefScheduler] Warning: No streams provided")
            return {}
            
        if not profiles:
            print("[ThiefScheduler] Warning: No profiles provided")
            return {}
            
        # check if all streams have valid cfg_ids
        valid_streams = []
        for s in streams:
            valid_cfg_ids = [cfg_id for cfg_id in s.get("cfg_ids", []) if cfg_id in profiles]
            if valid_cfg_ids:
                s_copy = s.copy()
                s_copy["cfg_ids"] = valid_cfg_ids
                valid_streams.append(s_copy)
            else:
                print(f"[ThiefScheduler] Warning: Stream {s.get('id')} has no valid config IDs")
                
        if not valid_streams:
            print("[ThiefScheduler] Warning: No valid streams after filtering")
            # default decision: equal resource allocation to each stream
            return {s["id"]: {"gpu_frac": 1.0 / len(streams), "cfg_id": None, "expected_acc": s.get("cur_acc", 0.0)} 
                    for s in streams}
            
        print(f"[ThiefScheduler] Scheduling {len(valid_streams)} streams with window size {window_secs}s")
        
        try:
            n = len(valid_streams)
            # start with equal gpu share
            alloc = {s["id"]: 1.0 / n for s in valid_streams}
            best_acc = {s["id"]: s.get("cur_acc", 0.0) for s in valid_streams}
            best_cfg = {s["id"]: None for s in valid_streams}
            
            # set max number of iterations
            max_iterations = 100
            iteration = 0
            improved = True
            
            while improved and iteration < max_iterations:
                iteration += 1
                improved = False
                
                for thief in valid_streams:
                    thief_id = thief["id"]
                    for victim in valid_streams:
                        if victim["id"] == thief_id:
                            continue
                        # attempt to steal delta from victim
                        if alloc[victim["id"]] <= self.delta:
                            continue
                        tmp_alloc = alloc.copy()
                        tmp_alloc[victim["id"]] -= self.delta
                        tmp_alloc[thief_id] += self.delta
                        # evaluate total accuracy
                        tot = 0.0
                        tmp_best_cfg = {}
                        for s in valid_streams:
                            sid = s["id"]
                            best_stream_acc = -1
                            chosen_cfg = None
                            for cfg_id in s["cfg_ids"]:
                                try:
                                    acc = self._estimate_window_accuracy(
                                        s.get("cur_acc", 0.0), profiles[cfg_id], tmp_alloc[sid], window_secs, s.get("target_epochs", 1))
                                    if acc > best_stream_acc:
                                        best_stream_acc = acc
                                        chosen_cfg = cfg_id
                                except Exception as e:
                                    print(f"[ThiefScheduler] Error estimating accuracy for config {cfg_id}: {e}")
                                    # skip this config
                                    continue
                            tmp_best_cfg[sid] = chosen_cfg
                            tot += best_stream_acc
                        # if total accuracy improves, accept
                        prev_tot = sum(best_acc.values())
                        if tot > prev_tot + 1e-4:  # small margin
                            alloc = tmp_alloc
                            for s in valid_streams:
                                sid = s["id"]
                                if tmp_best_cfg[sid] is not None:
                                    try:
                                        best_acc[sid] = self._estimate_window_accuracy(
                                            s.get("cur_acc", 0.0), profiles[tmp_best_cfg[sid]], alloc[sid], window_secs, s.get("target_epochs", 1))
                                    except Exception as e:
                                        print(f"[ThiefScheduler] Error updating accuracy for stream {sid}: {e}")
                                        # keep current accuracy
                                        best_acc[sid] = s.get("cur_acc", 0.0)
                                best_cfg[sid] = tmp_best_cfg[sid]
                            improved = True
                            print(f"[ThiefScheduler] Iteration {iteration}: Improved allocation - total acc: {tot:.4f} (was {prev_tot:.4f})")
                
            # list of original stream IDs remaining in memory
            all_stream_ids = {s["id"] for s in streams}
            
            # build result
            decision = {}
            for sid in all_stream_ids:
                if sid in alloc:
                    decision[sid] = {
                        "gpu_frac": alloc[sid], 
                        "cfg_id": best_cfg[sid], 
                        "expected_acc": best_acc[sid]
                    }
                else:
                    # invalid streams will get default allocation
                    decision[sid] = {
                        "gpu_frac": 1.0 / len(streams), 
                        "cfg_id": None, 
                        "expected_acc": 0.0
                    }
                    
            print(f"[ThiefScheduler] Final allocation after {iteration} iterations:")
            for sid, d in decision.items():
                print(f"  Stream {sid}: GPU {d['gpu_frac']:.2f}, Config {d['cfg_id']}, Expected acc {d['expected_acc']:.4f}")
                
            return decision
            
        except Exception as e:
            import traceback
            print(f"[ThiefScheduler] Error during scheduling: {e}")
            traceback.print_exc()
            
            # return default decision if error occurs
            return {s["id"]: {"gpu_frac": 1.0 / len(streams), "cfg_id": None, "expected_acc": s.get("cur_acc", 0.0)} 
                    for s in streams}

# EkyaPlugin should only handle resource scheduling (batch size, learning rate, resource fraction)
class EkyaPlugin(SupervisedPlugin):
    """
    EkyaPlugin handles resource scheduling (batch size, learning rate, resource fraction)
    based on profiling and scheduling logic. It does NOT touch memory or replay buffer.
    Batch size and learning rate are adjusted by updating strategy.train_mb_size and optimizer.param_groups,
    relying on Avalanche's dataloader and optimizer mechanisms, just like other modes.
    """
    def __init__(self, resource_fraction=0.5, inference_priority=0.6, min_retraining_samples=100, resource_check_interval=10, stream_id=0, probe_epochs=3, window_secs=200.0, enable_profiling=True, ekya_focus="balanced"):
        super().__init__()
        self.resource_fraction = resource_fraction
        self.inference_priority = inference_priority
        self.min_retraining_samples = min_retraining_samples
        self.resource_check_interval = resource_check_interval
        self.stream_id = stream_id
        self.window_secs = window_secs
        self.enable_profiling = enable_profiling
        self.ekya_focus = ekya_focus
        self.last_accuracy = 0

    def before_training_exp(self, strategy, **kwargs):
        # Optionally adjust batch size or learning rate before each experience
        # Example: set initial batch size and learning rate
        strategy.train_mb_size = getattr(strategy, 'train_mb_size', 32)
        for param_group in strategy.optimizer.param_groups:
            param_group['lr'] = getattr(strategy, 'initial_lr', 0.001)
        print(f"[EkyaPlugin] Set batch size to {strategy.train_mb_size}, learning rate to {param_group['lr']}")

    def after_eval(self, strategy, **kwargs):
        # Adjust batch size and learning rate after evaluation, using Avalanche's mechanisms
        if hasattr(strategy, 'evaluator') and strategy.evaluator is not None:
            metrics = strategy.evaluator.get_last_metrics()
            if metrics and 'Top1_Acc_Stream' in metrics:
                acc = metrics['Top1_Acc_Stream/eval_phase/test_stream']
                self.last_accuracy = acc
                print(f"[EkyaPlugin] Current accuracy: {acc:.4f}, Focus: {self.ekya_focus}")

                if self.ekya_focus == "continuous_eval":
                    # For continuous_eval, prioritize more frequent updates.
                    # Keep batch size moderate, prevent it from growing too large.
                    # Adjust LR more gently.
                    if acc < 0.65 and strategy.train_mb_size > 16: # Slightly higher acc threshold for reduction
                        strategy.train_mb_size = max(16, strategy.train_mb_size // 2)
                        print(f"[EkyaPlugin][ContEval] Reducing batch size to {strategy.train_mb_size}")
                        for param_group in strategy.optimizer.param_groups:
                            param_group['lr'] = max(1e-5, param_group['lr'] * 0.8) # Gentler LR reduction
                            print(f"[EkyaPlugin][ContEval] Reducing learning rate to {param_group['lr']}")
                    elif acc > 0.75 and strategy.train_mb_size < 64: # Lower cap for BS increase
                        strategy.train_mb_size = min(64, strategy.train_mb_size * 2)
                        print(f"[EkyaPlugin][ContEval] Increasing batch size to {strategy.train_mb_size}")
                        for param_group in strategy.optimizer.param_groups:
                            param_group['lr'] = min(0.01, param_group['lr'] * 1.1) # Gentler LR increase
                            print(f"[EkyaPlugin][ContEval] Increasing learning rate to {param_group['lr']}")
                    # Resource fraction adjustments could also be made more conservative
                    # For now, focusing on batch_size and LR to make training iterations potentially shorter/more frequent.
                else: # balanced focus (original logic)
                    # Example: adjust resource_fraction for logging/analysis
                    if acc > 0.8 and self.resource_fraction > 0.4:
                        self.resource_fraction *= 0.95
                    elif acc < 0.6:
                        self.resource_fraction = min(0.9, self.resource_fraction * 1.2)
                        print(f"[EkyaPlugin] (LOG) Would increase resource fraction to {self.resource_fraction:.2f}")
                    # Adjust batch size and learning rate as in other modes
                    if acc < 0.5 and strategy.train_mb_size > 8:
                        strategy.train_mb_size = max(8, strategy.train_mb_size // 2)
                        print(f"[EkyaPlugin] Reducing batch size to {strategy.train_mb_size} due to low accuracy")
                        for param_group in strategy.optimizer.param_groups:
                            param_group['lr'] = max(1e-5, param_group['lr'] * 0.7)
                            print(f"[EkyaPlugin] Reducing learning rate to {param_group['lr']}")
                    elif acc > 0.8 and strategy.train_mb_size < 128:
                        strategy.train_mb_size = min(128, strategy.train_mb_size * 2)
                        print(f"[EkyaPlugin] Increasing batch size to {strategy.train_mb_size} due to high accuracy")
                        for param_group in strategy.optimizer.param_groups:
                            param_group['lr'] = min(0.01, param_group['lr'] * 1.2)
                            print(f"[EkyaPlugin] Increasing learning rate to {param_group['lr']}")

class Ekya(Replay):
    """
    Ekya strategy inherits from Avalanche Replay and relies 100% on its memory management.
    Ekya only adds resource scheduling logic via EkyaPlugin (batch size, learning rate, resource fraction).
    """
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        train_mb_size: int = 32,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every: int = -1,
        resource_fraction: float = 0.5,
        mem_size: int = 5000,
        inference_priority: float = 0.6,
        enable_profiling: bool = True,
        initial_lr: float = 0.001,
        ekya_focus: str = "balanced",
        **kwargs
    ):
        # Add EkyaPlugin for resource scheduling
        ekya_plugin = EkyaPlugin(
            resource_fraction=resource_fraction,
            inference_priority=inference_priority,
            enable_profiling=enable_profiling,
            ekya_focus=ekya_focus
        )
        if plugins is None:
            plugins = []
        plugins.append(ekya_plugin)
        if evaluator is None:
            from avalanche.training.plugins.evaluation import default_evaluator
            evaluator = default_evaluator
        # Store initial learning rate for use in plugin
        self.initial_lr = initial_lr
        # Use Avalanche Replay for memory management
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            mem_size=mem_size,
            **kwargs
        )
        self.ekya_plugin = ekya_plugin
        print(f"[Ekya] Initialized with Avalanche Replay memory (size={mem_size}), resource scheduling only.")

    # No custom memory/replay code. All memory management is handled by Avalanche Replay.
    # Only resource scheduling logic is handled by EkyaPlugin. 