"""
Configuration settings and argument parsing.
"""

import argparse
import os
from src.globals import (
    default_training_batch_size,
    default_eval_batch_size,
    default_timeslice
)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Continual Learning with Avalanche")
    
    # Hardware options
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    
    # Benchmark options
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
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="endless",
        choices=["endless", "split_cifar10", "split_cifar100", "core50", "perm_mnist", "soft_robot"]
    )

    parser.add_argument(
        "--scenario_soft_robot",
        type=str,
        default="ic",
        choices=["ic", "il"], # "ic" for incremental class, "il" for incremental illumination
    )

    # Training options
    parser.add_argument("--training_bs", type=int, default=default_training_batch_size)
    parser.add_argument("--eval_bs", type=int, default=default_eval_batch_size)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--mem_size", type=int, default=50000)
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="naive", 
        choices=["naive", "replay", "gem", "ewc", "gss_greedy", "agem", "mir", "scr", "ar1"]
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet20",
        choices=["simple_mlp", "resnet20", "resnet56", "resnet110", "resnet1001", "resnet50", "resnet101", "resnet152", "resnet200", "efficientnet_b0", "mobilenetv1", "vit_tiny"]
    )
    parser.add_argument(
        "--optimization", 
        type=str, 
        default="none",
        choices=["none", "gem", "ewc", "both"]
    )
    parser.add_argument("--download_only", action="store_true", default=False)

    # Scheduler options
    parser.add_argument(
        "--global_scheduler_mode", 
        type=str, 
        default="default",
        choices=["default", "fully_parallel", "continuous_eval", "adaptive_time", "adaptive_accuracy", "ekya", "adaptocl", "recl_sched"],
        help="Choose how train/eval processes are scheduled."
    )
    parser.add_argument(
        "--timeslice", 
        type=float, 
        default=default_timeslice,
        help="Time slice (in seconds) for toggling processes."
    )
    parser.add_argument(
        "--adaptive_priority_percent", 
        type=float, 
        default=0.3,
        help="For adaptive_time mode: percentage of experiences to prioritize training."
    )
    parser.add_argument(
        "--adaptive_accuracy_threshold", 
        type=float, 
        default=0.4,
        help="For adaptive_accuracy mode: accuracy threshold to switch scheduling."
    )

    # Ekya scheduler options
    parser.add_argument(
        "--ekya_profiling_interval",
        type=float,
        default=5.0,
        help="Interval (in seconds) for Ekya's micro-profiling"
    )
    parser.add_argument(
        "--ekya_min_resource",
        type=float,
        default=0.1,
        help="Minimum resource fraction for Ekya scheduler"
    )
    parser.add_argument(
        "--ekya_max_resource",
        type=float,
        default=1.0,
        help="Maximum resource fraction for Ekya scheduler"
    )
    parser.add_argument(
        "--ekya_utility_threshold",
        type=float,
        default=1.5,
        help="Utility improvement threshold for resource stealing"
    )

    # Ekya focus option
    parser.add_argument(
        "--ekya_focus",
        type=str,
        default="balanced", # Default Ekya behavior
        choices=["balanced", "continuous_eval"],
        help="Focus for Ekya scheduler: 'balanced' (standard Ekya resource allocation) or 'continuous_eval' (prioritize yielding models for continuous evaluation)."
    )

    # Monitoring options
    parser.add_argument(
        "--enable_memory_monitor", 
        action="store_true", 
        default=False,
        help="Enable Jetson-based memory monitor (disabled by default)."
    )

    # Feature flags
    parser.add_argument(
        "--enable_double_buffer", 
        action="store_true", 
        default=False,
        help="Enable lock-free double buffering for model state sharing"
    )
    parser.add_argument(
        "--enable_dynamic_reconfiguration", 
        action="store_true", 
        default=False,
        help="Enable dynamic reconfiguration of batch size"
    )
    parser.add_argument(
        "--reconfiguration_interval", 
        type=float, 
        default=10.0,
        help="Batch size reconfiguration interval (seconds)"
    )
    
    # Maximum runtime
    parser.add_argument(
        "--max_runtime", 
        type=int, 
        default=36000,
        help="Maximum runtime in seconds (default: 10 hour)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config_file", 
        type=str, 
        default=None,
        help="Path to the dynamic configuration file (YAML or JSON)"
    )
    
    # Add debug option
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug messages")
    
    # AdaptOCL options (use --global_scheduler_mode adaptocl)
    parser.add_argument(
        "--uam_omega",
        type=float,
        default=0.5,
        help="UAM: Weight for accuracy term (ω)."
    )
    parser.add_argument(
        "--uam_gamma",
        type=float,
        default=0.5,
        help="UAM: Weight for latency term (γ)."
    )
    parser.add_argument(
        "--uam_eta",
        type=float,
        default=0.5,
        help="UAM: Weight for alternation penalty (η)."
    )
    parser.add_argument(
        "--uam_alpha",
        type=float,
        default=0.5,
        help="UAM: Weight for batch/time-slice reconfiguration penalty (α)."
    )
    parser.add_argument(
        "--uam_delta_acc",
        type=float,
        default=0.01,
        help="UAM: Accuracy improvement threshold for alternation (δ_acc)."
    )
    
    # New AdaptOCL operational focus parameter
    parser.add_argument(
        "--adaptocl_focus",
        type=str,
        default="balanced",
        choices=["balanced", "continuous_eval"],
        help="Focus for AdaptOCL scheduler: 'balanced' (original LA/AA switching) or 'continuous_eval' (prioritize continuous evaluation)."
    )
    
    # New AdaptOCL parameters
    parser.add_argument(
        "--latency_budget",
        type=float,
        default=2.0,
        help="Maximum latency budget for normalization (seconds)."
    )
    parser.add_argument(
        "--switch_penalty",
        type=float,
        default=0.1,
        help="Penalty coefficient (λ) for mode/config switches."
    )

    # RECL Scheduler options
    parser.add_argument(
        "--recl_micro_ms",
        type=int,
        default=1000,
        help="RECL Scheduler: Micro-window length in milliseconds for time-slicing."
    )
    parser.add_argument(
        "--recl_eval_weight",
        type=float,
        default=0.4,
        help="RECL Scheduler: Initial fraction of micro-windows reserved for evaluation (0.0 to 1.0)."
    )
    parser.add_argument(
        "--recl_adapt_alpha",
        type=float,
        default=0.1,
        help="RECL Scheduler: SLO miss ratio threshold to adapt eval_weight (e.g., if miss_ratio > alpha, increase eval_weight)."
    )
    parser.add_argument(
        "--slo_ms",
        type=int,
        default=500,
        help="Evaluation Service Level Objective (SLO) in milliseconds. Used by RECL-SCHED for adaptive weight."
    )
    parser.add_argument(
        "--recl_operation_focus",
        type=str,
        default="balanced",
        choices=["balanced", "inference_focus"],
        help="RECL Scheduler: Operation focus. 'balanced' for adaptive weight, 'inference_focus' to prioritize evaluation."
    )
    parser.add_argument(
        "--recl_min_eval_weight",
        type=float,
        default=0.9,
        help="RECL Scheduler: Minimum evaluation weight when in 'inference_focus' mode (0.0 to 1.0)."
    )

    # Parse arguments
    args = parser.parse_args()
    return args