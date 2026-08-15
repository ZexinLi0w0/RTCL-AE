#!/bin/sh
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4 
