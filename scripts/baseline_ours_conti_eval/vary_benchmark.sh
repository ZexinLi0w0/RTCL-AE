#!/bin/sh
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "adaptocl" --adaptocl_focus continuous_eval --eval_bs 16 --enable_double_buffer
