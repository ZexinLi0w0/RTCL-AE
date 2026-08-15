#!/bin/sh
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "recl_sched" --training_bs 16 --eval_bs 16 --semseg --enable_double_buffer
python main.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --global_scheduler_mode "recl_sched" --training_bs 16 --eval_bs 16 --semseg --enable_double_buffer
python main.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --global_scheduler_mode "recl_sched" --training_bs 16 --eval_bs 16 --semseg --enable_double_buffer