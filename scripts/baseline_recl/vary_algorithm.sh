#!/bin/sh
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "gss_greedy" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "gem" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "agem" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
