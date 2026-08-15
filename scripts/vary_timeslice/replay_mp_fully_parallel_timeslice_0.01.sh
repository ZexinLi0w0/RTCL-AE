#!/bin/sh
python test_avalanche_lib_mp.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "split_cifar100"  --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "fully_parallel"  --timeslice 0.01
