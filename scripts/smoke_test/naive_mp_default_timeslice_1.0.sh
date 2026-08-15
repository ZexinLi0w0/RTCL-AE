#!/bin/sh
python test_avalanche_lib_mp.py --benchmark "split_cifar10" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "split_cifar100"  --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Classes" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Illumination" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "endless" --scenario "Weather" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "ni" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "nc" --global_scheduler_mode "default"  --timeslice 1.0
python test_avalanche_lib_mp.py --benchmark "core50" --scenario_core50 "nic" --global_scheduler_mode "default"  --timeslice 1.0
