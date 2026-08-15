#!/bin/sh
python test_avalanche_lib.py --benchmark "soft_robot" --dataset_root "/experiment/.avalanche/data/soft_robot_data_raw/" --scenario_soft_robot "il" --algorithm "replay"
python test_avalanche_lib.py --benchmark "soft_robot" --dataset_root "/experiment/.avalanche/data/soft_robot_data_raw/" --scenario_soft_robot "ic" --algorithm "replay"