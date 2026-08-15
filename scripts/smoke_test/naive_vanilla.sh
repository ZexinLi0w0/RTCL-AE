#!/bin/sh
python test_avalanche_lib.py --benchmark "split_cifar10"
python test_avalanche_lib.py --benchmark "split_cifar100"
python test_avalanche_lib.py --benchmark "endless" --scenario "Classes"
python test_avalanche_lib.py --benchmark "endless" --scenario "Illumination"
python test_avalanche_lib.py --benchmark "endless" --scenario "Weather"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "ni"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nic"
