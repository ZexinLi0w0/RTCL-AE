#!/bin/sh
python test_avalanche_lib.py --benchmark "split_cifar10" --download_only
python test_avalanche_lib.py --benchmark "split_cifar100" --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Classes" --semseg --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Illumination" --semseg --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Weather" --semseg --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Classes" --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Illumination" --download_only
python test_avalanche_lib.py --benchmark "endless" --scenario "Weather" --download_only
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "ni" --download_only
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --download_only
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nic" --download_only