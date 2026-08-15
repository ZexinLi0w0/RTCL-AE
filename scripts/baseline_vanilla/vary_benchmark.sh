#!/bin/sh
python test_avalanche_lib.py --benchmark "split_cifar10" --algorithm "replay"
python test_avalanche_lib.py --benchmark "split_cifar100" --algorithm "replay"
python test_avalanche_lib.py --benchmark "endless" --scenario "Classes" --algorithm "replay"
python test_avalanche_lib.py --benchmark "endless" --scenario "Illumination" --algorithm "replay"
python test_avalanche_lib.py --benchmark "endless" --scenario "Weather" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay"