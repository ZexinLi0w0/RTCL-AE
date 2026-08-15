#!/bin/sh
python test_avalanche_lib.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --semseg
python test_avalanche_lib.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --semseg
python test_avalanche_lib.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --semseg