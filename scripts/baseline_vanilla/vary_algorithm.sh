#!/bin/sh
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "gss_greedy"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "gem"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "agem"