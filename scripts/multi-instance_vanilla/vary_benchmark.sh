#!/bin/bash
# Usage: ./vary_benchmark.sh <parallel_instances>
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <parallel_instances>"
  exit 1
fi

K="$1"

# List of benchmark commands
commands=(
  "python test_avalanche_lib.py --benchmark split_cifar10 --algorithm replay"
  "python test_avalanche_lib.py --benchmark split_cifar100 --algorithm replay"
  "python test_avalanche_lib.py --benchmark core50 --scenario_core50 ni --algorithm replay"
  "python test_avalanche_lib.py --benchmark core50 --scenario_core50 nc --algorithm replay"
  "python test_avalanche_lib.py --benchmark core50 --scenario_core50 nic --algorithm replay"
)

for cmd in "${commands[@]}"; do
  echo "Launching $K instances of: $cmd"
  for i in $(seq 1 "$K"); do
    echo "Executing instance $i: $cmd"
    $cmd &
  done
  wait
done