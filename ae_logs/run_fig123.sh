#!/bin/bash
set -u
cd /experiment/zexin/RTCL-AE || exit 1
export PATH=/home/zexin/.conda/envs/ocl/bin:$PATH
export PYTHONUNBUFFERED=1
LD=/experiment/zexin/RTCL-AE/ae_logs
mkdir -p "$LD"
MASTER="$LD/ae_master.log"
TIMEOUT=${TIMEOUT:-36000}

bench_flags(){
  case "$1" in
    split_cifar10)  echo "--benchmark split_cifar10" ;;
    split_cifar100) echo "--benchmark split_cifar100" ;;
    core50_ni)      echo "--benchmark core50 --scenario_core50 ni" ;;
    core50_nc)      echo "--benchmark core50 --scenario_core50 nc" ;;
    core50_nic)     echo "--benchmark core50 --scenario_core50 nic" ;;
  esac
}

run(){ # sub name cmd...
  local sub="$1"; shift
  local name="$1"; shift
  local dir="$LD/$sub"; mkdir -p "$dir"
  local log="$dir/$name.log"
  if grep -q "\[RUN $sub/$name\] END .* rc=0" "$MASTER" 2>/dev/null; then
    echo "[SKIP $sub/$name] already done" >> "$MASTER"; return 0
  fi
  local attempt rc t0 t1
  for attempt in 1 2; do
    echo "[RUN $sub/$name] START attempt=$attempt $(date -Is)" >> "$MASTER"
    t0=$(date +%s)
    timeout -s KILL "$TIMEOUT" "$@" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "[RUN $sub/$name] END attempt=$attempt $(date -Is) rc=$rc dur=$((t1-t0))s" >> "$MASTER"
    [ "$rc" = "0" ] && return 0
    echo "[CLEANUP $sub/$name] rc=$rc killing stragglers" >> "$MASTER"
    pkill -9 -f "python -u main.py" 2>/dev/null
    pkill -9 -f "test_avalanche_lib.py" 2>/dev/null
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
    sleep 8
  done
  return "$rc"
}

# ===== STAGE fig1: GPU utilization traces (tegrastats), CORe50-NC, ER/replay =====
fig1_run(){ # name cmd...
  local name="$1"; shift
  mkdir -p "$LD/fig1"
  if grep -q "\[RUN fig1/$name\] END .* rc=0" "$MASTER" 2>/dev/null; then
    echo "[SKIP fig1/$name] already done" >> "$MASTER"; return 0
  fi
  tegrastats --interval 1000 --logfile "$LD/fig1/tegra_$name.log" &
  local tpid=$!
  run fig1 "$name" "$@"
  local rc=$?
  tegrastats --stop 2>/dev/null
  kill -9 "$tpid" 2>/dev/null
  return "$rc"
}
fig1_run vanilla python -u test_avalanche_lib.py --benchmark core50 --scenario_core50 nc --algorithm replay
fig1_run da   python -u main.py --benchmark core50 --scenario_core50 nc --algorithm replay --global_scheduler_mode default        --training_bs 16 --eval_bs 16
fig1_run fp   python -u main.py --benchmark core50 --scenario_core50 nc --algorithm replay --global_scheduler_mode fully_parallel --training_bs 16 --eval_bs 16
fig1_run ours python -u main.py --benchmark core50 --scenario_core50 nc --algorithm replay --global_scheduler_mode adaptocl       --training_bs 16 --eval_bs 16 --enable_double_buffer
echo "STAGE fig1 DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig1.done"

# ===== STAGE fig2a: alternation methods at fixed 0.1s timeslice =====
for b in split_cifar10 core50_nc core50_nic; do
  bf=$(bench_flags "$b")
  run fig2a "da_$b" python -u main.py $bf --algorithm replay --global_scheduler_mode default           --training_bs 16 --eval_bs 16 --timeslice 0.1
  run fig2a "fp_$b" python -u main.py $bf --algorithm replay --global_scheduler_mode fully_parallel    --training_bs 16 --eval_bs 16 --timeslice 0.1
  run fig2a "ta_$b" python -u main.py $bf --algorithm replay --global_scheduler_mode adaptive_time     --training_bs 16 --eval_bs 16 --timeslice 0.1 --adaptive_priority_percent 0.5
  run fig2a "aa_$b" python -u main.py $bf --algorithm replay --global_scheduler_mode adaptive_accuracy --training_bs 16 --eval_bs 16 --timeslice 0.1 --adaptive_accuracy_threshold 0.4
done
echo "STAGE fig2a DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig2a.done"

# ===== STAGE fig2b: alternation-interval sweep on core50_nic =====
run fig2b "vanilla_core50_nic" python -u test_avalanche_lib.py --benchmark core50 --scenario_core50 nic --algorithm replay
declare -A GM2=( [da]="default" [fp]="fully_parallel" )
for mode in da fp; do
  for ts in 0.001 0.01 0.1 1.0; do
    run fig2b "${mode}_ts${ts}_core50_nic" python -u main.py --benchmark core50 --scenario_core50 nic --algorithm replay --global_scheduler_mode "${GM2[$mode]}" --training_bs 16 --eval_bs 16 --timeslice "$ts"
  done
done
echo "STAGE fig2b DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig2b.done"

# ===== STAGE fig3: batch-size sweep (2-minute smoke test per config) =====
smoke(){ # name cmd...
  local name="$1"; shift
  local dir="$LD/fig3"; mkdir -p "$dir"
  local log="$dir/$name.log"
  if grep -q "\[SMOKE fig3/$name\] END .* verdict=PASS" "$MASTER" 2>/dev/null; then
    echo "[SKIP fig3/$name] already done" >> "$MASTER"; return 0
  fi
  echo "[SMOKE fig3/$name] START $(date -Is)" >> "$MASTER"
  timeout -s KILL 120 "$@" > "$log" 2>&1
  local rc=$?
  local verdict=FAIL
  if [ "$rc" = "0" ] || [ "$rc" = "124" ] || [ "$rc" = "137" ]; then verdict=PASS; fi
  echo "[SMOKE fig3/$name] END $(date -Is) rc=$rc verdict=$verdict" >> "$MASTER"
  if [ "$verdict" = "FAIL" ]; then
    echo "[CLEANUP fig3/$name] rc=$rc killing stragglers" >> "$MASTER"
    pkill -9 -f "python -u main.py" 2>/dev/null
    pkill -9 -f "test_avalanche_lib.py" 2>/dev/null
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
    sleep 8
    return 1
  fi
  return 0
}
for b in split_cifar10 core50_nc core50_nic; do
  bf=$(bench_flags "$b")
  for mode in da fp; do
    for bs in 8 16 32 64 128 256; do
      smoke "${mode}_bs${bs}_${b}" python -u main.py $bf --algorithm replay --global_scheduler_mode "${GM2[$mode]}" --training_bs "$bs" --eval_bs 16 --timeslice 0.1
    done
  done
done
echo "STAGE fig3 DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig3.done"

echo "FIG123 ALL STAGES DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig123_all.done"
