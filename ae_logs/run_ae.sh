#!/bin/bash
set -u
cd /experiment/zexin/RTCL-AE || exit 1
export PATH=/home/zexin/.conda/envs/ocl/bin:$PATH
export PYTHONUNBUFFERED=1
LD=/experiment/zexin/RTCL-AE/ae_logs
mkdir -p "$LD"
MASTER="$LD/ae_master.log"
DSR=/experiment/.avalanche/data/soft_robot_data_raw/
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

BENCHES="split_cifar10 split_cifar100 core50_ni core50_nc core50_nic"
declare -A GM=( [da]="default" [ekya]="ekya" [recl]="recl_sched" [fp]="fully_parallel" [ours]="adaptocl" )

# ===== STAGE tab4: ablation new configs (TA=adaptive_time, AA=adaptive_accuracy), ER ResNet-20 =====
for b in $BENCHES; do
  bf=$(bench_flags "$b")
  run tab4 "adaptive_time_$b"     python -u main.py $bf --algorithm replay --global_scheduler_mode adaptive_time     --training_bs 16 --eval_bs 16 --adaptive_priority_percent 0.5
  run tab4 "adaptive_accuracy_$b" python -u main.py $bf --algorithm replay --global_scheduler_mode adaptive_accuracy --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
done
echo "STAGE tab4 DONE $(date -Is)" >> "$MASTER"; touch "$LD/tab4.done"

# ===== STAGE fig7: CORe50-NC x 4 algorithms x 6 methods =====
for a in replay gss_greedy gem agem; do
  run fig7 "vanilla_$a" python -u test_avalanche_lib.py --benchmark core50 --scenario_core50 nc --algorithm "$a"
  run fig7 "da_$a"   python -u main.py --benchmark core50 --scenario_core50 nc --algorithm "$a" --global_scheduler_mode default        --training_bs 16 --eval_bs 16
  run fig7 "ekya_$a" python -u main.py --benchmark core50 --scenario_core50 nc --algorithm "$a" --global_scheduler_mode ekya           --training_bs 16 --eval_bs 16
  run fig7 "recl_$a" python -u main.py --benchmark core50 --scenario_core50 nc --algorithm "$a" --global_scheduler_mode recl_sched     --eval_bs 16 --enable_double_buffer
  run fig7 "fp_$a"   python -u main.py --benchmark core50 --scenario_core50 nc --algorithm "$a" --global_scheduler_mode fully_parallel --training_bs 16 --eval_bs 16
  run fig7 "ours_$a" python -u main.py --benchmark core50 --scenario_core50 nc --algorithm "$a" --global_scheduler_mode adaptocl       --eval_bs 16 --enable_double_buffer
done
echo "STAGE fig7 DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig7.done"

# ===== STAGE tab3: robotic case study =====
endless_run(){ # task(ilc|ss) scenario method
  local task="$1" scen="$2" m="$3"; local semseg=""; [ "$task" = "ss" ] && semseg="--semseg"
  local name="${m}_${task}_${scen}"
  if [ "$m" = "vanilla" ]; then
    run tab3 "$name" python -u test_avalanche_lib.py --benchmark endless --scenario "$scen" --algorithm replay $semseg
  elif [ "$m" = "recl" ] || [ "$m" = "ours" ]; then
    run tab3 "$name" python -u main.py --benchmark endless --scenario "$scen" --algorithm replay --global_scheduler_mode "${GM[$m]}" --training_bs 16 --eval_bs 16 $semseg --enable_double_buffer
  else
    run tab3 "$name" python -u main.py --benchmark endless --scenario "$scen" --algorithm replay --global_scheduler_mode "${GM[$m]}" --training_bs 16 --eval_bs 16 $semseg
  fi
}
for task in ilc ss; do
  for scen in Classes Illumination Weather; do
    for m in vanilla da ekya recl fp ours; do endless_run "$task" "$scen" "$m"; done
  done
done
sr_run(){ # scenario(ic|il) method
  local scen="$1" m="$2"; local name="softrobot_${m}_${scen}"
  if [ "$m" = "vanilla" ]; then
    run tab3 "$name" python -u test_avalanche_lib.py --benchmark soft_robot --dataset_root "$DSR" --scenario_soft_robot "$scen" --algorithm replay
  elif [ "$m" = "recl" ] || [ "$m" = "ours" ]; then
    run tab3 "$name" python -u main.py --benchmark soft_robot --dataset_root "$DSR" --scenario_soft_robot "$scen" --algorithm replay --global_scheduler_mode "${GM[$m]}" --training_bs 16 --eval_bs 16 --enable_double_buffer
  else
    run tab3 "$name" python -u main.py --benchmark soft_robot --dataset_root "$DSR" --scenario_soft_robot "$scen" --algorithm replay --global_scheduler_mode "${GM[$m]}" --training_bs 16 --eval_bs 16
  fi
}
for scen in ic il; do
  for m in vanilla da ekya recl fp ours; do sr_run "$scen" "$m"; done
done
echo "STAGE tab3 DONE $(date -Is)" >> "$MASTER"; touch "$LD/tab3.done"

echo "AE ALL STAGES DONE $(date -Is)" >> "$MASTER"; touch "$LD/ae_all.done"
