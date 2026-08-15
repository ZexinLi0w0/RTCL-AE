# AdaptOCL
enabling concurrent on-device online continual learning inference and retraining using real-time streaming data

# Prerequisites

- Python                  3.10.12 (JetPack 6.2, L4T 36.4.3)
- avalanche-lib           0.6.0a
- PyTorch                 2.5.0
- Torchvision             0.17.0
- CUDA			  12.6.68

# Installation

Install the avalanche from source code v0.6.0a to Jetson aarch64 platforms, referring to [Avalanche: an End-to-End Library for Continual Learning](https://avalanche.continualai.org/)

Tested version: 0.6.0a - SHA: eb075be393e1f458b2c352514ff6c17b5a2c0f4e

```bash
git clone https://github.com/ContinualAI/avalanche.git
cd avalanche
pip install -e ".[dev]"
```

Remove PyTorch and Torchvision if they are installed, and then install PyTorch and Torchvision from source code.

```bash
pip uninstall torch torchvision
```

Install for GPU support of PyTorch: need to build from source on Jetson, referring to [NVIDIA: PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

Referring to my blog post [PyTorch for Jetson](https://zexinli.prof/post/160b.html)

Install for GPU support of Torchvision: need to build from source on Jetson, referring to [NVIDIA: Torchvision for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

Referring to my blog post [Torchvision for Jetson](https://zexinli.prof/post/5c1d.html)

```bash
# apt install some necessary packages
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-dev python3-pip libopenblas-base libopenmpi-dev libomp-dev libopenblas-dev libopenmpi-dev libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

# Download a newer version of cmake for building PyTorch from source
wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-aarch64.tar.gz
tar -xvzf cmake-3.28.3-linux-aarch64.tar.gz
export PATH=$PATH_TO_CMAKE$/cmake-3.28.3-linux-aarch64/bin:$PATH

# avalanche-lib requires this pytorch.distributed; so cannot use Jetson pre-built wheels
# Build GPU-enabled PyTorch from source for v2.5.0
git clone --recursive --branch v2.5.0 http://github.com/pytorch/pytorch
cd pytorch
pip install -r requirements.txt
pip install pip testresources setuptools scikit-build ninja
export USE_NCCL=0
export USE_DISTRIBUTED=1 # required                   
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export TORCH_CUDA_ARCH_LIST="8.7"   # "8.7" for Ampere arch for Orin
export PYTORCH_BUILD_VERSION=2.5.0  # without the leading 'v', e.g. 1.3.0 for PyTorch v1.3.0
export PYTORCH_BUILD_NUMBER=1
export USE_PRIORITIZED_TEXT_FOR_LD=1
export MAX_JOBS=4 # limit maximal parallel job numbers to avoid OOM
python setup.py bdist_wheel
python setup.py install
python setup.py develop

# Build GPU-enabled Torchvision from source for v0.16.0
git clone --branch v0.17.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.17.0  # where 0.17.0 is the torchvision version; for instance: 0.17.0 refers to PyTorch v2.5.0
python setup.py install --user

# install jtop by pip
pip install jetson-stats
```

Also needs to manually install the following packages since I bypassed cvxopt in the installation of avalanche-lib to avoid error. Refer to [#1610](https://github.com/ContinualAI/avalanche/issues/1610).

```bash
sudo apt-get install libsuitesparse-dev libblas-dev liblapack-dev gfortran
pip install osqp ecos scs qpsolvers quadprog tinyimagenet timm
conda install cvxopt
```

# Download Benchmark Datasets

```bash
# Download the benchmark datasets
sh download_benchmark.sh
```

# Specific Change for EndlessCL-Sim semantic segmentation

```bash
# copy the specific change for EndlessCL-Sim semantic segmentation to the avalanche-lib
# warning! this will overwrite the original files. Please backup the original files before running the following commands.
cp modified/avalanche/evaluation/metrics/*.py $AVALANCHE_ROOT/avalanche/evaluation/metrics/
cp modified/avalanche/benchmarks/classic/*.py $AVALANCHE_ROOT/avalanche/benchmarks/classic/
cp modified/avalanche/benchmarks/datasets/endless_cl_sim/*.py $AVALANCHE_ROOT/avalanche/benchmarks/datasets/endless_cl_sim/
```

# Download soft robot dataset

```bash
gdown "https://drive.google.com/uc?id=1fCZZF0BThM3Jk-D_ijoWlyFammKZVd8l&confirm=t&uuid=f7a082fc-623a-47b3-b0c4-b07dab0997dc"
# if this does not work, try gdown --fuzzy
unzip soft_robot_raw_images.zip
```

# Smoke Test

```bash
# Run the test script: modify from the official Github repo of avalanche-lib
python test_all_features.py
```

# Running Experiments (Artifact Evaluation)

There are two entry points:

- `test_avalanche_lib.py` — the **Avalanche (sequential)** baseline, running the unmodified sequential OCL pipeline.
- `main.py` — all **concurrent** methods; the method is selected by `--global_scheduler_mode`:

| Method (paper name) | Entry point | `--global_scheduler_mode` |
|---|---|---|
| Avalanche (sequential) | `test_avalanche_lib.py` | — |
| Default Alternation (DA) | `main.py` | `default` |
| Ekya | `main.py` | `ekya` |
| RECL | `main.py` | `recl_sched` |
| AOCL_basic (fully parallel) | `main.py` | `fully_parallel` |
| AOCL (ours) | `main.py` | `adaptocl` |

### Metrics: streaming accuracy (SA) and training throughput (QPS)

Each run prints the **streaming accuracy (SA)** and the overall evaluation latency to stdout (lines `Overall Streaming Accuracy: ...` and `Overall Evaluation Latency: ...`; the sequential Avalanche baseline reports SA as `Top1_Acc_Stream/eval_phase/test_stream/Task000`). The paper's throughput metric, **training throughput (QPS, queries per second)**, is *not* emitted by the runtime; it is computed from the run's end-to-end wall-clock time as

```
QPS = N_train / T_wall
```

where `N_train` is the (method-independent) number of training samples in the benchmark and `T_wall` is the total wall-clock time of the run. Because `N_train` is fixed per benchmark, the QPS ratio between two methods equals the inverse of their wall-clock ratio, so the relative throughput ordering — the claim under evaluation — follows directly from wall-clock time. The fixed training-set sizes (from the benchmark constructors in `src/workers/train_worker.py`) are:

| Benchmark | `N_train` (samples) | Train experiences |
|---|---|---|
| SplitCIFAR10 | 50,000 | 10 |
| SplitCIFAR100 † | 5,000 | 10 |
| CORe50-NI | 119,894 | 8 |
| CORe50-NC | 119,894 | 9 |
| CORe50-NIC | 119,894 | 79 |

† SplitCIFAR100 is constructed with `fixed_class_order=range(10)`, i.e. only the first 10 classes of CIFAR-100 are used (10 one-class experiences; 5,000 train / 1,000 test samples).

Runs should be executed on both Jetson Orin Nano and AGX Orin to reproduce the per-platform numbers. `run_scripts.sh` is a convenience wrapper that runs a list of scripts and collects logs and generated `.pth` files under `./output/test_results/`.

## Fig. 1: GPU-utilization traces

Fig. 1 traces GPU utilization over time for four modes on CORe50-NC with the ER algorithm (`--algorithm replay`) and ResNet-20: Avalanche (sequential), DA, AOCL_basic, and AOCL (with double buffering). Each run is sampled at 1 Hz with `tegrastats --interval 1000 --logfile ...` running alongside the workload:

```bash
# Start the sampler for each run (stop it when the run exits):
sudo tegrastats --interval 1000 --logfile ae_logs/fig1/<mode>_tegrastats.log &

# --- Avalanche (sequential) ---
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay"
# --- Default Alternation (DA) ---
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16
# --- AOCL_basic ---
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16
# --- AOCL (ours) ---
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
```

`bash ae_logs/run_fig123.sh` is the driver that runs all Fig. 1–3 stages with skip-on-done; the tegrastats logs land in `ae_logs/fig1/`.

## Fig. 2: Alternation method and interval

Fig. 2(a) compares four alternation policies — DA (`default`), AOCL_basic (`fully_parallel`), TA (`adaptive_time --adaptive_priority_percent 0.5`), and AA (`adaptive_accuracy --adaptive_accuracy_threshold 0.4`) — on SplitCIFAR10, CORe50-NC, and CORe50-NIC, all at `--timeslice 0.1`. Representative command (vary `--global_scheduler_mode` and `--benchmark`):

```bash
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "adaptive_time" --adaptive_priority_percent 0.5 --training_bs 16 --eval_bs 16 --timeslice 0.1
```

Fig. 2(b) sweeps the alternation interval `--timeslice` over {0.001, 0.01, 0.1, 1.0} seconds for DA and AOCL_basic on CORe50-NIC, plus an Avalanche sequential run as the vanilla reference. Representative command (vary `--timeslice` and `--global_scheduler_mode`):

```bash
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16 --timeslice 0.01
```

## Fig. 3: Batch-size sweep (smoke test)

Fig. 3 sweeps the training batch size over {8, 16, 32, 64, 128, 256} for DA and AOCL_basic on SplitCIFAR10, CORe50-NC, and CORe50-NIC (36 configurations). **Per AE decision, the repository ships a 2-minute smoke test per configuration** — each command runs under `timeout -s KILL 120`, and a configuration that survives the full 2 minutes counts as PASS — because the full 36-run sweep takes many hours on Jetson. All 36 configurations PASS. An evaluator who wants the full numbers can run any cell to completion with:

```bash
python -u main.py --benchmark split_cifar10 --algorithm replay --global_scheduler_mode default --training_bs 64 --eval_bs 16 --timeslice 0.1
```

substituting `--benchmark` (`split_cifar10` / `core50 --scenario_core50 nc` / `core50 --scenario_core50 nic`), `--global_scheduler_mode` (`default` / `fully_parallel`), and `--training_bs`.

## Fig. 5: Overall effectiveness across deep-learning models

The large-model support for the Fig. 5 scaling study is available directly on `main` (these results were originally produced on the `large_model_resnet50` branch, since merged). `--model` now additionally accepts `resnet50, resnet101, resnet152, resnet200, efficientnet_b0`, implemented via `pytorchcv` in `src/models/model_init.py`. For SplitCIFAR-10/100 the input pipeline upsamples the native 32×32 images to 224×224 before the backbone (see `create_benchmark` in `main.py` and `src/workers/train_worker.py`).

Prerequisite (in addition to the base install):

```bash
pip install pytorchcv
```

Reproduce Fig. 5 by sweeping the backbones on SplitCIFAR10 (repeat per method by changing `--global_scheduler_mode`, as in Fig. 6):

```bash
for M in resnet50 resnet101 resnet152 resnet200; do
  # AOCL (ours):
  python main.py --benchmark "split_cifar10" --algorithm "replay" --model "$M" --global_scheduler_mode "adaptocl" --training_bs 16 --eval_bs 16 --enable_double_buffer
  # AOCL_basic / DA / Ekya: swap --global_scheduler_mode to fully_parallel / default / ekya
done
```

Notes:
- The large-model backbones run through `main.py` (routed to the ImageNet path in `model_init.py`). The `test_avalanche_lib.py` sequential baseline was **not** modified for the large models and still builds CIFAR-style ResNets by depth, so it does not support these `--model` values; use `main.py` scheduler modes for the cross-model comparison.
- Validated on Jetson AGX Orin (pytorchcv 0.0.73): `resnet50`, `resnet101`, `resnet152`, `resnet200` all build and forward-pass; `resnet50` was smoke-tested through the full AOCL `main.py` pipeline on SplitCIFAR10 (224×224 resize) and trained without error.
- `efficientnet_b0` is listed in `--model` but is **not functional** on this pytorchcv build: its `output` head is a `Sequential` ending in `Linear`, whereas the head-rebind logic in `src/models/model_init.py` expects a `Conv2d` tail, so it raises `NotImplementedError`. Use the ResNet backbones for Fig. 5.
- The paper's Fig. 5 uses ResNet-50, ResNet-101, MobileNetV1, and ViT-Tiny; the reproduced sweep below covers exactly these four backbones (`resnet50, resnet101, mobilenetv1, vit_tiny`). ResNet-152/-200 also exist in the code as deeper scaling points but are not part of the paper's Fig. 5.
- 224×224 inputs make these backbones far heavier than the CIFAR ResNets; expect long per-run wall-clock on Jetson.

## Fig. 6: Overall effectiveness (ER algorithm, ResNet-20, five benchmarks)

Fig. 6 evaluates all six methods with the ER algorithm (`--algorithm replay`) and the default ResNet-20 model on five benchmarks: SplitCIFAR10, SplitCIFAR100, CORe50-NI, CORe50-NC, and CORe50-NIC. Repeat the commands below on each Jetson platform (Nano and Orin).

```bash
# --- Avalanche (sequential baseline): uses test_avalanche_lib.py ---
python test_avalanche_lib.py --benchmark "split_cifar10" --algorithm "replay"
python test_avalanche_lib.py --benchmark "split_cifar100" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay"
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay"

# --- Default Alternation (DA): fixed round-robin alternation, batch size 16 ---
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16

# --- Ekya (adapted to embedded OCL) ---
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "ekya" --training_bs 16 --eval_bs 16
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "ekya" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "ekya" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "ekya" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "ekya" --training_bs 16 --eval_bs 16

# --- RECL (adapted to embedded OCL): needs double buffering ---
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer

# --- AOCL_basic: fully concurrent training + evaluation, static configuration ---
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "fully_parallel" --training_bs 16 --eval_bs 16

# --- AOCL (ours): UAM-driven adaptive scheduler, needs double buffering ---
python main.py --benchmark "split_cifar10" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "split_cifar100" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "ni" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nic" --algorithm "replay" --global_scheduler_mode "adaptocl" --eval_bs 16 --enable_double_buffer
```

The full per-method benchmark sweeps (including the EndlessCL-Sim scenarios) are also available as batch scripts under `scripts/`:

```bash
# One folder per method: baseline_vanilla (Avalanche), baseline_da, baseline_ekya,
# baseline_recl, baseline_fp (AOCL_basic), baseline_ours (AOCL).
sh scripts/baseline_vanilla/vary_benchmark.sh   # Avalanche across all benchmarks
sh scripts/baseline_ours/vary_benchmark.sh      # AOCL across all benchmarks
sh scripts/baseline_ours/vary_algorithm.sh      # AOCL across OCL algorithms (ER/GSS/GEM/AGEM)
```

Notes:

- CORe50 uses the mini (32x32) variant (`mini=True`) due to embedded memory limitations.
- The model defaults to `--model resnet20`; the artifact's model factory currently provides `simple_mlp, resnet20, resnet56, resnet110, resnet1001` (see `src/config.py`). The Fig. 5 backbones (ResNet-50/101, MobileNetV1, ViT-Tiny) are **not** included in this release — see the Fig. 5 note below.
- Batch sizes for DA/Ekya/AOCL_basic are fixed at 16; RECL and AOCL manage the training batch size internally, so only `--eval_bs` is passed together with `--enable_double_buffer`.

## Fig. 7: Overall effectiveness across OCL algorithms (CORe50-NC)

Fig. 7 fixes the benchmark to CORe50-NC and the model to ResNet-20, and sweeps the four OCL algorithms (ER = `replay`, GSS = `gss_greedy`, GEM = `gem`, AGEM = `agem`) across all six methods:

```bash
# Avalanche (sequential) — repeat for algorithm in replay / gss_greedy / gem / agem:
python test_avalanche_lib.py --benchmark "core50" --scenario_core50 "nc" --algorithm "<ALGO>"
# DA / Ekya / AOCL_basic (main.py); <MODE> in default|ekya|fully_parallel:
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "<ALGO>" --global_scheduler_mode "<MODE>" --training_bs 16 --eval_bs 16
# RECL and AOCL use double buffering:
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "<ALGO>" --global_scheduler_mode "recl_sched" --eval_bs 16 --enable_double_buffer
python main.py --benchmark "core50" --scenario_core50 "nc" --algorithm "<ALGO>" --global_scheduler_mode "adaptocl"   --eval_bs 16 --enable_double_buffer
```

Equivalently, `sh scripts/baseline_<method>/vary_algorithm.sh` runs all four algorithms for one method.

## Fig. 8 and Table II: Inference-latency CDF and deadline-miss rate

Fig. 8 (per-frame inference-latency CDF) and Table II (deadline-miss rate, DMR) are both derived from the per-eval-step inference latency `L_inf` that `main.py` already records during the sustained ER / ResNet-20 runs of Fig. 6. Each evaluation mini-batch logs a line

```
[Eval][Exp N][MiniBatch M] acc_so_far=..., batch_lat=<L_inf seconds>
```

so no extra experiments are required — the five concurrent methods (DA, Ekya, RECL, AOCL_basic, AOCL) are parsed straight from the Fig. 6 logs (Avalanche is excluded because it never co-runs training and inference). `L_inf` is the per-eval-step latency (`--eval_bs 16`); the DMR at deadline `d` is the fraction of steps with `L_inf > d`, reported at `d = 16 / 33 / 100 ms`. Regenerate with:

```bash
python ae_logs/parse_fig8_tab2.py   # reads fig6_logs/*.log -> fig8_latency_cdf.csv, tab2_dmr.csv
```

> **Fidelity note.** The paper collects >=10,000 *consecutive* per-frame samples per cell in a dedicated steady-state pass, whereas the Fig. 6 logs are sub-sampled (every 10th eval mini-batch) and shorter. The central distribution (p50/p90/p99) reproduces the paper's regime, but the extreme tail (p99.9) of AOCL is inflated by reconfiguration / double-buffer transients that the dedicated pass isolates; treat p99.9 as an upper bound rather than a faithful value.

## Table III: Robotic case studies

Table III(a) is the autonomous-navigation study on EndlessCL-Sim with two perception tasks — instance-level classification (ILC) and semantic segmentation (SS) — each under three scenarios: Incremental Class (`Classes`), Incremental Illumination (`Illumination`), and Weather (`Weather`). Table III(b) is the soft-robot study under scenarios incremental class (`ic`) and incremental illumination (`il`).

```bash
# (a) EndlessCL-Sim — ILC (classification) drops --semseg; SS (segmentation) adds --semseg:
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "adaptocl" --training_bs 16 --eval_bs 16 --enable_double_buffer            # ILC
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "adaptocl" --training_bs 16 --eval_bs 16 --semseg --enable_double_buffer   # SS
# (b) soft robot:
python main.py --benchmark "soft_robot" --dataset_root "/experiment/.avalanche/data/soft_robot_data_raw/" --scenario_soft_robot "ic" --algorithm "replay" --global_scheduler_mode "adaptocl" --training_bs 16 --eval_bs 16 --enable_double_buffer
```

The `scripts/endless_ss_<method>/` and `scripts/soft_robot_<method>/` folders hold the SS and soft-robot commands for every method; ILC uses the same endless commands without `--semseg`. Metrics are training throughput (TT = QPS) and streaming accuracy (SA).

## Table IV: Ablation study

Table IV ablates the AOCL scheduler on ER / ResNet-20 across the five benchmarks, comparing `AOCL_basic` (`fully_parallel`), `TA` (time-adaptive, `adaptive_time`), `AA` (accuracy-adaptive, `adaptive_accuracy`), and the full `AOCL` (`adaptocl`). `AOCL_basic` and `AOCL` reuse the Fig. 6 numbers; the two adaptive components run as:

```bash
python main.py --benchmark <B> --algorithm "replay" --global_scheduler_mode "adaptive_time"     --training_bs 16 --eval_bs 16 --adaptive_priority_percent 0.5
python main.py --benchmark <B> --algorithm "replay" --global_scheduler_mode "adaptive_accuracy" --training_bs 16 --eval_bs 16 --adaptive_accuracy_threshold 0.4
```

# Result Files

Several aggregated result tables are checked into the repository root. They are produced by running the commands above on the AGX Orin under the MAXN power mode (`nvpmodel -m 0`) and parsing the per-run logs.

## `fig1_gpu_util.csv` — Fig. 1 GPU-utilization traces

Raw 1 Hz tegrastats `GR3D_FREQ` samples for the four Fig. 1 runs. One row per sample.

| Column | Meaning |
|---|---|
| `mode` | Run mode (`vanilla` = Avalanche sequential, `da`, `fp` = AOCL_basic, `ours` = AOCL). |
| `sample_idx` | Sample index within the run (1 Hz, so also seconds since sampling start). |
| `gr3d_pct` | GPU (GR3D) utilization percent from tegrastats. |

## `fig1_gpu_util_summary.csv` — Fig. 1 per-run summary

| Column | Meaning |
|---|---|
| `mode` | Run mode as above. |
| `samples` | Number of 1 Hz samples in the trace. |
| `mean_gr3d_pct` | Mean GPU utilization over the run. |
| `run_wallclock_sec` | End-to-end wall-clock time of the run, in seconds. |

## `fig2a_alternation_methods.csv` — Fig. 2(a) alternation methods

One row per (method, benchmark) run at `--timeslice 0.1`. Columns: `method` (`da`, `fp` = AOCL_basic, `ta`, `aa`), `benchmark`, `timeslice_s`, `train_samples`, `wallclock_sec`, `qps`, `streaming_accuracy`, `return_code`, `log_file`.

## `fig2b_alternation_interval.csv` — Fig. 2(b) alternation-interval sweep

Same columns as `fig2a_alternation_methods.csv`; one row per (method, timeslice) on CORe50-NIC, plus the Avalanche sequential reference row (`method=vanilla`), whose `timeslice_s` is `-` since it has no alternation interval.

## `fig3_batch_size.csv` — Fig. 3 batch-size sweep (smoke verdicts only)

One row per configuration of the 36-cell grid. These are **smoke-test verdicts, not full-run results**: each cell ran under `timeout -s KILL 120`, so surviving the 2-minute window is a PASS (the KILL-induced `return_code=137` is expected).

| Column | Meaning |
|---|---|
| `method` | `da` or `fp` (AOCL_basic). |
| `benchmark` | Benchmark / CORe50 scenario. |
| `training_bs` | Training batch size (8/16/32/64/128/256). |
| `smoke_status` | `PASS` if the run survived the 2-minute smoke window. |
| `return_code` | Process exit code (137 = killed by the smoke-test timeout, expected). |
| `log_file` | Per-run log file name (under `ae_logs/fig3/`). |
| `note` | Caveat that this is a smoke test only, with the full command in the Fig. 3 section above. |

## `fig5_results.csv` — Fig. 5 sweep results

The reproduced Fig. 5 sweep results are checked in as `fig5_results.csv` at the repository root. One row per run.

| Column | Meaning |
|---|---|
| `method` | Paper method name (Avalanche, DA, Ekya, RECL, AOCL_basic, AOCL). |
| `benchmark` | Benchmark (SplitCIFAR10). |
| `model` | Backbone (`resnet50`, `resnet101`, `mobilenetv1`, `vit_tiny`). |
| `entry_point` | Script used (`test_avalanche_lib.py` or `main.py`). |
| `scheduler_mode` | Value passed to `--global_scheduler_mode` (`-` for the Avalanche baseline). |
| `algorithm` | OCL algorithm (`replay` = ER). |
| `train_samples` | Fixed number of training samples (50000 for `split_cifar10`). |
| `wallclock_sec` | End-to-end wall-clock time of the run, in seconds. |
| `qps` | Training throughput = `train_samples / wallclock_sec`. |
| `streaming_accuracy` | Final Overall Streaming Accuracy printed by `main.py`. |
| `return_code` | Process exit code (`0` = success). |
| `log_file` | Per-run log file name. |
| `note` | Caveats for the row. |

Driver scripts live in `ae_logs_fig5/`:
- `run_fig5.sh` — full sweep with per-run timeout, retry, and skip-on-done.
- `aggregate_fig5.py` — builds `fig5_results.csv` from `fig5_master.log`.
- `rerun_r101_db.sh` — reruns the two ResNet-101 double-buffer cells (RECL, AOCL).

Reproducibility notes:
- The Avalanche sequential baseline (`test_avalanche_lib.py`) is unsupported for these backbones because it builds CIFAR-style ResNets by depth — its rows show `return_code=2` with a note.
- **IMPORTANT:** the double-buffer runs (RECL, AOCL) on `resnet101` require a raised file-descriptor limit — run `ulimit -n 65535` first. With the default 1024, the double-buffer state sync exhausts file descriptors (`OSError` errno 24), evaluation silently fails to load model state, and no streaming accuracy is reported.
- ViT-Tiny trains from scratch, so its streaming accuracy (~0.16–0.20) is expectedly lower than the CNN backbones.

## `fig6_results.csv` — full Fig. 6 sweep (30 runs)

All six methods × five benchmarks (SplitCIFAR10, SplitCIFAR100, CORe50-NI/NC/NIC) with the ER algorithm (`--algorithm replay`) and ResNet-20. One row per run.

| Column | Meaning |
|---|---|
| `method` | Paper method name (Avalanche, DA, Ekya, RECL, AOCL_basic, AOCL). |
| `benchmark` | Benchmark / CORe50 scenario. |
| `entry_point` | Script used (`test_avalanche_lib.py` or `main.py`). |
| `scheduler_mode` | Value passed to `--global_scheduler_mode` (`-` for the sequential baseline). |
| `algorithm` | OCL algorithm (`replay` = ER). |
| `train_samples` | Fixed number of training samples `N_train` for the benchmark (see the Metrics table above). |
| `wallclock_sec` | End-to-end wall-clock time of the run, in seconds. |
| `qps` | Training throughput = `train_samples / wallclock_sec`. |
| `speedup_vs_avalanche` | QPS (equivalently wall-clock) speedup relative to the Avalanche sequential baseline on the same benchmark. |
| `streaming_accuracy` | Final overall streaming accuracy (SA). |
| `return_code` | Process exit code (`0` = success). |
| `log_file` | Per-run log file name (under `fig6_logs/`). |
| `note` | Caveats for the row (e.g. RECL degenerating to near-chance SA on SplitCIFAR100; SplitCIFAR100 using only the first 10 classes). |

## `reduced_sweep_results.csv` — smoke-test subset (7 runs)

A fast subset used to sanity-check the toolchain before launching the full sweep: three methods (Avalanche, AOCL_basic, AOCL) on two benchmarks (SplitCIFAR10, CORe50-NIC), plus one extra AOCL re-run to check run-to-run variance. Columns mirror `fig6_results.csv`, except this table predates the computed-QPS column: throughput is reported as `throughput_speedup_vs_avalanche` (the wall-clock speedup, which equals the QPS speedup because `N_train` is fixed per benchmark), and there is no absolute `qps` or `train_samples` column. The `notes` column flags the first AOCL SplitCIFAR10 run, whose final-snapshot SA was low (0.12); the re-run row shows the representative value (0.44).

## `fig7_results.csv` — Fig. 7 sweep (CORe50-NC × 4 OCL algorithms × 6 methods)

One row per (method, algorithm) run on CORe50-NC with ResNet-20. Columns match `fig6_results.csv`; `algorithm` varies over `replay` (ER), `gss_greedy` (GSS), `gem`, `agem`, and `speedup_vs_avalanche` is taken against the Avalanche run of the *same* algorithm.

## `fig8_latency_cdf.csv` — Fig. 8 inference-latency distribution

Per (method, benchmark) summary of per-eval-step inference latency `L_inf` (milliseconds) parsed from the Fig. 6 logs.

| Column | Meaning |
|---|---|
| `method` | Concurrent method (Avalanche excluded — never co-runs). |
| `benchmark` | Benchmark / CORe50 scenario. |
| `samples` | Number of `L_inf` samples parsed for the cell. |
| `p50_ms`, `p90_ms`, `p99_ms`, `p99_9_ms` | Latency percentiles (ms). |
| `mean_ms`, `max_ms` | Mean and maximum latency (ms). |

See the fidelity note above regarding p99.9.

## `tab2_dmr.csv` — Table II deadline-miss rate

Per (method, benchmark) deadline-miss rate from the same `L_inf` samples.

| Column | Meaning |
|---|---|
| `method`, `benchmark`, `samples` | As above. |
| `dmr_16ms_pct`, `dmr_33ms_pct`, `dmr_100ms_pct` | Percent of eval steps with `L_inf` exceeding 16 / 33 / 100 ms. |

## `tab3_results.csv` — Table III robotic case studies

One row per (method, task, scenario). `task` is `endless_ilc` / `endless_ss` (autonomous navigation: classification / segmentation) or `soft_robot`; `scenario` is `Classes/Illumination/Weather` (endless) or `ic/il` (soft robot). Columns: `method, task, scenario, entry_point, scheduler_mode, wallclock_sec, streaming_accuracy, return_code, log_file, note`. For `endless_ss` rows, the `note` column records how the pixel accuracy was derived (mean over experiences of the final eval cycle).

**Reproducibility notes.** `endless_ilc` (classification), `endless_ss` (semantic segmentation), and `soft_robot` (via `main.py`) all reproduce. `endless_ss` requires the `--semseg` flag **and** the per-pixel accuracy switch in the bundled Avalanche fork (`is_semseg_acc` in `avalanche/evaluation/metrics/accuracy.py`); the AE code now sets this automatically whenever `--semseg` is passed — both in the entry scripts and inside the spawned train/eval worker processes (spawn re-imports modules, so the parent's flag does not propagate on its own). For SS rows produced before the eval-worker overall-counter fix, `streaming_accuracy` is the mean per-experience pixel accuracy of the final evaluation cycle (marked in the `note` column as "pixel accuracy: mean over experiences of final eval cycle"); these reference values land within ~0.01–0.05 of the paper's Table III(a) SS accuracies. The `soft_robot` Avalanche baseline (`return_code=2`) remains unsupported because `test_avalanche_lib.py` has no `soft_robot` benchmark. See the `note` column of each row for the exact caveat.

## `tab4_ablation.csv` — Table IV ablation

One row per (config, benchmark) on ER / ResNet-20. `config` is `AOCL_basic` (fully_parallel), `TA` (adaptive_time), `AA` (adaptive_accuracy), or `AOCL` (adaptocl). Columns: `config, benchmark, scheduler_mode, train_samples, wallclock_sec, qps, streaming_accuracy, return_code, log_file, note`. The `AOCL_basic` and `AOCL` rows are copied from `fig6_results.csv`.

# Useful Links
[Avalanche: an End-to-End Library for Continual Learning](https://avalanche.continualai.org/)

[NVIDIA: PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

[PyTorch for Jetson](https://zexinli.prof/post/160b.html)

[NVIDIA: Torchvision for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

[Torchvision for Jetson](https://zexinli.prof/post/5c1d.html)

[Deepspeed end2end FLOPs profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)
