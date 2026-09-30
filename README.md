# AdaptOCL
enabling concurrent on-device online continual learning inference and retraining using real-time streaming data

For reproducing the paper results (artifact evaluation), see [ArtifactEvaluation.md](ArtifactEvaluation.md).

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

From here on, run all commands from the repository root.

```bash
# Download the benchmark datasets
sh download_benchmark.sh
```

# Specific Change for EndlessCL-Sim semantic segmentation

```bash
export AVALANCHE_ROOT=/path/to/avalanche   # the avalanche clone installed with `pip install -e` above
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

# Useful Links
[Avalanche: an End-to-End Library for Continual Learning](https://avalanche.continualai.org/)

[NVIDIA: PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

[PyTorch for Jetson](https://zexinli.prof/post/160b.html)

[NVIDIA: Torchvision for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

[Torchvision for Jetson](https://zexinli.prof/post/5c1d.html)

[Deepspeed end2end FLOPs profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)
