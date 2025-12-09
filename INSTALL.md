#!/usr/bin/env fish
# ============================================================
# FastGS 环境配置脚本 (Fish Shell)
# 基于已有的3DGS环境配置，保持 Python 3.10.12, PyTorch 2.4.1, CUDA 12.1
# FastGS: Training 3D Gaussian Splatting in 100 Seconds
# ============================================================

# 环境名称
set ENV_NAME "fastgs"

echo "=========================================="
echo "开始配置 FastGS 环境"
echo "=========================================="

# 创建conda环境
conda create -y -n $ENV_NAME python=3.10.12
conda activate $ENV_NAME

# ============================================================
# PyTorch + CUDA 12.1
# ============================================================
echo "安装 PyTorch + CUDA 12.1..."
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install cuda-toolkit -c nvidia/label/cuda-12.1.0 --yes

# 解决 undefined symbol: iJIT_NotifyEvent
conda install mkl==2023.1.0 mkl-include -c conda-forge --yes

# 解决 cannot find -lcudart
conda install cuda-cudart=12.1.55 -c nvidia/label/cuda-12.1.0 --yes

# ============================================================
# 基础依赖 (来自你的3DGS配置 + FastGS需求)
# ============================================================
echo "安装基础依赖..."
pip install iopath
pip install plyfile==1.1 \
    tqdm \
    matplotlib==3.5.3 \
    opencv-python==4.10.0.84 \
    pillow==11.0.0 \
    lpips==0.1.4 \
    websockets

# ============================================================
# FastGS 额外依赖
# ============================================================
echo "安装 FastGS 额外依赖..."
pip install scipy \
    tensorboard \
    ninja

# 固定 numpy 版本
pip install numpy==1.24.0


# ============================================================
# 安装 submodules (diff-gaussian-rasterization)
# ============================================================
echo "=========================================="
echo "安装 submodules..."
echo "=========================================="

# 设置 CUDA_HOME 环境变量
set -x CUDA_HOME $CONDA_PREFIX

pip install --no-build-isolation submodules/diff-gaussian-rasterization_fastgs/ \
    submodules/fused-ssim/ \
    submodules/simple-knn/
