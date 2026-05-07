#!/usr/bin/env bash
# Install Dockerfile.stable.vllm dependencies into conda env cu129py312
# Run: bash install_cu129py312.sh 2>&1 | tee install_cu129py312.log

set -euo pipefail

ENV_NAME="cu129py312"
ENV_PATH="/home/jovyan/.mlspace/envs/${ENV_NAME}"

echo "========================================"
echo " Step 1: Create conda env (Python 3.12)"
echo "========================================"
conda create -n "${ENV_NAME}" python=3.12 -y

echo "========================================"
echo " Step 2: Install CUDA 12.9 toolkit"
echo "========================================"
conda install -n "${ENV_NAME}" -c nvidia cuda-toolkit=12.9 -y || \
  conda install -n "${ENV_NAME}" -c nvidia cuda-toolkit=12.9.1 -y || \
  echo "WARNING: cuda-toolkit=12.9 not found in conda, falling back to pip-based CUDA. nvcc may not be available."

echo "========================================"
echo " Step 3: Install PyTorch 2.10.0 (cu129)"
echo "========================================"
conda run -n "${ENV_NAME}" pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129

echo "========================================"
echo " Step 4: Install build tools"
echo "========================================"
conda run -n "${ENV_NAME}" pip install pybind11 wheel ninja

echo "========================================"
echo " Step 5: Install cuDNN (via pip)"
echo "========================================"
conda run -n "${ENV_NAME}" pip install nvidia-cudnn-cu12==9.16.0.29 || \
  conda run -n "${ENV_NAME}" pip install nvidia-cudnn-cu12

echo "========================================"
echo " Step 6: Install nvidia-mathdx"
echo "========================================"
conda run -n "${ENV_NAME}" pip install nvidia-mathdx

echo "========================================"
echo " Step 7: Install APEX"
echo "========================================"
conda run -n "${ENV_NAME}" bash -c "
  MAX_JOBS=16 pip install -v \
    --disable-pip-version-check \
    --no-build-isolation \
    --config-settings '--build-option=--cpp_ext' \
    --config-settings '--build-option=--cuda_ext' \
    git+https://github.com/NVIDIA/apex.git
"

echo "========================================"
echo " Step 7b: Install NCCL (headers needed by TransformerEngine)"
echo "========================================"
conda install -n "${ENV_NAME}" -c nvidia nccl -y || \
  conda install -n "${ENV_NAME}" -c conda-forge nccl -y

# Step 8: TransformerEngine v2.12 — OPTIONAL (FP8 training only, not needed for vllm inference)
# Uncomment below if you need FP8 training with Megatron-LM / verl:
#
# SP="${ENV_PATH}/lib/python3.12/site-packages"
# conda run -n "${ENV_NAME}" bash -c "
#   export NVTE_FRAMEWORK=pytorch
#   export CPATH=\"${ENV_PATH}/include:${SP}/nvidia/nvtx/include:${SP}/nvidia/nccl/include:${SP}/nvidia/cudnn/include:${SP}/nvidia/cuda_runtime/include:${SP}/nvidia/cuda_nvrtc/include:\${CPATH:-}\"
#   export LIBRARY_PATH=\"${ENV_PATH}/lib:${SP}/nvidia/nccl/lib:${SP}/nvidia/cudnn/lib:\${LIBRARY_PATH:-}\"
#   MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=4 \
#   pip3 install --resume-retries 999 --no-build-isolation \
#     git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.12
# "
echo "Step 8: TransformerEngine SKIPPED (uncomment in script if FP8 training is needed)"

echo "========================================"
echo " Step 9: Install misc Python packages"
echo "========================================"
conda run -n "${ENV_NAME}" pip install \
  codetiming mathruler pylatexenc qwen_vl_utils cachetools pytest-asyncio

echo "========================================"
echo " Step 10: Install flash-attn 2.8.3"
echo "========================================"
conda run -n "${ENV_NAME}" bash -c "
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  MAX_JOBS=16 pip install --no-build-isolation flash_attn==2.8.3
"

echo "========================================"
echo " Step 11: Install vllm 0.18.0"
echo "========================================"
conda run -n "${ENV_NAME}" pip install vllm==0.18.0

echo "========================================"
echo " Step 12: Install trl 0.27.0 (no-deps)"
echo "========================================"
conda run -n "${ENV_NAME}" pip3 install --no-deps trl==0.27.0

echo "========================================"
echo " Step 13: Install nvtx, matplotlib, liger_kernel"
echo "========================================"
conda run -n "${ENV_NAME}" pip3 install nvtx matplotlib liger_kernel

#echo "========================================"
#echo " Step 14: Install mbridge"
#echo "========================================"
#conda run -n "${ENV_NAME}" pip install -U \
#  git+https://github.com/ISEEKYAN/mbridge.git@641a5a0

echo "Step 14: mbridge SKIPPED (bridge library between Megatron-LM and vllm (for weight loading))"


#echo "========================================"
#echo " Step 15: Install Megatron-LM (no-deps)"
#echo "========================================"
#conda run -n "${ENV_NAME}" pip install --no-deps \
#  git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0

echo "Step 15: Megatron-LM SKIPPED (large-scale distributed training framework. Only needed for training transformer models with tensor/pipeline parallelism at scale)"

echo "========================================"
echo " Step 16: Install transformers 5.3.0"
echo "========================================"
conda run -n "${ENV_NAME}" pip install transformers==5.3.0

echo "========================================"
echo " Step 17: Install verl 0.7.1 then uninstall"
echo "========================================"
conda run -n "${ENV_NAME}" pip install git+https://github.com/verl-project/verl.git@v0.7.1

echo ""
echo "========================================"
echo " NOTE: DeepEP (Step 11 in Dockerfile) was SKIPPED."
echo " It requires gdrcopy kernel module (needs root/system install)."
echo " To install manually if gdrcopy is available:"
echo "   git clone -b hybrid-ep https://github.com/deepseek-ai/DeepEP.git"
echo "   cd DeepEP && python setup.py install"
echo "========================================"
echo ""
echo "========================================"
echo " NOTE: Nsight Systems (Dockerfile step) was SKIPPED."
echo " It requires apt/root. If needed, install separately."
echo "========================================"
echo ""
echo "DONE. Activate with: conda activate ${ENV_NAME}"
