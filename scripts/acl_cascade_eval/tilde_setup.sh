#!/usr/bin/env bash
# note (luojiaxuan): Tilde 上 v4 scheduled-sampling 验证的一次性环境搭建。
# 版本对齐 hyper00 已验证的栈（torch 2.11.0+cu130 / transformers 5.6.0），
# 因为 MOSS-TTS 的 causal-mask 补丁就是针对 transformers 5.6 签名写的。
set -euo pipefail

ROOT=/home/guests/zhen/s2s_omni_v4
ENV_PREFIX=/home/guests/zhen/miniconda3/envs/s2s_v4
CONDA=/home/guests/zhen/miniconda3/bin/conda
MOSS_BASE_COMMIT=58b20a0d5fcc6766658d50967a90a9d890009a46

mkdir -p "$ROOT"/{code,data,runs,logs}
cd "$ROOT/code"

echo "[1/5] clone S2S_omni"
if [ ! -d S2S_omni ]; then
  git clone -q --depth 50 https://github.com/luojiaxuan/S2S_omni.git
fi
git -C S2S_omni fetch -q origin main && git -C S2S_omni checkout -q main && git -C S2S_omni pull -q
echo "  S2S_omni @ $(git -C S2S_omni rev-parse --short HEAD)"

echo "[2/5] clone MOSS-TTS + apply patch"
if [ ! -d MOSS-TTS ]; then
  git clone -q https://github.com/OpenMOSS/MOSS-TTS.git
fi
cd MOSS-TTS
git checkout -q "$MOSS_BASE_COMMIT"
git checkout -q -- . 2>/dev/null || true
git apply "$ROOT/code/S2S_omni/third_party/moss_tts/moss_tts_s2s_omni.patch"
echo "  MOSS-TTS @ $(git rev-parse --short HEAD) + patch applied"
grep -q "context_only" moss_tts_realtime/finetuning/dataset.py || { echo "PATCH_VERIFY_FAILED"; exit 3; }
cd "$ROOT/code"

echo "[3/5] conda env"
if [ ! -d "$ENV_PREFIX" ]; then
  "$CONDA" create -y -q -p "$ENV_PREFIX" python=3.12 >/dev/null
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -m pip install -q --upgrade pip

echo "[4/5] torch stack (cu130, matches hyper00)"
"$PY" -m pip install -q torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install -q transformers==5.6.0 accelerate==1.14.0 huggingface_hub soundfile \
  safetensors orjson tqdm PyYAML einops scipy librosa tiktoken psutil packaging ninja wandb

echo "[5/5] verify"
"$PY" -c "import torch, transformers, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
echo "SETUP_DONE"
