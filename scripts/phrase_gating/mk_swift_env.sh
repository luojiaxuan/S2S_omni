#!/usr/bin/env bash
# ms-swift env for the phrase-gated thinker SFT. Everything lives on the gemini NFS: aries'
# root filesystem is full, so conda, pip and their temporaries must all stay off it.
# conda is driven through its own interpreter because the install was relocated and its
# console script still carries the original shebang.
set -x
W=/mnt/gemini/home/jiaxuanluo/phrase_sft_20260904
M=/mnt/gemini/home/jiaxuanluo/miniconda3
export TMPDIR="$W/tmp" PIP_CACHE_DIR="$W/pipcache" XDG_CACHE_HOME="$W/pipcache" \
       CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 CONDA_PKGS_DIRS="$W/condapkgs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
rm -rf "$W/env"
"$M/bin/python" "$M/bin/conda" create -y -p "$W/env" python=3.11 || exit 1
V="$W/env/bin"
"$V/python" -V || exit 1
"$V/python" -m pip install --no-input -U pip wheel || exit 1
"$V/python" -m pip install --no-input "ms-swift==3.9.1" || exit 1
"$V/python" -m pip install --no-input "transformers>=4.57" accelerate deepspeed peft \
    librosa soundfile av qwen-omni-utils || exit 1
"$V/python" - <<'PY'
import torch, transformers, swift
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
print("transformers", transformers.__version__, "| swift", swift.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("qwen3_omni_moe in transformers:", "qwen3_omni_moe" in CONFIG_MAPPING_NAMES)
PY
echo SWIFT_ENV_DONE
