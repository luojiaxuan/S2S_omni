# Every cache and temporary path must stay on the gemini NFS: aries' root filesystem is
# full, and each library that falls back to $HOME dies with ENOSPC mid-run.
W=/mnt/gemini/home/jiaxuanluo/phrase_sft_20260904
export TMPDIR="$W/tmp"
export XDG_CACHE_HOME="$W/cache"
export HF_HOME="$W/cache/huggingface"
export HF_DATASETS_CACHE="$W/cache/huggingface/datasets"
export MODELSCOPE_CACHE="$W/cache/modelscope"
export TRITON_CACHE_DIR="$W/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$W/cache/inductor"
export PIP_CACHE_DIR="$W/pipcache"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
