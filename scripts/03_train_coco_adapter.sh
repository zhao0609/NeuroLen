#!/usr/bin/env bash
set -euo pipefail

: "${NEUROLENS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${DATA_ROOT:?Set DATA_ROOT to the data directory.}"
: "${OUT_ROOT:=${NEUROLENS_ROOT}/outputs}"
: "${CLIP_L_PATH:?Set CLIP_L_PATH to the CLIP ViT-L/14 model path or HF id.}"
: "${BIGG_CKPT:?Set BIGG_CKPT to the OpenCLIP ViT-bigG checkpoint path.}"
: "${CUDA_VISIBLE_DEVICES:=0}"

python "${NEUROLENS_ROOT}/src/train_bigg_to_shikra_adapter.py" \
  --hdf5_path="${DATA_ROOT}/coco_images_224_float16.hdf5" \
  --out_dir="${OUT_ROOT}/bigG_to_shikra_adapter" \
  --clip_l_path="${CLIP_L_PATH}" \
  --bigg_ckpt="${BIGG_CKPT}" \
  --bigg_cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}" \
  --batch_size=128 \
  --num_epochs=8 \
  --steps_per_epoch=100 \
  --max_lr=3e-4 \
  --weight_decay=1e-2 \
  --device=cuda
