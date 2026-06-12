#!/usr/bin/env bash
set -euo pipefail

: "${NEUROLENS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${DATA_ROOT:?Set DATA_ROOT to the data directory.}"
: "${OUT_ROOT:=${NEUROLENS_ROOT}/outputs}"
: "${SHIKRA_REPO:?Set SHIKRA_REPO to the local Shikra source repository.}"
: "${SHIKRA_PATH:?Set SHIKRA_PATH to the Shikra model directory.}"
: "${CLIP_L_PATH:?Set CLIP_L_PATH to the CLIP ViT-L/14 model path or HF id.}"
: "${BIGG_CKPT:?Set BIGG_CKPT to the OpenCLIP ViT-bigG checkpoint path.}"

python "${NEUROLENS_ROOT}/src/train_bigg_to_shikra_vlm_ce.py" \
  --mode=residual \
  --hdf5_path="${DATA_ROOT}/coco_images_224_float16.hdf5" \
  --subj_indices_path="${DATA_ROOT}/COCO_73k_subj_indices.hdf5" \
  --annots_path="${DATA_ROOT}/subj01_annots.npy" \
  --shared1000_path="${DATA_ROOT}/shared1000.npy" \
  --init_adapter="${OUT_ROOT}/bigG_to_shikra_adapter/last.pth" \
  --out_dir="${OUT_ROOT}/restricted_residual_bridge_subj01" \
  --shikra_path="${SHIKRA_PATH}" \
  --clip_l_path="${CLIP_L_PATH}" \
  --bigg_ckpt="${BIGG_CKPT}" \
  --bigg_cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}" \
  --question="Describe the image briefly." \
  --batch_size=4 \
  --max_steps=3000 \
  --eval_every=200 \
  --save_every=500 \
  --lr=2e-5 \
  --weight_decay=1e-2 \
  --residual_max_gate=0.05 \
  --device=cuda
