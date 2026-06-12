#!/usr/bin/env bash
set -euo pipefail

: "${NEUROLENS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${DATA_ROOT:?Set DATA_ROOT to the MindEye/NSD data directory.}"
: "${OUT_ROOT:=${NEUROLENS_ROOT}/outputs}"
: "${GLOBAL_BATCH_SIZE:=16}"
: "${CUDA_VISIBLE_DEVICES:=0}"

mkdir -p "${OUT_ROOT}/logs"

cd "${NEUROLENS_ROOT}/src"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  PYTHONUNBUFFERED=1 \
  python train_visual_prior.py \
    --data_path="${DATA_ROOT}" \
    --cache_dir="${HF_HOME:-${DATA_ROOT}}" \
    --model_name="neurolens_subj01_visual_prior" \
    --no-multi_subject \
    --subj=1 \
    --batch_size="${GLOBAL_BATCH_SIZE}" \
    --num_sessions=40 \
    --hidden_dim=4096 \
    --clip_scale=1 \
    --blurry_recon \
    --blur_scale=.5 \
    --use_prior \
    --prior_scale=30 \
    --n_blocks=4 \
    --max_lr=3e-4 \
    --mixup_pct=.33 \
    --num_epochs=80 \
    --no-use_image_aug \
    --ckpt_interval=10 \
    --no-wandb_log \
    --no-new_test \
    --skip_nonfinite_loss
