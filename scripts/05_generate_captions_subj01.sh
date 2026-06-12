#!/usr/bin/env bash
set -euo pipefail

: "${NEUROLENS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${OUT_ROOT:=${NEUROLENS_ROOT}/outputs}"
: "${EVAL_ROOT:=${OUT_ROOT}/evals}"
: "${SHIKRA_REPO:?Set SHIKRA_REPO to the local Shikra source repository.}"
: "${SHIKRA_PATH:?Set SHIKRA_PATH to the Shikra model directory.}"
: "${BRIDGE_CKPT:=${OUT_ROOT}/restricted_residual_bridge_subj01/last.pth}"
: "${BASE_ADAPTER_CKPT:=${OUT_ROOT}/bigG_to_shikra_adapter/last.pth}"

python "${NEUROLENS_ROOT}/src/experimentA_generate_calibrated_shikra_prompts.py" \
  --eval_root="${EVAL_ROOT}" \
  --base_name="base_official_prior" \
  --variant_name="neurolens_subj01" \
  --calibrator_ckpt="${BRIDGE_CKPT}" \
  --bridge_adapter_ckpt="${BASE_ADAPTER_CKPT}" \
  --shikra_path="${SHIKRA_PATH}" \
  --question="Describe the image briefly." \
  --batch_size=1 \
  --max_new_tokens=48 \
  --temperature=0.0 \
  --device=cuda
