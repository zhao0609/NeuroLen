#!/usr/bin/env bash
set -euo pipefail

: "${NEUROLENS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${COCO_ROOT:?Set COCO_ROOT to the COCO directory containing train2017/ and val2017/.}"
: "${DATA_ROOT:?Set DATA_ROOT to the output data directory.}"

python "${NEUROLENS_ROOT}/src/build_coco_trainval_hdf5.py" \
  --coco_root="${COCO_ROOT}" \
  --out="${DATA_ROOT}/coco_images_224_float16.hdf5" \
  --size=224 \
  --chunk_size=256 \
  --num_workers=16
