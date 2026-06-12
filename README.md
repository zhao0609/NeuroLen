# NeuroLens

This repository contains the training release for **NeuroLens: A Unified Multi-Task Model for Visual Reconstruction and Language Interaction from fMRI Signals**.

NeuroLens decodes fMRI responses into a shared brain-visual representation that supports image reconstruction, caption generation, open-ended visual-language interaction, and spatial grounding. This public package first releases the Subject 1 training pipeline.

## What Is Included

- `src/train_visual_prior.py`: MindEye-style visual prior training for NSD Subject 1.
- `src/train_bigg_to_shikra_adapter.py`: COCO image-side adapter training from reconstruction-oriented CLIP visual tokens to Shikra-compatible tokens.
- `src/train_bigg_to_shikra_vlm_ce.py`: VLM-supervised restricted residual bridge training.
- `src/recon_inference.py` and `src/enhanced_recon_inference.py`: reconstruction and semantic-guided reconstruction scripts.
- `src/experimentA_*.py`: caption generation, reconstruction enhancement, and evaluation utilities.
- `src/generate_neurolens_interaction_answers.py` and `src/evaluate_neurolens_interaction_bertscore.py`: open-ended interaction evaluation.

The release does not include NSD, COCO images, Shikra weights, SDXL weights, CLIP weights, or trained checkpoints.

## Environment

Create a Python environment with PyTorch and install the common dependencies:

```bash
pip install -r requirements.txt
```

The code also depends on the Shikra source tree. Set `SHIKRA_REPO` to the local Shikra repository path before running scripts that call Shikra.

## Data And Model Paths

Set the following environment variables according to your local machine:

```bash
export NEUROLENS_ROOT=/path/to/NeuroLens_subj01_release
export DATA_ROOT=/path/to/nsd_mindeye2_data
export COCO_ROOT=/path/to/coco
export OUT_ROOT=/path/to/neurolens_outputs

export SHIKRA_REPO=/path/to/shikra-main
export SHIKRA_PATH=/path/to/shikra-7b-v1-0708
export CLIP_L_PATH=/path/to/clip-vit-large-patch14
export BIGG_CKPT=/path/to/open_clip_pytorch_model.bin
export HF_HOME=/path/to/huggingface_cache
```

Expected NSD/MindEye-style files under `DATA_ROOT` include the Subject 1 webdataset shards and auxiliary files used by MindEye-style training. For bridge training, the examples assume:

```text
$DATA_ROOT/coco_images_224_float16.hdf5
$DATA_ROOT/COCO_73k_subj_indices.hdf5
$DATA_ROOT/subj01_annots.npy
$DATA_ROOT/shared1000.npy
```

If your files have different names, pass the corresponding command-line arguments directly.

## Training Subject 1

Run the scripts in order.

```bash
bash scripts/01_train_visual_prior_subj01.sh
bash scripts/02_build_coco_hdf5.sh
bash scripts/03_train_coco_adapter.sh
bash scripts/04_train_residual_bridge_subj01.sh
```

`01_train_visual_prior_subj01.sh` trains the fMRI-to-CLIP visual prior for NSD Subject 1. The bridge stages use COCO image-side features to learn the adapter and residual semantic bridge, then apply the learned bridge to brain-predicted CLIP tokens during inference.

## Inference And Evaluation

After obtaining the visual prior checkpoint and residual bridge checkpoint, use the generation and evaluation utilities:

```bash
bash scripts/05_generate_captions_subj01.sh
bash scripts/06_evaluate_interaction_subj01.sh
```

The scripts are templates. Update checkpoint paths and evaluation folders to match your training outputs.

## Notes

- This release is focused on the NSD Subject 1 protocol.
- The visual prior follows the MindEye/MindEye2-style training recipe and is trained for this pipeline rather than relying on a bundled pretrained checkpoint.
- The adapter is pretrained on COCO image-side feature pairs.
- The restricted residual bridge keeps the reconstruction-oriented visual prior available for image generation while producing Shikra-compatible semantic tokens for captioning, VQA, and grounding.

## Acknowledgements

This implementation builds on components and training conventions from MindEye/MindEye2, Shikra, OpenCLIP, and Stable Diffusion/SDXL. Please also follow the licenses and data-use terms of NSD, COCO, Shikra, CLIP/OpenCLIP, and SDXL.
