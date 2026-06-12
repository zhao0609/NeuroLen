# Release File Map

- `src/train_visual_prior.py`: Subject 1 fMRI-to-CLIP visual prior training.
- `src/train_bigg_to_shikra_adapter.py`: first-stage COCO adapter training.
- `src/train_bigg_to_shikra_vlm_ce.py`: VLM-supervised restricted residual bridge training.
- `src/recon_inference.py`: base diffusion reconstruction.
- `src/enhanced_recon_inference.py`: semantic-guided reconstruction enhancement.
- `src/experimentA_generate_calibrated_shikra_prompts.py`: caption generation from residual-bridge tokens.
- `src/generate_neurolens_interaction_answers.py`: open-ended interaction answer generation.
- `src/evaluate_neurolens_interaction_bertscore.py`: BERTScore evaluation for interaction answers.
- `src/generative_models/`: SDXL/generative-models dependency copied for local imports.
- `scripts/`: runnable Subject 1 templates.

Large datasets, pretrained model weights, and generated outputs are intentionally excluded.
