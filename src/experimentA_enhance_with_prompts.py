#!/usr/bin/env python
"""Run the MindEye2 SDXL img2img enhancement for a prompt variant."""

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR / "generative_models"))

import torch
import types
from accelerate import Accelerator
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

import utils


torch.backends.cuda.matmul.allow_tf32 = True


def patch_openclip_batch_first(embedder):
    """Adapt older SGM text wrapper code to newer OpenCLIP batch_first blocks."""
    model = getattr(embedder, "model", None)
    transformer = getattr(model, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if not resblocks:
        return
    first_attn = getattr(resblocks[0], "attn", None)
    if not getattr(first_attn, "batch_first", False):
        return

    def text_transformer_forward(self, x, attn_mask=None):
        outputs = {}
        for i, block in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x
            x = block(x, attn_mask=attn_mask)
        outputs["last"] = x
        return outputs

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        out = x["last"]
        out = self.model.ln_final(out)
        x["pooled"] = self.pool(out, text)
        return x

    embedder.text_transformer_forward = types.MethodType(text_transformer_forward, embedder)
    embedder.encode_with_transformer = types.MethodType(encode_with_transformer, embedder)


def patch_conditioner_openclip_batch_first(conditioner):
    for embedder in getattr(conditioner, "embedders", []):
        patch_openclip_batch_first(embedder)


def parse_args():
    parser = argparse.ArgumentParser("Experiment A MindEye2 enhanced reconstruction")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--base_ckpt_path", default=os.environ.get("SDXL_CKPT", "models/zavychromaxl_v30.safetensors"))
    parser.add_argument(
        "--openclip_bigg_ckpt",
        default=os.environ.get("BIGG_CKPT", ""),
    )
    parser.add_argument("--num_eval", type=int, default=0, help="<=0 means all.")
    parser.add_argument("--img2img_timepoint", type=int, default=13)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=25, help="Save partial enhanced recon every N images; <=0 disables partial saves.")
    return parser.parse_args()


def build_sdxl_engine(base_ckpt_path, openclip_bigg_ckpt, device):
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    sampler_config = config["model"]["params"]["sampler_config"]
    sampler_config["params"]["num_steps"] = 38

    config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
    refiner_params = OmegaConf.to_container(config, resolve=True)["model"]["params"]
    # This host does not have xformers. Keep the checkpoint and sampler fixed,
    # but use the native VAE attention implementation to avoid decode failures.
    refiner_params["first_stage_config"]["params"]["ddconfig"]["attn_type"] = "vanilla"
    openclip_bigg_ckpt = Path(openclip_bigg_ckpt)
    if openclip_bigg_ckpt.exists():
        refiner_params["conditioner_config"]["params"]["emb_models"][1]["params"]["version"] = str(openclip_bigg_ckpt)

    base_engine = DiffusionEngine(
        network_config=refiner_params["network_config"],
        denoiser_config=refiner_params["denoiser_config"],
        first_stage_config=refiner_params["first_stage_config"],
        conditioner_config=refiner_params["conditioner_config"],
        sampler_config=sampler_config,
        scale_factor=refiner_params["scale_factor"],
        disable_first_stage_autocast=refiner_params["disable_first_stage_autocast"],
        ckpt_path=str(base_ckpt_path),
    )
    base_engine.eval().requires_grad_(False).to(device)
    patch_conditioner_openclip_batch_first(base_engine.conditioner)

    conditioner_config = refiner_params["conditioner_config"]
    base_text_embedder1 = FrozenCLIPEmbedder(
        layer=conditioner_config["params"]["emb_models"][0]["params"]["layer"],
        layer_idx=conditioner_config["params"]["emb_models"][0]["params"]["layer_idx"],
    ).to(device)
    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config["params"]["emb_models"][1]["params"]["arch"],
        version=str(openclip_bigg_ckpt)
        if Path(openclip_bigg_ckpt).exists()
        else conditioner_config["params"]["emb_models"][1]["params"]["version"],
        freeze=conditioner_config["params"]["emb_models"][1]["params"]["freeze"],
        layer=conditioner_config["params"]["emb_models"][1]["params"]["layer"],
        always_return_pooled=conditioner_config["params"]["emb_models"][1]["params"]["always_return_pooled"],
        legacy=conditioner_config["params"]["emb_models"][1]["params"]["legacy"],
    ).to(device)
    patch_openclip_batch_first(base_text_embedder2)
    return base_engine, base_text_embedder1, base_text_embedder2


def base_conditioning(base_engine, device):
    batch = {
        "txt": [""],
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
        "target_size_as_tuple": torch.ones(1, 2, device=device) * 1024,
    }
    out = base_engine.conditioner(batch)
    vector_suffix = out["vector"][:, -1536:].to(device)
    batch_uc = {
        "txt": ["painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime"],
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
        "target_size_as_tuple": torch.ones(1, 2, device=device) * 1024,
    }
    out_uc = base_engine.conditioner(batch_uc)
    return vector_suffix, out_uc["crossattn"].to(device), out_uc["vector"].to(device)


@torch.no_grad()
def enhance_one(image, prompt, base_engine, text1, text2, vector_suffix, crossattn_uc, vector_uc, args, device):
    image = image.to(device)
    base_engine.sampler.num_steps = args.num_steps
    z = base_engine.encode_first_stage(image * 2 - 1)
    openai_clip_text = text1([prompt])
    clip_text_tokenized, clip_text_emb = text2([prompt])
    clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
    clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
    c = {"crossattn": clip_text_tokenized, "vector": clip_text_emb}
    uc = {"crossattn": crossattn_uc, "vector": vector_uc}

    noise = torch.randn_like(z)
    sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
    init_z = (z + noise * append_dims(sigmas[-args.img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
    sigmas = sigmas[-args.img2img_timepoint:].repeat(1, 1)
    base_engine.sampler.num_steps = sigmas.shape[-1] - 1
    noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
        init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps
    )

    def denoiser(x, sigma, cond):
        return base_engine.denoiser(base_engine.model, x, sigma, cond)

    for timestep in range(base_engine.sampler.num_steps):
        noised_z = base_engine.sampler.sampler_step(
            sigmas[:, timestep],
            sigmas[:, timestep + 1],
            denoiser,
            noised_z,
            cond=c,
            uc=uc,
            gamma=0,
        )
    samples_x = base_engine.decode_first_stage(noised_z)
    return torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main():
    args = parse_args()
    utils.seed_everything(args.seed)
    ckpt = Path(args.base_ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing SDXL checkpoint: {ckpt}. Download zavychromaxl_v30.safetensors before running enhancement."
        )

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    eval_root = Path(args.eval_root)
    model_dir = eval_root / args.model_name
    all_images = torch.load(eval_root / "all_images.pt", map_location="cpu")
    all_recons = torch.load(model_dir / f"{args.model_name}_all_recons.pt", map_location="cpu")
    all_predcaptions = torch.load(model_dir / f"{args.model_name}_all_predcaptions.pt", map_location="cpu")
    if args.num_eval > 0:
        all_images = all_images[: args.num_eval]
        all_recons = all_recons[: args.num_eval]
        all_predcaptions = all_predcaptions[: args.num_eval]

    all_recons = transforms.Resize((768, 768), antialias=True)(all_recons).float()
    base_engine, text1, text2 = build_sdxl_engine(ckpt, args.openclip_bigg_ckpt, device)
    base_engine.sampler.guider.scale = args.cfg
    base_engine.sampler.num_steps = args.num_steps
    vector_suffix, crossattn_uc, vector_uc = base_conditioning(base_engine, device)

    out_path = model_dir / f"{args.model_name}_all_enhancedrecons.pt"
    partial_path = model_dir / f"{args.model_name}_all_enhancedrecons.partial.pt"
    if args.resume and partial_path.exists():
        enhanced = list(torch.load(partial_path, map_location="cpu"))
        start_idx = len(enhanced)
    else:
        enhanced = []
        start_idx = 0

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
        for img_idx in tqdm(range(start_idx, len(all_recons)), desc=f"enhance:{args.model_name}"):
            prompt = str(all_predcaptions[img_idx])
            sample = enhance_one(
                all_recons[[img_idx]],
                prompt,
                base_engine,
                text1,
                text2,
                vector_suffix,
                crossattn_uc,
                vector_uc,
                args,
                device,
            )
            enhanced.append(sample[0])
            if args.save_every > 0 and len(enhanced) % args.save_every == 0:
                torch.save(torch.stack(enhanced), partial_path)

    all_enhanced = transforms.Resize((256, 256), antialias=True)(torch.stack(enhanced)).float()
    torch.save(all_enhanced, out_path)
    metadata = {
        "model_name": args.model_name,
        "base_ckpt_path": str(ckpt),
        "num_enhanced": int(len(all_enhanced)),
        "img2img_timepoint": args.img2img_timepoint,
        "num_steps": args.num_steps,
        "cfg": args.cfg,
        "save_every": args.save_every,
    }
    (model_dir / "enhance_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
