#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR / "generative_models"))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
import open_clip
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

import utils
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from generative_models.sgm.util import append_dims
from models import BrainDiffusionPrior, BrainNetwork, PriorNetwork


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_BIGG_CKPT = os.environ.get("BIGG_CKPT", "")


class BigGImageEmbedder(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-bigG-14",
            pretrained=str(ckpt_path),
            device=device,
        )
        if hasattr(model, "transformer"):
            del model.transformer
        model.visual.output_tokens = True
        self.model = model.eval().requires_grad_(False)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    def forward(self, image):
        image = self.preprocess(image)
        _, tokens = self.model.visual(image)
        return tokens


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([nn.Linear(input_size, out_features) for input_size in input_sizes])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def parse_args():
    parser = argparse.ArgumentParser("Fast MindEye2 scratch checkpoint reconstruction/evaluation.")
    parser.add_argument("--data_path", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", os.environ.get("DATA_ROOT", "data")))
    parser.add_argument("--model_name", default="subj01_mindeye2_scratch_40sess_full_4096")
    parser.add_argument(
        "--ckpt",
        default="../train_logs/subj01_mindeye2_scratch_40sess_full_4096/last_model_only.pth",
    )
    parser.add_argument("--out_dir", default=os.environ.get("EVAL_ROOT", "outputs/evals/subj01_scratch_fast"))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--prior_eval", type=int, default=200)
    parser.add_argument("--num_recons", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prior_batch_size", type=int, default=4)
    parser.add_argument("--prior_timesteps", type=int, default=20)
    parser.add_argument("--unclip_steps", type=int, default=38)
    parser.add_argument("--bigg_ckpt", default=DEFAULT_BIGG_CKPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_unclip", action="store_true")
    return parser.parse_args()


def load_test_arrays(args):
    voxels = {}
    with h5py.File(f"{args.data_path}/betas_all_subj0{args.subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.tensor(f["betas"][:], dtype=torch.float32)
    voxels[f"subj0{args.subj}"] = betas
    num_voxels = betas.shape[-1]

    num_test = 3000 if args.subj not in (3, 4, 6, 8) else {3: 2371, 4: 2188, 6: 2371, 8: 2188}[args.subj]
    test_url = f"{args.data_path}/wds/subj0{args.subj}/new_test/0.tar"

    def my_split_by_node(urls):
        return urls

    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)

    test_images_idx = []
    test_voxels = None
    for test_i, (behav, _past_behav, _future_behav, _old_behav) in enumerate(test_dl):
        test_voxels = voxels[f"subj0{args.subj}"][behav[:, 0, 5].cpu().long()]
        test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
    assert test_i == 0
    test_images_idx = test_images_idx.astype(int)
    unique_images = np.unique(test_images_idx)
    if args.num_eval > 0:
        unique_images = unique_images[: args.num_eval]
    return num_voxels, test_voxels, test_images_idx, unique_images


def build_model(args, num_voxels, device):
    clip_seq_dim = 256
    clip_emb_dim = 1664

    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=args.hidden_dim)
    model.backbone = BrainNetwork(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=1,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        n_blocks=args.n_blocks,
        blurry_recon=True,
        clip_scale=1,
    )
    prior_network = PriorNetwork(
        dim=clip_emb_dim,
        depth=6,
        dim_head=52,
        heads=clip_emb_dim // 52,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, ckpt.get("epoch", None)


def build_autoenc(args, device):
    from diffusers import AutoencoderKL

    autoenc = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f"{args.cache_dir}/sd_image_var_autoenc.pth", map_location="cpu")
    autoenc.load_state_dict(ckpt)
    autoenc.eval().requires_grad_(False).to(device)
    return autoenc


def build_unclip(args, device):
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    params = OmegaConf.to_container(config, resolve=True)["model"]["params"]
    params["sampler_config"]["params"]["num_steps"] = args.unclip_steps
    params["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
    params["first_stage_config"]["params"]["ddconfig"]["attn_type"] = "vanilla"

    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=params["first_stage_config"],
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
    )
    ckpt = torch.load(f"{args.cache_dir}/unclip6_epoch0_step110000.ckpt", map_location="cpu")
    engine.load_state_dict(ckpt["state_dict"])
    engine.eval().requires_grad_(False).to(device)
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    vector_suffix = engine.conditioner(batch)["vector"].to(device)
    offset_noise_level = float(params["loss_fn_config"]["params"].get("offset_noise_level", 0.04))
    return engine, vector_suffix, offset_noise_level


def prepare_voxel_batch(test_voxels, test_images_idx, uniq_imgs):
    rows = []
    for uniq_img in uniq_imgs:
        locs = np.where(test_images_idx == int(uniq_img))[0]
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = locs.repeat(2)[:3]
        assert len(locs) == 3, (uniq_img, len(locs))
        rows.append(test_voxels[locs][None])
    return torch.vstack(rows)


@torch.no_grad()
def forward_brain(model, voxel, device):
    voxel = voxel.to(device)
    backbone = None
    clip_voxels = None
    blurry = None
    for rep in range(3):
        voxel_ridge = model.ridge(voxel[:, [rep]], 0)
        backbone0, clip_voxels0, blurry0 = model.backbone(voxel_ridge)
        if rep == 0:
            backbone = backbone0
            clip_voxels = clip_voxels0
            blurry = blurry0[0]
        else:
            backbone = backbone + backbone0
            clip_voxels = clip_voxels + clip_voxels0
            blurry = blurry + blurry0[0]
    return backbone / 3, clip_voxels / 3, blurry / 3


@torch.no_grad()
def unclip_recon(tokens, engine, vector_suffix, offset_noise_level, seed):
    device = next(engine.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.cuda.amp.autocast(dtype=torch.float16), engine.ema_scope():
        z = torch.randn(1, 4, 96, 96, device=device, generator=generator)
        c = {"crossattn": tokens.to(device), "vector": vector_suffix}
        uc_tokens = torch.randn(tokens.shape, device=device, dtype=tokens.dtype, generator=generator)
        uc = {"crossattn": uc_tokens, "vector": vector_suffix}
        noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
        sigmas = engine.sampler.discretization(engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)
        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device=device, generator=generator), z.ndim)
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

        def denoiser(x, sigma, cond):
            return engine.denoiser(engine.model, x, sigma, cond)

        samples_z = engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = engine.decode_first_stage(samples_z)
        return torch.clamp(samples_x * 0.8 + 0.2, min=0.0, max=1.0).float().cpu()


def topk_acc(sim, k):
    k = min(k, sim.shape[1])
    labels = torch.arange(sim.shape[0], device=sim.device)
    return float((sim.topk(k, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item())


def retrieval_metrics(pred_norm, target_norm, prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = pred_norm.to(device)
    target = target_norm.to(device)
    sim_target_to_brain = target @ pred.T
    sim_brain_to_target = pred @ target.T
    return {
        f"{prefix}_diag_cos": float(torch.diag(sim_target_to_brain).mean().item()),
        f"{prefix}_image_to_brain_top1": topk_acc(sim_target_to_brain, 1),
        f"{prefix}_image_to_brain_top5": topk_acc(sim_target_to_brain, 5),
        f"{prefix}_brain_to_image_top1": topk_acc(sim_brain_to_target, 1),
        f"{prefix}_brain_to_image_top5": topk_acc(sim_brain_to_target, 5),
    }


def tensor_to_pil(x, size=192):
    return transforms.ToPILImage()(x.detach().cpu().clamp(0, 1)).resize((size, size), Image.Resampling.LANCZOS)


def save_grid(originals, recons, blurry, out_path):
    cell = 192
    header = 30
    columns = ["original", "brain_prior_unclip", "blurry"]
    grid = Image.new("RGB", (cell * len(columns), header + cell * len(originals)), "white")
    draw = ImageDraw.Draw(grid)
    for j, name in enumerate(columns):
        draw.text((j * cell + 8, 8), name, fill=(0, 0, 0))
    for i in range(len(originals)):
        grid.paste(tensor_to_pil(originals[i], cell), (0, header + i * cell))
        grid.paste(tensor_to_pil(recons[i], cell), (cell, header + i * cell))
        grid.paste(tensor_to_pil(blurry[i], cell), (2 * cell, header + i * cell))
    grid.save(out_path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_voxels, test_voxels, test_images_idx, unique_images = load_test_arrays(args)
    print(f"unique test images: {len(unique_images)} num_voxels={num_voxels}")

    model, ckpt_epoch = build_model(args, num_voxels, device)
    autoenc = build_autoenc(args, device)
    clip_img_embedder = BigGImageEmbedder(args.bigg_ckpt, device).eval().requires_grad_(False)

    images_h5 = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")["images"]

    pred_norms = []
    target_norms = []
    prior_norms = []
    target_prior_norms = []
    clip_diag_cos_sum = 0.0
    clip_mse_sum = 0.0
    prior_diag_cos_sum = 0.0
    prior_mse_sum = 0.0
    blurry_pixcorr_sum = 0.0
    n_clip = 0
    n_prior = 0
    selected = {"originals": [], "blurry": [], "prior_tokens": [], "target_tokens": []}
    prior_limit = max(args.prior_eval, args.num_recons)

    for start in tqdm(range(0, len(unique_images), args.batch_size), desc="feature_eval"):
        uniq_imgs = unique_images[start : start + args.batch_size]
        image = torch.tensor(images_h5[uniq_imgs], dtype=torch.float32)
        voxel = prepare_voxel_batch(test_voxels, test_images_idx, uniq_imgs)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            backbone, clip_voxels, blurry_latent = forward_brain(model, voxel, device)
            target = clip_img_embedder(image.to(device))
            blurry_img = (autoenc.decode(blurry_latent / 0.18215).sample / 2 + 0.5).clamp(0, 1)

        bsz = len(uniq_imgs)
        clip_flat = clip_voxels.float().flatten(1)
        target_flat = target.float().flatten(1)
        clip_diag_cos_sum += float(F.cosine_similarity(clip_flat, target_flat, dim=-1).sum().item())
        clip_mse_sum += float(F.mse_loss(clip_voxels.float(), target.float(), reduction="sum").item())
        blurry_pixcorr_sum += float(utils.pixcorr(image, blurry_img.cpu().float()).item() * bsz)
        n_clip += bsz

        pred_norms.append(F.normalize(clip_flat, dim=-1).half().cpu())
        target_norms.append(F.normalize(target_flat, dim=-1).half().cpu())

        if n_prior < prior_limit:
            take = min(bsz, prior_limit - n_prior)
            sub_backbone = backbone[:take]
            sub_target = target[:take]
            prior_rows = []
            for pstart in range(0, take, args.prior_batch_size):
                pend = min(take, pstart + args.prior_batch_size)
                prior_rows.append(
                    model.diffusion_prior.p_sample_loop(
                        sub_backbone[pstart:pend].shape,
                        text_cond=dict(text_embed=sub_backbone[pstart:pend]),
                        cond_scale=1.0,
                        timesteps=args.prior_timesteps,
                    ).float()
                )
            prior_out = torch.cat(prior_rows, dim=0)
            prior_flat = prior_out.flatten(1)
            target_prior_flat = sub_target.float().flatten(1)
            prior_diag_cos_sum += float(F.cosine_similarity(prior_flat, target_prior_flat, dim=-1).sum().item())
            prior_mse_sum += float(F.mse_loss(prior_out, sub_target.float(), reduction="sum").item())
            prior_norms.append(F.normalize(prior_flat, dim=-1).half().cpu())
            target_prior_norms.append(F.normalize(target_prior_flat, dim=-1).half().cpu())

            selected_count = sum(len(x) for x in selected["originals"])
            keep = max(0, min(args.num_recons - selected_count, take))
            if keep:
                selected["originals"].append(image[:keep].cpu())
                selected["blurry"].append(blurry_img[:keep].float().cpu())
                selected["prior_tokens"].append(prior_out[:keep].half().cpu())
                selected["target_tokens"].append(sub_target[:keep].half().cpu())
            n_prior += take

    pred_norm = torch.cat(pred_norms, dim=0)
    target_norm = torch.cat(target_norms, dim=0)
    metrics = {
        "model_name": args.model_name,
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt_epoch,
        "num_eval": int(n_clip),
        "prior_eval": int(n_prior),
        "clip_voxels_diag_cos": clip_diag_cos_sum / n_clip,
        "clip_voxels_mse": clip_mse_sum / (n_clip * 256 * 1664),
        "blurry_pixcorr": blurry_pixcorr_sum / n_clip,
    }
    metrics.update(retrieval_metrics(pred_norm.float(), target_norm.float(), "clip_voxels"))

    if prior_norms:
        prior_norm = torch.cat(prior_norms, dim=0)
        target_prior_norm = torch.cat(target_prior_norms, dim=0)
        metrics["prior_diag_cos"] = prior_diag_cos_sum / n_prior
        metrics["prior_mse"] = prior_mse_sum / (n_prior * 256 * 1664)
        metrics.update(retrieval_metrics(prior_norm.float(), target_prior_norm.float(), "prior"))

    for k in list(selected):
        if selected[k]:
            selected[k] = torch.cat(selected[k], dim=0)

    out_metrics = out_dir / "metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    torch.save(selected, out_dir / "selected_conditions.pt")
    print(json.dumps(metrics, indent=2))
    print(f"saved metrics: {out_metrics}")

    if args.skip_unclip or args.num_recons <= 0:
        return

    del model, autoenc, clip_img_embedder
    torch.cuda.empty_cache()

    engine, vector_suffix, offset_noise_level = build_unclip(args, device)
    recons = []
    for i, tokens in enumerate(tqdm(selected["prior_tokens"], desc="unclip_recon")):
        recons.append(
            transforms.Resize((256, 256), antialias=True)(
                unclip_recon(tokens[None].to(device), engine, vector_suffix, offset_noise_level, args.seed + i)[0]
            )
        )
    recons = torch.stack(recons).clamp(0, 1)
    originals = transforms.Resize((256, 256), antialias=True)(selected["originals"].float())
    blurry = transforms.Resize((256, 256), antialias=True)(selected["blurry"].float())
    torch.save({"originals": originals, "recons": recons, "blurry": blurry}, out_dir / "recons.pt")
    save_grid(originals, recons, blurry, out_dir / "comparison_grid.png")

    del engine
    torch.cuda.empty_cache()
    clip_img_embedder = BigGImageEmbedder(args.bigg_ckpt, device).eval().requires_grad_(False)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        recon_tokens = clip_img_embedder(recons.to(device)).float().flatten(1)
        orig_tokens = clip_img_embedder(originals.to(device)).float().flatten(1)
    image_metrics = {
        "recon_count": int(len(recons)),
        "recon_pixcorr": float(utils.pixcorr(originals, recons).item()),
        "recon_clip_bigG_cos": float(F.cosine_similarity(recon_tokens, orig_tokens, dim=-1).mean().item()),
        "recon_mean": float(recons.mean().item()),
        "original_mean": float(originals.mean().item()),
        "blurry_pixcorr_selected": float(utils.pixcorr(originals, blurry).item()),
    }
    (out_dir / "image_metrics.json").write_text(json.dumps(image_metrics, indent=2))
    print(json.dumps(image_metrics, indent=2))
    print(f"saved grid: {out_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()
