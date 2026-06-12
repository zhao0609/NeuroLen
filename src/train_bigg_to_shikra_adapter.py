#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR / "generative_models"))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_BIGG_CKPT = os.environ.get("BIGG_CKPT", "")


class BigGToShikraAdapter(nn.Module):
    """Notebook-style separable linear adapter: Bx256x1664 -> Bx256x1024."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(1664, 1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


def parse_args():
    parser = argparse.ArgumentParser("Train a MindEye2 bigG-token to Shikra CLIP-L-token adapter.")
    parser.add_argument("--hdf5_path", default=os.environ.get("COCO_HDF5", "data/coco_images_224_float16.hdf5"))
    parser.add_argument("--out_dir", default=os.environ.get("OUT_ROOT", "outputs/bigG_to_shikra_adapter"))
    parser.add_argument("--clip_l_path", default=os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14"))
    parser.add_argument("--bigg_cache_dir", default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    parser.add_argument(
        "--bigg_ckpt",
        default=DEFAULT_BIGG_CKPT,
        help="Local OpenCLIP ViT-bigG checkpoint. Falls back to OpenCLIP pretrained tag if missing.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--smoke_steps", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume adapter training from.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_batch(images_h5, indices, device):
    # h5py needs sorted indices for efficient fancy indexing.
    indices = np.sort(indices)
    image = torch.tensor(images_h5[indices], dtype=torch.float32, device=device)
    return image


def normalize_for_clip_l(image):
    mean = torch.tensor(OPENAI_CLIP_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


@torch.no_grad()
def compute_targets(image, bigg_embedder, clip_l_model):
    bigg_tokens = bigg_embedder(image)
    pixel_values = normalize_for_clip_l(image)
    clip_out = clip_l_model(pixel_values, output_hidden_states=True)
    shikra_tokens = clip_out.hidden_states[-2][:, 1:]
    return bigg_tokens.detach(), shikra_tokens.detach()


@torch.no_grad()
def validate(model, images_h5, val_indices, device, bigg_embedder, clip_l_model):
    model.eval()
    image = load_batch(images_h5, val_indices, device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        bigg_tokens, shikra_tokens = compute_targets(image, bigg_embedder, clip_l_model)
        pred = model(bigg_tokens)
    mse = F.mse_loss(pred.float(), shikra_tokens.float()).item()
    cos = F.cosine_similarity(pred.float().flatten(1), shikra_tokens.float().flatten(1), dim=-1).mean().item()
    token_cos = F.cosine_similarity(pred.float(), shikra_tokens.float(), dim=-1).mean().item()
    return {"mse": mse, "flat_cos": cos, "token_cos": token_cos}


def save_ckpt(path, model, optimizer, scheduler, epoch, step, args, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "adapter_type": "bigg_to_shikra_clip_l_minus2_256x1024",
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    with h5py.File(args.hdf5_path, "r") as f:
        images_h5 = f["images"]
        num_images = len(images_h5)

        bigg_version = args.bigg_ckpt if args.bigg_ckpt and Path(args.bigg_ckpt).exists() else "laion2b_s39b_b160k"
        if bigg_version != args.bigg_ckpt:
            print(json.dumps({"warning": "bigG checkpoint missing; falling back to OpenCLIP tag", "bigg_ckpt": args.bigg_ckpt}), flush=True)

        bigg_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version=bigg_version,
            output_tokens=True,
            only_tokens=True,
            cache_dir=args.bigg_cache_dir,
        ).to(device)
        bigg_embedder.eval().requires_grad_(False)

        clip_l_model = CLIPVisionModel.from_pretrained(args.clip_l_path, torch_dtype=torch.float16)
        clip_l_model.to(device).eval().requires_grad_(False)

        model = BigGToShikraAdapter().to(device)
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(grouped, lr=args.max_lr)

        # Keep the notebook's scheduler convention: total_steps uses the whole HDF5 size,
        # even when each epoch samples only steps_per_epoch random batches.
        scheduler_total_steps = int(np.floor(args.num_epochs * num_images // args.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=max(1, scheduler_total_steps),
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=min(0.3, 2 / max(1, args.num_epochs)),
        )

        start_epoch = 0
        global_step = 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("step", 0))
            print(json.dumps({"resumed": args.resume, "start_epoch": start_epoch, "global_step": global_step}), flush=True)

        val_size = min(args.val_size, num_images)
        val_indices = np.arange(val_size)
        metadata = {
            "hdf5_path": args.hdf5_path,
            "num_images": num_images,
            "clip_l_path": args.clip_l_path,
            "bigg_version": bigg_version,
            "params": count_params(model),
            "args": vars(args),
            "scheduler_total_steps": scheduler_total_steps,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)

        max_steps = args.smoke_steps if args.smoke_steps > 0 else args.num_epochs * args.steps_per_epoch
        start_time = time.time()

        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            for step_in_epoch in range(args.steps_per_epoch):
                if global_step >= max_steps:
                    break
                batch = np.random.choice(np.arange(1, num_images), size=args.batch_size, replace=False)
                image = load_batch(images_h5, batch, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    bigg_tokens, shikra_tokens = compute_targets(image, bigg_embedder, clip_l_model)
                    pred = model(bigg_tokens)
                    loss = F.mse_loss(pred.float(), shikra_tokens.float())

                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1
                if global_step % args.log_every == 0 or global_step == 1:
                    elapsed = time.time() - start_time
                    with torch.no_grad():
                        flat_cos = F.cosine_similarity(
                            pred.float().flatten(1), shikra_tokens.float().flatten(1), dim=-1
                        ).mean().item()
                        token_cos = F.cosine_similarity(pred.float(), shikra_tokens.float(), dim=-1).mean().item()
                    rec = {
                        "epoch": epoch + 1,
                        "step_in_epoch": step_in_epoch + 1,
                        "global_step": global_step,
                        "loss": float(loss.item()),
                        "flat_cos": flat_cos,
                        "token_cos": token_cos,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": elapsed,
                    }
                    print(json.dumps(rec), flush=True)
                    with log_path.open("a", encoding="utf-8") as f_log:
                        f_log.write(json.dumps(rec) + "\n")

            val_metrics = validate(model, images_h5, val_indices, device, bigg_embedder, clip_l_model)
            val_rec = {"epoch": epoch + 1, "global_step": global_step, "val": val_metrics}
            print(json.dumps(val_rec), flush=True)
            with log_path.open("a", encoding="utf-8") as f_log:
                f_log.write(json.dumps(val_rec) + "\n")

            if ((epoch + 1) % args.save_every == 0) or (epoch + 1 == args.num_epochs):
                save_ckpt(out_dir / f"epoch{epoch + 1}.pth", model, optimizer, scheduler, epoch + 1, global_step, args, val_metrics)
                save_ckpt(out_dir / "last.pth", model, optimizer, scheduler, epoch + 1, global_step, args, val_metrics)

            if global_step >= max_steps:
                break

        final_metrics = validate(model, images_h5, val_indices, device, bigg_embedder, clip_l_model)
        save_ckpt(out_dir / "last.pth", model, optimizer, scheduler, epoch + 1, global_step, args, final_metrics)
        print(json.dumps({"finished": True, "global_step": global_step, "final": final_metrics}), flush=True)


if __name__ == "__main__":
    main()
