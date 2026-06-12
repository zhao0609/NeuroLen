#!/usr/bin/env python
"""Train a bigG->Shikra adapter with Shikra caption CE loss in the loop.

This is intentionally heavier than the cached-token calibration scripts:

    image -> frozen OpenCLIP ViT-bigG tokens -> trainable adapter -> Shikra

The Shikra LLM is frozen, but gradients are propagated through it to the
adapter output. CLIP-L token losses are kept as stabilizers so the optimized
tokens remain close to the vision-token manifold expected by Shikra.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR / "generative_models"))
SHIKRA_REPO = os.environ.get("SHIKRA_REPO", str(SRC_DIR.parent / "third_party" / "shikra-main"))
sys.path.insert(0, SHIKRA_REPO)

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPVisionModel

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from mllm.conversation import get_conv_template
from mllm.models.shikra import ShikraLlamaForCausalLM


IGNORE_INDEX = -100
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_BIGG_CKPT = os.environ.get("BIGG_CKPT", "")


class BigGToShikraAdapter(nn.Module):
    """Notebook-style separable adapter: Bx256x1664 -> Bx256x1024."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(1664, 1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        return self.linear2(x.permute(0, 2, 1))


class ResidualShikraAdapter(nn.Module):
    """Frozen COCO adapter plus a gated residual correction."""

    def __init__(self, base_adapter, hidden_dim=1536, max_gate=0.05, init_gate_bias=-4.0, dropout=0.05):
        super().__init__()
        self.base_adapter = base_adapter.eval().requires_grad_(False)
        self.max_gate = max_gate
        self.bigg_context = nn.Sequential(
            nn.LayerNorm(1664),
            nn.Linear(1664, 1024),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 384),
            nn.GELU(),
            nn.Linear(384, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    def forward(self, bigg_tokens):
        base = self.base_adapter(bigg_tokens)
        joint = torch.cat((base, self.bigg_context(bigg_tokens)), dim=-1)
        delta = self.delta(joint)
        gate = torch.sigmoid(self.gate(joint)) * self.max_gate
        refined = base + gate * delta
        return refined, base, delta, gate


def parse_args():
    parser = argparse.ArgumentParser("Train bigG->Shikra adapter with frozen Shikra CE loss.")
    parser.add_argument("--hdf5_path", default=os.environ.get("COCO_HDF5", "data/coco_images_224_float16.hdf5"))
    parser.add_argument("--subj_indices_path", default=os.environ.get("SUBJ_INDICES", "data/COCO_73k_subj_indices.hdf5"))
    parser.add_argument("--annots_path", default=os.environ.get("SUBJ01_ANNOTS", "data/subj01_annots.npy"))
    parser.add_argument("--shared1000_path", default=os.environ.get("SHARED1000", "data/shared1000.npy"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["adapter", "residual"], default="adapter")
    parser.add_argument("--init_adapter", default=os.environ.get("INIT_ADAPTER", "outputs/bigG_to_shikra_adapter/last.pth"))
    parser.add_argument("--shikra_path", default=os.environ.get("SHIKRA_PATH", "shikra-7b-v1-0708"))
    parser.add_argument("--clip_l_path", default=os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14"))
    parser.add_argument("--bigg_cache_dir", default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    parser.add_argument("--bigg_ckpt", default=DEFAULT_BIGG_CKPT)
    parser.add_argument("--question", default="Describe the image briefly.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--target_mse_weight", type=float, default=0.05)
    parser.add_argument("--target_cos_weight", type=float, default=0.2)
    parser.add_argument("--preserve_mse_weight", type=float, default=0.1)
    parser.add_argument("--preserve_cos_weight", type=float, default=0.2)
    parser.add_argument("--norm_weight", type=float, default=0.2)
    parser.add_argument("--residual_max_gate", type=float, default=0.05)
    parser.add_argument("--residual_init_gate_bias", type=float, default=-4.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_adapter_state(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "bridge_train"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    return {k.removeprefix("module."): v for k, v in payload.items()}


def make_adapter(args, device):
    base = BigGToShikraAdapter()
    if args.init_adapter and Path(args.init_adapter).exists():
        base.load_state_dict(load_adapter_state(args.init_adapter), strict=True)
    if args.mode == "adapter":
        return base.to(device), None
    frozen_base = BigGToShikraAdapter()
    frozen_base.load_state_dict(base.state_dict(), strict=True)
    model = ResidualShikraAdapter(
        frozen_base,
        max_gate=args.residual_max_gate,
        init_gate_bias=args.residual_init_gate_bias,
    )
    return model.to(device), frozen_base.to(device).eval().requires_grad_(False)


def normalize_for_clip_l(image):
    mean = torch.tensor(OPENAI_CLIP_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


@torch.no_grad()
def compute_vision_tokens(image, bigg_embedder, clip_l_model):
    bigg_tokens = bigg_embedder(image)
    clip_out = clip_l_model(normalize_for_clip_l(image), output_hidden_states=True)
    shikra_tokens = clip_out.hidden_states[-2][:, 1:]
    return bigg_tokens.detach(), shikra_tokens.detach()


def build_prompt_ids(tokenizer, question, answer):
    image_tokens = "<im_start>" + "<im_patch>" * 256 + "<im_end>"

    conv_full = get_conv_template("vicuna_v1.1")
    conv_full.append_message(conv_full.roles[0], image_tokens + "\n" + question)
    conv_full.append_message(conv_full.roles[1], str(answer).strip())
    full_ids = tokenizer(conv_full.get_prompt(), return_tensors="pt").input_ids[0]

    conv_prefix = get_conv_template("vicuna_v1.1")
    conv_prefix.append_message(conv_prefix.roles[0], image_tokens + "\n" + question)
    conv_prefix.append_message(conv_prefix.roles[1], None)
    prefix_ids = tokenizer(conv_prefix.get_prompt(), return_tensors="pt").input_ids[0]

    labels = full_ids.clone()
    labels[: min(prefix_ids.numel(), labels.numel())] = IGNORE_INDEX
    return full_ids, labels


def collate_text(tokenizer, captions, question, device):
    ids, labels = [], []
    for caption in captions:
        input_ids, target = build_prompt_ids(tokenizer, question, caption)
        ids.append(input_ids)
        labels.append(target)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to(device)
    attention_mask = input_ids.ne(pad_id)
    return input_ids, attention_mask, labels


def load_image_batch(images_h5, image_indices, device):
    imgs = [torch.from_numpy(images_h5[int(idx)][None]) for idx in image_indices]
    image = torch.cat(imgs, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
    return image


def make_train_val_indices(args):
    captions = np.load(args.annots_path, allow_pickle=True).astype(str)
    with h5py.File(args.subj_indices_path, "r") as f:
        image_indices = f["subj01"][:].astype(np.int64)
    if args.shared1000_path and Path(args.shared1000_path).exists():
        shared = np.load(args.shared1000_path, allow_pickle=True).astype(bool)
        train_mask = ~shared[image_indices]
    else:
        train_mask = np.ones_like(image_indices, dtype=bool)
    train_rows = np.where(train_mask)[0]
    val_rows = np.where(~train_mask)[0]
    if len(val_rows) == 0:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(train_rows))
        val_rows = train_rows[perm[: args.val_size]]
        train_rows = train_rows[perm[args.val_size :]]
    return image_indices, captions, train_rows, val_rows[: args.val_size]


def loss_terms(pred, target, base):
    pred_f = pred.float()
    target_f = target.float()
    base_f = base.float()
    target_mse = F.smooth_l1_loss(pred_f, target_f)
    target_cos = 1.0 - F.cosine_similarity(pred_f.flatten(1), target_f.flatten(1), dim=-1).mean()
    preserve_mse = F.smooth_l1_loss(pred_f, base_f)
    preserve_cos = 1.0 - F.cosine_similarity(pred_f.flatten(1), base_f.flatten(1), dim=-1).mean()
    norm_loss = (pred_f.flatten(1).norm(dim=-1).mean() / base_f.flatten(1).norm(dim=-1).mean().clamp_min(1e-6) - 1.0).abs()
    return target_mse, target_cos, preserve_mse, preserve_cos, norm_loss


@torch.no_grad()
def evaluate(args, model, base_adapter, images_h5, image_indices, captions, val_rows, tokenizer, shikra, bigg, clip_l, device):
    model.eval()
    total = {
        "loss": 0.0,
        "ce": 0.0,
        "target_mse": 0.0,
        "target_cos": 0.0,
        "preserve_mse": 0.0,
        "preserve_cos": 0.0,
        "norm": 0.0,
        "pred_to_target_cos": 0.0,
        "pred_to_base_cos": 0.0,
    }
    n = 0
    rows = list(val_rows[: args.val_size])
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        img_idx = image_indices[batch_rows]
        cap = captions[batch_rows]
        image = load_image_batch(images_h5, img_idx, device)
        input_ids, attention_mask, labels = collate_text(tokenizer, cap, args.question, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            bigg_tokens, shikra_tokens = compute_vision_tokens(image, bigg, clip_l)
            if args.mode == "residual":
                pred, base, _delta, _gate = model(bigg_tokens)
            else:
                pred = model(bigg_tokens)
                base = base_adapter(bigg_tokens)
            out = shikra(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_features=pred.to(dtype=torch.float16),
                use_cache=False,
            )
            tmse, tcos, pmse, pcos, nloss = loss_terms(pred, shikra_tokens, base)
            loss = (
                args.ce_weight * out.loss.float()
                + args.target_mse_weight * tmse
                + args.target_cos_weight * tcos
                + args.preserve_mse_weight * pmse
                + args.preserve_cos_weight * pcos
                + args.norm_weight * nloss
            )
        bs = len(batch_rows)
        total["loss"] += float(loss.item()) * bs
        total["ce"] += float(out.loss.float().item()) * bs
        total["target_mse"] += float(tmse.item()) * bs
        total["target_cos"] += float(tcos.item()) * bs
        total["preserve_mse"] += float(pmse.item()) * bs
        total["preserve_cos"] += float(pcos.item()) * bs
        total["norm"] += float(nloss.item()) * bs
        total["pred_to_target_cos"] += float(F.cosine_similarity(pred.float().flatten(1), shikra_tokens.float().flatten(1), dim=-1).mean().item()) * bs
        total["pred_to_base_cos"] += float(F.cosine_similarity(pred.float().flatten(1), base.float().flatten(1), dim=-1).mean().item()) * bs
        n += bs
    return {k: v / max(1, n) for k, v in total.items()}


def save_ckpt(path, model, optimizer, scheduler, args, step, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "step": step,
        "args": vars(args),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"

    image_indices, captions, train_rows, val_rows = make_train_val_indices(args)
    metadata = {
        "args": vars(args),
        "num_pairs": int(len(captions)),
        "num_train_rows": int(len(train_rows)),
        "num_val_rows": int(len(val_rows)),
        "trainable_note": "Shikra, OpenCLIP-bigG, and CLIP-L are frozen; gradients flow through Shikra to adapter tokens.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.shikra_path, use_fast=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    shikra = ShikraLlamaForCausalLM.from_pretrained(
        args.shikra_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    shikra.to(device=device).eval().requires_grad_(False)
    shikra.config.use_cache = False
    shikra.generation_config.pad_token_id = tokenizer.pad_token_id
    shikra.initialize_vision_tokenizer(True, tokenizer, device=device)

    bigg_version = args.bigg_ckpt if args.bigg_ckpt and Path(args.bigg_ckpt).exists() else "laion2b_s39b_b160k"
    bigg = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version=bigg_version,
        output_tokens=True,
        only_tokens=True,
        cache_dir=args.bigg_cache_dir,
    ).to(device)
    bigg.eval().requires_grad_(False)

    clip_l = CLIPVisionModel.from_pretrained(args.clip_l_path, torch_dtype=torch.float16)
    clip_l.to(device).eval().requires_grad_(False)

    model, frozen_base = make_adapter(args, device)
    if args.mode == "adapter":
        frozen_base = BigGToShikraAdapter().to(device).eval().requires_grad_(False)
        frozen_base.load_state_dict(load_adapter_state(args.init_adapter), strict=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)) * max(0.0, 1.0 - step / max(1, args.max_steps)),
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(json.dumps({"resumed": args.resume, "start_step": start_step}), flush=True)

    best_val = None
    rng = np.random.default_rng(args.seed + start_step)
    start_time = time.time()
    with h5py.File(args.hdf5_path, "r") as f:
        images_h5 = f["images"]
        for step in range(start_step, args.max_steps):
            model.train()
            batch_rows = rng.choice(train_rows, size=args.batch_size, replace=False)
            img_idx = image_indices[batch_rows]
            cap = captions[batch_rows]
            image = load_image_batch(images_h5, img_idx, device)
            input_ids, attention_mask, labels = collate_text(tokenizer, cap, args.question, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                bigg_tokens, shikra_tokens = compute_vision_tokens(image, bigg, clip_l)
                if args.mode == "residual":
                    pred, base, delta, gate = model(bigg_tokens)
                else:
                    pred = model(bigg_tokens)
                    with torch.no_grad():
                        base = frozen_base(bigg_tokens)
                    delta = pred - base
                    gate = torch.ones(pred.shape[:-1] + (1,), device=pred.device, dtype=pred.dtype)
                out = shikra(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_features=pred.to(dtype=torch.float16),
                    use_cache=False,
                )
                tmse, tcos, pmse, pcos, nloss = loss_terms(pred, shikra_tokens, base)
                loss = (
                    args.ce_weight * out.loss.float()
                    + args.target_mse_weight * tmse
                    + args.target_cos_weight * tcos
                    + args.preserve_mse_weight * pmse
                    + args.preserve_cos_weight * pcos
                    + args.norm_weight * nloss
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            global_step = step + 1
            if global_step % args.log_every == 0 or global_step == 1:
                rec = {
                    "step": global_step,
                    "train/loss": float(loss.item()),
                    "train/ce": float(out.loss.float().item()),
                    "train/target_mse": float(tmse.item()),
                    "train/target_cos_loss": float(tcos.item()),
                    "train/preserve_mse": float(pmse.item()),
                    "train/preserve_cos_loss": float(pcos.item()),
                    "train/norm": float(nloss.item()),
                    "train/pred_to_target_cos": float(
                        F.cosine_similarity(pred.float().flatten(1), shikra_tokens.float().flatten(1), dim=-1).mean().item()
                    ),
                    "train/pred_to_base_cos": float(
                        F.cosine_similarity(pred.float().flatten(1), base.float().flatten(1), dim=-1).mean().item()
                    ),
                    "train/delta_l2": float(delta.float().flatten(1).norm(dim=1).mean().item()),
                    "train/gate_mean": float(gate.float().mean().item()),
                    "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": time.time() - start_time,
                    "gpu_mem_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0,
                }
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(json.dumps(rec) + "\n")

            if global_step % args.eval_every == 0 or global_step == args.max_steps:
                metrics = evaluate(
                    args,
                    model,
                    frozen_base,
                    images_h5,
                    image_indices,
                    captions,
                    val_rows,
                    tokenizer,
                    shikra,
                    bigg,
                    clip_l,
                    device,
                )
                rec = {"step": global_step, **{f"val/{k}": v for k, v in metrics.items()}}
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(json.dumps(rec) + "\n")
                if best_val is None or metrics["loss"] < best_val:
                    best_val = metrics["loss"]
                    save_ckpt(out_dir / "best.pth", model, optimizer, scheduler, args, global_step, metrics)

            if global_step % args.save_every == 0 or global_step == args.max_steps:
                save_ckpt(out_dir / "last.pth", model, optimizer, scheduler, args, global_step, {"last_loss": float(loss.item())})

    print(json.dumps({"finished": True, "out_dir": str(out_dir), "best_val": best_val}), flush=True)


if __name__ == "__main__":
    main()
