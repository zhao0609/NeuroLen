#!/usr/bin/env python
"""Generate Shikra captions from a safe 1024-token calibration checkpoint.

This script keeps the MindEye2 1664 image path unchanged. It only changes the
1024 Shikra token used for text generation:

    prior_1664 -> frozen COCO bridge -> base_1024

For a residual checkpoint:

    refined_1024 = base_1024 + small_gate * residual(base_1024, prior_1664)

For an adapter checkpoint:

    refined_1024 = trainable_bridge(prior_1664)

The output directory mirrors a MindEye2 eval folder so the existing prompt
enhancement and evaluation scripts can consume it.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer


class BigGToShikraCocoAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(1664, 1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        return self.linear2(x.permute(0, 2, 1))


class ShikraTokenResidualCalibrator(nn.Module):
    def __init__(
        self,
        bigg_dim=1664,
        shikra_dim=1024,
        hidden_dim=1536,
        gate_hidden_dim=384,
        max_gate=0.02,
        init_gate_bias=-5.0,
        dropout=0.05,
    ):
        super().__init__()
        self.max_gate = max_gate
        self.bigg_to_context = nn.Sequential(
            nn.LayerNorm(bigg_dim),
            nn.Linear(bigg_dim, shikra_dim),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(shikra_dim * 2),
            nn.Linear(shikra_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shikra_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(shikra_dim * 2),
            nn.Linear(shikra_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    def forward(self, bigg_tokens, shikra_base):
        bigg_context = self.bigg_to_context(bigg_tokens)
        joint = torch.cat((shikra_base, bigg_context), dim=-1)
        delta = self.delta(joint)
        gate = torch.sigmoid(self.gate(joint)) * self.max_gate
        refined = shikra_base + gate * delta
        return refined, delta, gate


class VLMCETrainingResidualAdapter(nn.Module):
    """Residual module layout saved by train_bigg_to_shikra_vlm_ce.py."""

    def __init__(self, max_gate=0.05, init_gate_bias=-4.0, dropout=0.05):
        super().__init__()
        self.base_adapter = BigGToShikraCocoAdapter().eval().requires_grad_(False)
        self.max_gate = max_gate
        self.bigg_context = nn.Sequential(
            nn.LayerNorm(1664),
            nn.Linear(1664, 1024),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1536),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(1536),
            nn.Linear(1536, 1024),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 384),
            nn.GELU(),
            nn.Linear(384, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    def forward(self, bigg_tokens, shikra_base=None):
        base = self.base_adapter(bigg_tokens)
        joint = torch.cat((base, self.bigg_context(bigg_tokens)), dim=-1)
        delta = self.delta(joint)
        gate = torch.sigmoid(self.gate(joint)) * self.max_gate
        refined = base + gate * delta
        return refined, base, delta, gate


def parse_args():
    parser = argparse.ArgumentParser("Generate calibrated Shikra prompts")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--base_name", default="base_official_prior")
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--calibrator_ckpt", required=True)
    parser.add_argument(
        "--bridge_adapter_ckpt",
        default="",
        help="Defaults to the bridge path stored in the calibrator checkpoint.",
    )
    parser.add_argument("--shikra_path", default=os.environ.get("SHIKRA_PATH", "shikra-7b-v1-0708"))
    parser.add_argument("--question", default="Describe the image briefly.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_eval", type=int, default=0, help="<=0 means all prior tokens.")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_shikra_to_path():
    root = Path(os.environ.get("SHIKRA_REPO", Path(__file__).resolve().parents[1] / "third_party" / "shikra-main"))
    if not root.exists():
        raise FileNotFoundError(f"Shikra repo not found: {root}")
    sys.path.insert(0, str(root))


def load_bridge_state(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    return {k.removeprefix("module."): v for k, v in payload.items()}


def load_bridge(path, device):
    bridge = BigGToShikraCocoAdapter()
    bridge.load_state_dict(load_bridge_state(path), strict=True)
    return bridge.eval().requires_grad_(False).to(device)


def load_calibrator(path, device):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    mode = ckpt.get("mode") or args.get("mode")
    if mode is None:
        mode = "residual" if "calibrator" in ckpt else "adapter"

    if "model_state_dict" in ckpt:
        if mode == "adapter":
            model = BigGToShikraCocoAdapter()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval().requires_grad_(False).to(device)
            return mode, model, ckpt
        if mode == "residual":
            model = VLMCETrainingResidualAdapter(
                max_gate=float(args.get("residual_max_gate", 0.05)),
                init_gate_bias=float(args.get("residual_init_gate_bias", -4.0)),
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval().requires_grad_(False).to(device)
            return "residual_vlmce", model, ckpt

    if "head_state_dict" in ckpt:
        ckpt_args = ckpt.get("args", {})
        model = VLMCETrainingResidualAdapter(
            max_gate=float(ckpt_args.get("residual_max_gate", 0.05)),
            init_gate_bias=float(ckpt_args.get("residual_init_gate_bias", -4.0)),
        )
        model.load_state_dict(ckpt["head_state_dict"], strict=True)
        model.eval().requires_grad_(False).to(device)
        return "residual_dual", model, ckpt

    if mode == "residual":
        model = ShikraTokenResidualCalibrator(
            hidden_dim=int(args.get("residual_hidden_dim", 1536)),
            gate_hidden_dim=int(args.get("residual_gate_hidden_dim", 384)),
            max_gate=float(args.get("residual_max_gate", 0.02)),
            init_gate_bias=float(args.get("residual_init_gate_bias", -5.0)),
        )
        model.load_state_dict(ckpt["calibrator"], strict=True)
        model.eval().requires_grad_(False).to(device)
        return mode, model, ckpt

    if mode == "adapter":
        model = BigGToShikraCocoAdapter()
        model.load_state_dict(ckpt["bridge_train"], strict=True)
        model.eval().requires_grad_(False).to(device)
        return mode, model, ckpt

    raise ValueError(f"unsupported checkpoint mode: {mode}")


def build_shikra_prompt(tokenizer, question):
    from mllm.conversation import get_conv_template

    conv = get_conv_template("vicuna_v1.1")
    image_tokens = "<im_start>" + "<im_patch>" * 256 + "<im_end>"
    conv.append_message(conv.roles[0], image_tokens + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    patch_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    n_patch = int((input_ids == patch_token_id).sum().item())
    if n_patch != 256:
        raise RuntimeError(f"Prompt tokenized to {n_patch} image patches, expected 256.")
    return input_ids


def copy_base_eval_files(eval_root, base_name, variant_name, captions):
    base_dir = eval_root / base_name
    var_dir = eval_root / variant_name
    var_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["all_recons", "all_blurryrecons", "all_clipvoxels", "all_backbones", "all_prior_out"]:
        src = base_dir / f"{base_name}_{suffix}.pt"
        dst = var_dir / f"{variant_name}_{suffix}.pt"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except FileExistsError:
                pass
            except OSError:
                shutil.copy2(src, dst)
    torch.save(np.array(captions), var_dir / f"{variant_name}_all_predcaptions.pt")


def token_stats(x):
    flat = x.float().flatten(1)
    return {
        "mean": float(x.float().mean().item()),
        "std": float(x.float().std().item()),
        "absmean": float(x.float().abs().mean().item()),
        "norm_mean": float(flat.norm(dim=1).mean().item()),
        "min": float(x.float().min().item()),
        "max": float(x.float().max().item()),
    }


def main():
    args = parse_args()
    add_shikra_to_path()
    from mllm.models.shikra import ShikraLlamaForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    eval_root = Path(args.eval_root)
    base_dir = eval_root / args.base_name
    var_dir = eval_root / args.variant_name
    var_dir.mkdir(parents=True, exist_ok=True)

    prior_path = base_dir / f"{args.base_name}_all_prior_out.pt"
    prior = torch.load(prior_path, map_location="cpu").float()
    if args.num_eval > 0:
        prior = prior[: args.num_eval]

    mode, calibrator, ckpt = load_calibrator(args.calibrator_ckpt, device)
    ckpt_args = ckpt.get("args", {})
    bridge_path = args.bridge_adapter_ckpt or ckpt.get("bridge_adapter_ckpt") or ckpt_args.get("init_adapter")
    if not bridge_path:
        raise ValueError("bridge adapter path must be supplied or stored in checkpoint")
    bridge = load_bridge(bridge_path, device)

    tokenizer = AutoTokenizer.from_pretrained(args.shikra_path, use_fast=False)
    model = ShikraLlamaForCausalLM.from_pretrained(
        args.shikra_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device=device).eval().requires_grad_(False)
    model.initialize_vision_tokenizer(True, tokenizer, device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.unk_token_id
    input_ids = build_shikra_prompt(tokenizer, args.question).to(device)

    captions = []
    base_chunks = []
    refined_chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(prior), args.batch_size), desc=args.variant_name):
            bigg = prior[start : start + args.batch_size].to(device, dtype=torch.float32)
            base_1024 = bridge(bigg)
            if mode.startswith("residual"):
                refined = calibrator(bigg, base_1024)[0]
            else:
                refined = calibrator(bigg)
            refined = refined.to(dtype=torch.float16)
            base_chunks.append(base_1024.detach().cpu().float())
            refined_chunks.append(refined.detach().cpu())
            for i in range(refined.shape[0]):
                gen_kwargs = {
                    "input_ids": input_ids,
                    "image_features": refined[i : i + 1],
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    gen_kwargs["temperature"] = args.temperature
                generated = model.generate(**gen_kwargs)
                answer = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
                captions.append(answer)
            if len(captions) % 25 == 0:
                torch.save(np.array(captions), var_dir / f"{args.variant_name}_all_predcaptions.partial.pt")
                (var_dir / "captions.partial.json").write_text(
                    json.dumps({"captions": captions}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    base_tokens = torch.cat(base_chunks, dim=0)
    refined_tokens = torch.cat(refined_chunks, dim=0)
    refined_path = var_dir / f"{args.variant_name}_converted_shikra_tokens.pt"
    base_path = var_dir / f"{args.variant_name}_base_shikra_tokens.pt"
    torch.save(refined_tokens, refined_path)
    torch.save(base_tokens, base_path)
    copy_base_eval_files(eval_root, args.base_name, args.variant_name, captions)

    flat_base = base_tokens.float().flatten(1)
    flat_refined = refined_tokens.float().flatten(1)
    diagnostics = {
        "base_stats": token_stats(base_tokens),
        "refined_stats": token_stats(refined_tokens),
        "refined_vs_base_cos": float(torch.nn.functional.cosine_similarity(flat_refined, flat_base, dim=-1).mean().item()),
        "refined_vs_base_mse": float(torch.nn.functional.mse_loss(refined_tokens.float(), base_tokens.float()).item()),
        "empty_captions": int(sum(1 for cap in captions if not str(cap).strip())),
        "unique_captions": int(len(set(str(cap) for cap in captions))),
    }
    metadata = {
        "variant_name": args.variant_name,
        "base_name": args.base_name,
        "mode": mode,
        "calibrator_ckpt": args.calibrator_ckpt,
        "bridge_adapter_ckpt": str(bridge_path),
        "shikra_path": args.shikra_path,
        "question": args.question,
        "num_captions": len(captions),
        "converted_tokens": str(refined_path),
        "base_tokens": str(base_path),
        "diagnostics": diagnostics,
    }
    (var_dir / "prompt_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (var_dir / "captions.json").write_text(json.dumps({"captions": captions}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
