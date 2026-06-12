#!/usr/bin/env python
"""Generate Shikra prompts from a trained 1664/1024 interaction head.

This keeps the Experiment-A layout. It uses the original MindEye2 base
reconstructions for image generation, but replaces prompt tokens with:

    prior_out 1664 -> frozen COCO bridge 1024 -> trained gated fusion 1664
                   -> frozen COCO bridge 1024 -> Shikra text prompt

The goal is to test whether the trained 1024/1664 interaction improves prompt
guidance without changing the main MindEye2 reconstruction path.
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


class SemanticToVisualGatedFusion(nn.Module):
    def __init__(
        self,
        bigg_dim=1664,
        shikra_dim=1024,
        hidden_dim=2048,
        gate_hidden_dim=512,
        max_gate=0.05,
        init_gate_bias=-4.0,
    ):
        super().__init__()
        self.max_gate = max_gate
        self.semantic_to_delta = nn.Sequential(
            nn.LayerNorm(shikra_dim),
            nn.GELU(),
            nn.Linear(shikra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bigg_dim),
        )
        self.bigg_norm = nn.LayerNorm(bigg_dim)
        self.delta_norm = nn.LayerNorm(bigg_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(bigg_dim * 2),
            nn.Linear(bigg_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    def forward(self, bigg_tokens, shikra_tokens):
        delta = self.semantic_to_delta(shikra_tokens)
        gate_in = torch.cat((self.bigg_norm(bigg_tokens), self.delta_norm(delta)), dim=-1)
        gate = torch.sigmoid(self.gate(gate_in)) * self.max_gate
        fused = bigg_tokens + gate * delta
        return fused, delta, gate


def parse_args():
    parser = argparse.ArgumentParser("Experiment A interaction-head Shikra prompt generation")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--base_name", default="base_official_prior")
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--interaction_ckpt", required=True)
    parser.add_argument(
        "--bridge_adapter_ckpt",
        default="",
        help="Defaults to the bridge path stored inside the interaction checkpoint.",
    )
    parser.add_argument("--token_suffix", default="all_prior_out", choices=["all_prior_out", "all_backbones", "all_clipvoxels"])
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


def load_bridge(path, device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    bridge = BigGToShikraCocoAdapter()
    bridge.load_state_dict(state, strict=True)
    bridge.eval().requires_grad_(False).to(device)
    return bridge


def load_fusion(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("args", {})
    fusion = SemanticToVisualGatedFusion(
        hidden_dim=int(cfg.get("fusion_hidden_dim", 2048)),
        gate_hidden_dim=int(cfg.get("fusion_gate_hidden_dim", 512)),
        max_gate=float(cfg.get("fusion_max_gate", 0.05)),
        init_gate_bias=float(cfg.get("fusion_init_gate_bias", -4.0)),
    )
    if "fusion" in ckpt:
        state = ckpt["fusion"]
    else:
        raw = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        state = {
            key.removeprefix("fusion."): val
            for key, val in raw.items()
            if key.removeprefix("module.").startswith("fusion.")
        }
        state = {key.removeprefix("module.").removeprefix("fusion."): val for key, val in state.items()}
    fusion.load_state_dict(state, strict=True)
    fusion.eval().requires_grad_(False).to(device)
    return fusion, cfg


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


def main():
    args = parse_args()
    add_shikra_to_path()
    from mllm.models.shikra import ShikraLlamaForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    eval_root = Path(args.eval_root)
    base_dir = eval_root / args.base_name
    var_dir = eval_root / args.variant_name
    var_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = base_dir / f"{args.base_name}_{args.token_suffix}.pt"
    tokens = torch.load(tokens_path, map_location="cpu").float()
    if args.num_eval > 0:
        tokens = tokens[: args.num_eval]

    fusion, cfg = load_fusion(args.interaction_ckpt, device)
    bridge_path = args.bridge_adapter_ckpt or cfg.get("bridge_adapter_ckpt")
    if not bridge_path:
        raise ValueError("bridge adapter path must be supplied or stored in the interaction checkpoint")
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
    fused_bigg_chunks = []
    fused_shikra_chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(tokens), args.batch_size), desc=args.variant_name):
            batch_tokens = tokens[start : start + args.batch_size].to(device, dtype=torch.float32)
            shikra_anchor = bridge(batch_tokens)
            fused_bigg, delta, gate = fusion(batch_tokens, shikra_anchor)
            fused_shikra = bridge(fused_bigg).to(dtype=torch.float16)
            fused_bigg_chunks.append(fused_bigg.detach().cpu().float())
            fused_shikra_chunks.append(fused_shikra.detach().cpu())
            for i in range(fused_shikra.shape[0]):
                gen_kwargs = {
                    "input_ids": input_ids,
                    "image_features": fused_shikra[i : i + 1],
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

    fused_bigg_path = var_dir / f"{args.variant_name}_fused_bigg_tokens.pt"
    fused_shikra_path = var_dir / f"{args.variant_name}_converted_shikra_tokens.pt"
    torch.save(torch.cat(fused_bigg_chunks, dim=0), fused_bigg_path)
    torch.save(torch.cat(fused_shikra_chunks, dim=0), fused_shikra_path)
    copy_base_eval_files(eval_root, args.base_name, args.variant_name, captions)
    metadata = {
        "variant_name": args.variant_name,
        "base_name": args.base_name,
        "interaction_ckpt": args.interaction_ckpt,
        "bridge_adapter_ckpt": str(bridge_path),
        "token_suffix": args.token_suffix,
        "shikra_path": args.shikra_path,
        "question": args.question,
        "num_captions": len(captions),
        "fused_bigg_tokens": str(fused_bigg_path),
        "converted_tokens": str(fused_shikra_path),
        "fusion_max_gate": cfg.get("fusion_max_gate"),
    }
    (var_dir / "prompt_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (var_dir / "captions.json").write_text(json.dumps({"captions": captions}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
