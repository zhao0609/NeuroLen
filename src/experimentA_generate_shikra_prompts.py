#!/usr/bin/env python
"""Generate Shikra captions from exported MindEye2 prior tokens.

The output directory is arranged like a MindEye2 eval folder so that the same
enhanced reconstruction and evaluation code can consume each prompt variant.
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


class BigGToShikraAdapter(nn.Module):
    """Bx256x1664 -> Bx256x1024 Shikra CLIP-L patch tokens."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(1664, 1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        return self.linear2(x.permute(0, 2, 1))


def parse_args():
    parser = argparse.ArgumentParser("Experiment A Shikra prompt generation")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--base_name", default="base_official_prior")
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--adapter_ckpt", required=True)
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


def load_adapter(path, device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model = BigGToShikraAdapter()
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model


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
    common = [
        "all_recons",
        "all_blurryrecons",
        "all_clipvoxels",
        "all_backbones",
        "all_prior_out",
    ]
    for suffix in common:
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

    prior_path = base_dir / f"{args.base_name}_all_prior_out.pt"
    prior = torch.load(prior_path, map_location="cpu").float()
    if args.num_eval > 0:
        prior = prior[: args.num_eval]

    adapter = load_adapter(args.adapter_ckpt, device)
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
    token_path = var_dir / f"{args.variant_name}_converted_shikra_tokens.pt"
    converted_chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(prior), args.batch_size), desc=args.variant_name):
            tokens = prior[start : start + args.batch_size].to(device, dtype=torch.float32)
            mapped = adapter(tokens).to(dtype=torch.float16)
            converted_chunks.append(mapped.detach().cpu())
            for i in range(mapped.shape[0]):
                gen_kwargs = {
                    "input_ids": input_ids,
                    "image_features": mapped[i : i + 1],
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

    torch.save(torch.cat(converted_chunks, dim=0), token_path)
    copy_base_eval_files(eval_root, args.base_name, args.variant_name, captions)
    metadata = {
        "variant_name": args.variant_name,
        "base_name": args.base_name,
        "adapter_ckpt": args.adapter_ckpt,
        "shikra_path": args.shikra_path,
        "question": args.question,
        "num_captions": len(captions),
        "converted_tokens": str(token_path),
    }
    (var_dir / "prompt_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (var_dir / "captions.json").write_text(json.dumps({"captions": captions}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
