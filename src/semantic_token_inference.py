#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer


class BigGToLConverter(nn.Module):
    """MindEye2 official 256x1664 OpenCLIP-bigG tokens -> 257x1024 CLIP-L/GIT tokens."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 257)
        self.linear2 = nn.Linear(1664, 1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


class BigGToShikraAdapter(nn.Module):
    """Shikra-specific 256x1664 OpenCLIP-bigG tokens -> 256x1024 CLIP-L patch tokens."""

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
    parser = argparse.ArgumentParser("Run semantic inference from MindEye2 visual tokens.")
    parser.add_argument(
        "--selected_path",
        default=os.environ.get("SELECTED_CONDITIONS", "outputs/evals/selected_conditions.pt"),
        help="A .pt file containing prior_tokens and target_tokens with shape Bx256x1664.",
    )
    parser.add_argument(
        "--converter_ckpt",
        default=os.environ.get("CONVERTER_CKPT", "outputs/bigG_to_shikra_adapter/last.pth"),
        help="MindEye2 official bigG_to_L converter checkpoint.",
    )
    parser.add_argument(
        "--converter_type",
        choices=["mindeye2_git257", "shikra256"],
        default="mindeye2_git257",
        help="Which bigG-token adapter to load.",
    )
    parser.add_argument(
        "--out_json",
        default=None,
        help="Where to write generated captions/answers. Defaults next to selected_path.",
    )
    parser.add_argument(
        "--save_tokens",
        default=None,
        help="Optional .pt output for converted 257x1024 and 256x1024 tokens.",
    )
    parser.add_argument("--mode", choices=["git257", "shikra256", "both", "convert"], default="both")
    parser.add_argument("--token_key", choices=["prior_tokens", "target_tokens", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git_model", default="microsoft/git-large-coco")
    parser.add_argument("--shikra_path", default=os.environ.get("SHIKRA_PATH", "shikra-7b-v1-0708"))
    parser.add_argument("--question", default="Describe the image briefly.")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_converter(path, device, converter_type):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if converter_type == "mindeye2_git257":
        model = BigGToLConverter()
    elif converter_type == "shikra256":
        model = BigGToShikraAdapter()
    else:
        raise ValueError(f"Unknown converter_type: {converter_type}")
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    return model


def convert_token_dict(selected, converter, device, token_key, num_samples, converter_type):
    keys = ["prior_tokens", "target_tokens"] if token_key == "both" else [token_key]
    converted = {}
    with torch.no_grad():
        for key in keys:
            if key not in selected:
                continue
            bigg = selected[key][:num_samples].to(device=device, dtype=torch.float32)
            mapped = converter(bigg).detach().cpu()
            if converter_type == "mindeye2_git257":
                converted[key] = {
                    "bigg_256x1664": bigg.detach().cpu(),
                    "clip_l_257x1024": mapped,
                    "clip_l_patches_256x1024": mapped[:, 1:, :].contiguous(),
                }
            elif converter_type == "shikra256":
                converted[key] = {
                    "bigg_256x1664": bigg.detach().cpu(),
                    "clip_l_patches_256x1024": mapped,
                }
    return converted


def run_git_257(converted, git_model, device):
    from modeling_git import GitForCausalLMClipEmb

    processor = AutoProcessor.from_pretrained(git_model)
    model = GitForCausalLMClipEmb.from_pretrained(git_model)
    model.to(device=device).eval().requires_grad_(False)

    out = {}
    with torch.no_grad():
        for key, values in converted.items():
            tokens = values["clip_l_257x1024"].to(device=device, dtype=torch.float32)
            generated_ids = model.generate(pixel_values=tokens, max_length=20)
            out[key] = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return out


def add_shikra_to_path():
    root = Path(os.environ.get("SHIKRA_REPO", Path(__file__).resolve().parents[1] / "third_party" / "shikra-main"))
    if not root.exists():
        raise FileNotFoundError(f"Shikra repo not found: {root}")
    sys.path.insert(0, str(root))


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
    return prompt, input_ids


def run_shikra_256(converted, shikra_path, question, max_new_tokens, temperature, device):
    add_shikra_to_path()
    from mllm.models.shikra import ShikraLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(shikra_path, use_fast=False)
    model = ShikraLlamaForCausalLM.from_pretrained(
        shikra_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device=device).eval().requires_grad_(False)
    model.initialize_vision_tokenizer(True, tokenizer, device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.unk_token_id

    _, input_ids = build_shikra_prompt(tokenizer, question)
    input_ids = input_ids.to(device)

    out = {}
    with torch.no_grad():
        for key, values in converted.items():
            answers = []
            tokens = values["clip_l_patches_256x1024"].to(device=device, dtype=torch.float16)
            for i in range(tokens.shape[0]):
                kwargs = {
                    "input_ids": input_ids,
                    "image_features": tokens[i : i + 1],
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "use_cache": True,
                }
                if temperature > 0:
                    kwargs["temperature"] = temperature
                generated = model.generate(**kwargs)
                answer = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
                answers.append(answer)
            out[key] = answers
    return out


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    selected_path = Path(args.selected_path)
    out_json = Path(args.out_json) if args.out_json else selected_path.with_name("semantic_256x1024_outputs.json")
    save_tokens = Path(args.save_tokens) if args.save_tokens else selected_path.with_name("converted_semantic_tokens.pt")

    selected = torch.load(selected_path, map_location="cpu")
    converter = load_converter(args.converter_ckpt, device, args.converter_type)
    converted = convert_token_dict(selected, converter, device, args.token_key, args.num_samples, args.converter_type)
    torch.save(converted, save_tokens)

    results = {
        "selected_path": str(selected_path),
        "converter_ckpt": args.converter_ckpt,
        "converter_type": args.converter_type,
        "saved_tokens": str(save_tokens),
        "question": args.question,
        "num_samples": args.num_samples,
        "shapes": {
            key: {name: list(tensor.shape) for name, tensor in values.items()}
            for key, values in converted.items()
        },
        "git257": {},
        "shikra256": {},
    }

    if args.mode in ("git257", "both"):
        results["git257"] = run_git_257(converted, args.git_model, device)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if args.mode in ("shikra256", "both"):
        results["shikra256"] = run_shikra_256(
            converted,
            args.shikra_path,
            args.question,
            args.max_new_tokens,
            args.temperature,
            device,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
