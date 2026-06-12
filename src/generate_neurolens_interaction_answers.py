#!/usr/bin/env python
"""Generate open-ended interaction answers for NeuroLens evaluation.

The protocol mirrors BrainSeek-style semantic evaluation:

  reference: ground-truth stimulus image -> frozen Shikra -> answer
  candidate: fMRI-derived 1024 visual tokens -> frozen Shikra -> answer

The script can also evaluate image-based baselines, e.g. reconstructed images
fed into the same frozen Shikra model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel


DEFAULT_QUESTIONS = [
    {
        "id": "Q1_scene",
        "question": "Provide a general description of the perceived scene.",
        "category": "general scene understanding",
    },
    {
        "id": "Q2_objects_spatial",
        "question": "Specify the number and spatial arrangement of key objects.",
        "category": "object-level details and spatial arrangement",
    },
    {
        "id": "Q3_reasoning",
        "question": "What potential activities could be happening based on the scene?",
        "category": "contextual reasoning",
    },
]


def parse_named_path(item):
    if ":" not in item:
        raise argparse.ArgumentTypeError(f"expected NAME:PATH, got {item!r}")
    name, path = item.split(":", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME:PATH, got {item!r}")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser("Generate NeuroLens interaction answers")
    parser.add_argument("--shikra_path", default=os.environ.get("SHIKRA_PATH", "shikra-7b-v1-0708"))
    parser.add_argument(
        "--image_processor_path",
        default=os.environ.get("CLIP_L_PROCESSOR_PATH", os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14")),
        help="CLIP image processor path used for ground-truth/reconstruction image references.",
    )
    parser.add_argument(
        "--image_encoder_path",
        default=os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14"),
        help="CLIP-L image encoder path used to extract 256x1024 Shikra-compatible image tokens.",
    )
    parser.add_argument("--reference_images", default=os.environ.get("REFERENCE_IMAGES", "outputs/evals/all_images.pt"))
    parser.add_argument(
        "--candidate_tokens",
        action="append",
        type=parse_named_path,
        default=[],
        help="Candidate token set as NAME:PATH. May be repeated.",
    )
    parser.add_argument(
        "--candidate_images",
        action="append",
        type=parse_named_path,
        default=[],
        help="Candidate image set as NAME:PATH. May be repeated.",
    )
    parser.add_argument("--questions_json", default="")
    parser.add_argument("--out_dir", default=os.environ.get("INTERACTION_ROOT", "outputs/interaction_eval"))
    parser.add_argument("--num_eval", type=int, default=0, help="<=0 means all examples")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_shikra_to_path():
    root = Path(os.environ.get("SHIKRA_REPO", Path(__file__).resolve().parents[1] / "third_party" / "shikra-main"))
    if not root.exists():
        raise FileNotFoundError(f"Shikra repo not found: {root}")
    sys.path.insert(0, str(root))


def load_questions(path):
    if not path:
        return DEFAULT_QUESTIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    for item in questions:
        if "id" not in item or "question" not in item:
            raise ValueError("Each question must contain `id` and `question`.")
    return questions


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


def tensor_to_pil_batch(x):
    to_pil = transforms.ToPILImage()
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected image tensor [N,3,H,W], got {tuple(x.shape)}")
    x = x.detach().cpu().float().clamp(0, 1)
    return [to_pil(img) for img in x]


def load_image_tensor(path, n):
    images = torch.load(path, map_location="cpu")
    if isinstance(images, (list, tuple)):
        images = torch.stack(images)
    if not torch.is_tensor(images):
        raise TypeError(f"Unsupported image payload at {path}: {type(images)}")
    if n > 0:
        images = images[:n]
    return images.float()


def load_token_tensor(path, n):
    tokens = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tokens):
        raise TypeError(f"Unsupported token payload at {path}: {type(tokens)}")
    if n > 0:
        tokens = tokens[:n]
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 1024):
        raise ValueError(f"Expected token tensor [N,256,1024], got {tuple(tokens.shape)} from {path}")
    return tokens.float()


@torch.no_grad()
def image_to_clip_tokens(vision_model, processor, image, device):
    pil = tensor_to_pil_batch(image.unsqueeze(0))[0]
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(device=device, dtype=torch.float16)
    out = vision_model(pixel_values, output_hidden_states=True)
    return out.hidden_states[-2][:, 1:].float()


@torch.no_grad()
def generate_from_image(model, vision_model, processor, input_ids, image, device, max_new_tokens, temperature):
    image_features = image_to_clip_tokens(vision_model, processor, image, device).to(device=device, dtype=torch.float16)
    gen_kwargs = {
        "input_ids": input_ids,
        "image_features": image_features,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    generated = model.generate(**gen_kwargs)
    return generated


@torch.no_grad()
def generate_from_token(model, input_ids, token, device, max_new_tokens, temperature):
    image_features = token.unsqueeze(0).to(device=device, dtype=torch.float16)
    gen_kwargs = {
        "input_ids": input_ids,
        "image_features": image_features,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    generated = model.generate(**gen_kwargs)
    return generated


def decode_new_tokens(tokenizer, generated, input_len):
    return tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    add_shikra_to_path()
    from mllm.models.shikra import ShikraLlamaForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions_json)

    tokenizer = AutoTokenizer.from_pretrained(args.shikra_path, use_fast=False)
    model = ShikraLlamaForCausalLM.from_pretrained(
        args.shikra_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device=device).eval().requires_grad_(False)
    model.initialize_vision_tokenizer(True, tokenizer, device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.unk_token_id

    image_processor = CLIPImageProcessor.from_pretrained(args.image_processor_path)
    vision_model = CLIPVisionModel.from_pretrained(args.image_encoder_path)
    vision_model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)

    n = args.num_eval
    ref_images = load_image_tensor(args.reference_images, n)
    n_loaded = len(ref_images)
    token_sets = [(name, load_token_tensor(path, n)) for name, path in args.candidate_tokens]
    image_sets = [(name, load_image_tensor(path, n)) for name, path in args.candidate_images]
    for name, tokens in token_sets:
        if len(tokens) != n_loaded:
            raise ValueError(f"{name} has {len(tokens)} examples, reference has {n_loaded}")
    for name, images in image_sets:
        if len(images) != n_loaded:
            raise ValueError(f"{name} has {len(images)} examples, reference has {n_loaded}")

    results = {
        "metadata": {
            "shikra_path": args.shikra_path,
            "image_processor_path": args.image_processor_path,
            "image_encoder_path": args.image_encoder_path,
            "reference_images": args.reference_images,
            "candidate_tokens": {name: str(path) for name, path in args.candidate_tokens},
            "candidate_images": {name: str(path) for name, path in args.candidate_images},
            "num_eval": n_loaded,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "questions": [],
    }

    for q in questions:
        qid = q["id"]
        input_ids = build_shikra_prompt(tokenizer, q["question"]).to(device)
        input_len = input_ids.shape[1]
        q_out = {
            "id": qid,
            "question": q["question"],
            "category": q.get("category", ""),
            "reference": [],
            "candidates": {name: [] for name, _ in token_sets + image_sets},
        }
        for idx in tqdm(range(n_loaded), desc=qid):
            ref_gen = generate_from_image(
                model,
                vision_model,
                image_processor,
                input_ids,
                ref_images[idx],
                device,
                args.max_new_tokens,
                args.temperature,
            )
            q_out["reference"].append(decode_new_tokens(tokenizer, ref_gen, input_len))
            for name, tokens in token_sets:
                cand_gen = generate_from_token(
                    model,
                    input_ids,
                    tokens[idx],
                    device,
                    args.max_new_tokens,
                    args.temperature,
                )
                q_out["candidates"][name].append(decode_new_tokens(tokenizer, cand_gen, input_len))
            for name, images in image_sets:
                cand_gen = generate_from_image(
                    model,
                    vision_model,
                    image_processor,
                    input_ids,
                    images[idx],
                    device,
                    args.max_new_tokens,
                    args.temperature,
                )
                q_out["candidates"][name].append(decode_new_tokens(tokenizer, cand_gen, input_len))

            if (idx + 1) % 25 == 0:
                tmp = {"metadata": results["metadata"], "questions": results["questions"] + [q_out]}
                (out_dir / "interaction_answers.partial.json").write_text(
                    json.dumps(tmp, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        results["questions"].append(q_out)
        (out_dir / "interaction_answers.partial.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    out_path = out_dir / "interaction_answers.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
