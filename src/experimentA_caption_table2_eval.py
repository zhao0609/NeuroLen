#!/usr/bin/env python
"""Caption-only Table-2 style evaluation for Experiment A variants."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel


def parse_args():
    parser = argparse.ArgumentParser("Caption Table-2 style evaluation")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--tables_root", default=os.environ.get("TABLES_ROOT", "outputs/tables"))
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--num_eval", type=int, default=0, help="<=0 means all.")
    parser.add_argument("--sentence_model", default=os.environ.get("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--clip_b_text_model", default=os.environ.get("CLIP_B_PATH", "openai/clip-vit-base-patch32"))
    parser.add_argument(
        "--clip_l_text_model",
        default=os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def load_texts(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    return [str(x) for x in obj]


def normalize_lengths(*seqs, n=0):
    length = min(len(seq) for seq in seqs)
    if n and n > 0:
        length = min(length, n)
    return [list(seq[:length]) for seq in seqs]


def meteor_mean(preds, refs):
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for pred, ref in tqdm(list(zip(preds, refs)), desc="meteor", leave=False):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        scores.append(meteor_score([ref_tokens], pred_tokens))
    return float(np.mean(scores))


def rouge_mean(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1, rouge_l = [], []
    for pred, ref in tqdm(list(zip(preds, refs)), desc="rouge", leave=False):
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)
    return float(np.mean(rouge1)), float(np.mean(rouge_l))


@torch.no_grad()
def sentence_scores(preds, refs_by_name, model_path, device, batch_size):
    model = SentenceTransformer(model_path, device=device)
    pred_emb = model.encode(preds, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
    out = {}
    for name, refs in refs_by_name.items():
        ref_emb = model.encode(refs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
        out[name] = float(util.pytorch_cos_sim(pred_emb, ref_emb).diag().mean().item())
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def clip_text_scores(preds, refs_by_name, model_path, device, batch_size):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    def encode(texts):
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"clip {model_path.name}", leave=False):
            batch = texts[start : start + batch_size]
            tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = model.get_text_features(**tokenized)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu())
        return torch.cat(chunks, dim=0)

    pred_emb = encode(preds)
    out = {}
    for name, refs in refs_by_name.items():
        ref_emb = encode(refs)
        out[name] = float((pred_emb * ref_emb).sum(dim=-1).mean().item())
    del model
    torch.cuda.empty_cache()
    return out


def evaluate_one(args, model_name, refs_coco, refs_git, device):
    pred_path = Path(args.eval_root) / model_name / f"{model_name}_all_predcaptions.pt"
    preds = load_texts(pred_path)
    preds, coco, git = normalize_lengths(preds, refs_coco, refs_git, n=args.num_eval)
    refs_by_name = {"COCO captions": coco, "GIT captions": git}

    rows = []
    for ref_name, refs in refs_by_name.items():
        meteor = meteor_mean(preds, refs)
        rouge1, rouge_l = rouge_mean(preds, refs)
        rows.extend(
            [
                {"Reference": ref_name, "Metric": "METEOR", "Value": meteor},
                {"Reference": ref_name, "Metric": "ROUGE-L", "Value": rouge_l},
                {"Reference": ref_name, "Metric": "ROUGE-1", "Value": rouge1},
            ]
        )

    sent = sentence_scores(preds, refs_by_name, args.sentence_model, device, args.batch_size)
    clip_b = clip_text_scores(preds, refs_by_name, args.clip_b_text_model, device, args.batch_size)
    clip_l = clip_text_scores(preds, refs_by_name, args.clip_l_text_model, device, args.batch_size)

    for ref_name in refs_by_name:
        rows.extend(
            [
                {"Reference": ref_name, "Metric": "Sentence", "Value": sent[ref_name]},
                {"Reference": ref_name, "Metric": "CLIP-B", "Value": clip_b[ref_name]},
                {"Reference": ref_name, "Metric": "CLIP-L", "Value": clip_l[ref_name]},
            ]
        )

    df = pd.DataFrame(rows)
    tables_root = Path(args.tables_root)
    tables_root.mkdir(parents=True, exist_ok=True)
    csv_path = tables_root / f"{model_name}_caption_table2_style.csv"
    json_path = tables_root / f"{model_name}_caption_table2_style.json"
    df.to_csv(csv_path, sep="\t", index=False)
    json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "num_eval": len(preds),
                "pred_path": str(pred_path),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[{model_name}] n={len(preds)}")
    print(df.to_string(index=False))
    print(f"wrote {csv_path}")


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    eval_root = Path(args.eval_root)
    refs_coco = load_texts(eval_root / "all_captions.pt")
    refs_git = load_texts(eval_root / "all_git_generated_captions.pt")
    for model_name in args.model_names:
        evaluate_one(args, model_name, refs_coco, refs_git, device)


if __name__ == "__main__":
    main()
