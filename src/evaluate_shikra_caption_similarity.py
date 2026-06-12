#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Shikra decoded captions against MindEye2 reference captions.")
    parser.add_argument("--semantic_json", required=True)
    parser.add_argument("--ref_captions", default=os.environ.get("REF_CAPTIONS", "outputs/evals/all_captions.pt"))
    parser.add_argument("--sentence_model", default=os.environ.get("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_refs(path, n):
    refs = torch.load(path, map_location="cpu")
    if isinstance(refs, np.ndarray):
        refs = refs.tolist()
    refs = [str(x) for x in refs[:n]]
    return refs


def diag_cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def rank_metrics(query_emb, ref_emb):
    sims = query_emb @ ref_emb.T
    ranks = []
    for i in range(sims.shape[0]):
        order = torch.argsort(sims[i], descending=True)
        rank = int((order == i).nonzero(as_tuple=False)[0, 0].item()) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks)
    return {
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "median_rank": float(ranks_t.float().median().item()),
        "mean_rank": float(ranks_t.float().mean().item()),
        "ranks": ranks,
    }


def main():
    args = parse_args()
    semantic_json = Path(args.semantic_json)
    out_dir = Path(args.out_dir) if args.out_dir else semantic_json.parent / "caption_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(semantic_json.read_text(encoding="utf-8"))
    prior = data["shikra256"]["prior_tokens"]
    target = data["shikra256"]["target_tokens"]
    n = min(len(prior), len(target))
    refs = load_refs(args.ref_captions, n)

    device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    model = SentenceTransformer(args.sentence_model, device=device)
    with torch.no_grad():
        prior_emb = model.encode(prior[:n], convert_to_tensor=True, normalize_embeddings=True)
        target_emb = model.encode(target[:n], convert_to_tensor=True, normalize_embeddings=True)
        ref_emb = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)

    prior_ref = diag_cos(prior_emb, ref_emb).detach().cpu()
    target_ref = diag_cos(target_emb, ref_emb).detach().cpu()
    prior_target = diag_cos(prior_emb, target_emb).detach().cpu()

    metrics = {
        "semantic_json": str(semantic_json),
        "ref_captions": args.ref_captions,
        "sentence_model": args.sentence_model,
        "n": n,
        "prior_vs_ref_mean": float(prior_ref.mean().item()),
        "prior_vs_ref_std": float(prior_ref.std().item()),
        "target_vs_ref_mean": float(target_ref.mean().item()),
        "target_vs_ref_std": float(target_ref.std().item()),
        "prior_vs_target_mean": float(prior_target.mean().item()),
        "prior_vs_target_std": float(prior_target.std().item()),
        "relative_prior_ref_over_target_ref": float((prior_ref.mean() / target_ref.mean()).item()),
        "prior_to_ref_retrieval": rank_metrics(prior_emb, ref_emb),
        "target_to_ref_retrieval": rank_metrics(target_emb, ref_emb),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for i in range(n):
        rows.append(
            {
                "idx": i,
                "prior_vs_ref": float(prior_ref[i].item()),
                "target_vs_ref": float(target_ref[i].item()),
                "prior_vs_target": float(prior_target[i].item()),
                "ref_caption": refs[i],
                "prior_caption": prior[i],
                "target_token_caption": target[i],
            }
        )

    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "prior_vs_ref",
                "target_vs_ref",
                "prior_vs_target",
                "ref_caption",
                "prior_caption",
                "target_token_caption",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Shikra Caption Similarity",
        "",
        f"- n: {n}",
        f"- prior vs real caption: {metrics['prior_vs_ref_mean']:.4f} +/- {metrics['prior_vs_ref_std']:.4f}",
        f"- target-token vs real caption: {metrics['target_vs_ref_mean']:.4f} +/- {metrics['target_vs_ref_std']:.4f}",
        f"- prior vs target-token caption: {metrics['prior_vs_target_mean']:.4f} +/- {metrics['prior_vs_target_std']:.4f}",
        f"- relative prior/ref over target/ref: {metrics['relative_prior_ref_over_target_ref']:.4f}",
        f"- prior->real caption retrieval top1/top5: {metrics['prior_to_ref_retrieval']['top1']:.4f}/{metrics['prior_to_ref_retrieval']['top5']:.4f}",
        f"- target->real caption retrieval top1/top5: {metrics['target_to_ref_retrieval']['top1']:.4f}/{metrics['target_to_ref_retrieval']['top5']:.4f}",
        "",
        "| idx | prior-ref | target-ref | prior-target | real caption | brain caption |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        real = row["ref_caption"].replace("|", "\\|").replace("\n", " ")
        brain = row["prior_caption"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['idx']} | {row['prior_vs_ref']:.3f} | {row['target_vs_ref']:.3f} | "
            f"{row['prior_vs_target']:.3f} | {real} | {brain} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
