#!/usr/bin/env python
"""Evaluate open-ended interaction answers with BERTScore."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("Evaluate NeuroLens interaction answers with BERTScore")
    parser.add_argument("--answers_json", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_type", default="roberta-large")
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_bertscore():
    local = Path(os.environ.get("BERT_SCORE_REPO", ""))
    if local.exists():
        sys.path.insert(0, str(local))
    from bert_score import BERTScorer

    return BERTScorer


def clean(text):
    text = "" if text is None else str(text)
    return " ".join(text.replace("\n", " ").split()).strip() or "."


def main():
    args = parse_args()
    BERTScorer = load_bertscore()
    answers_path = Path(args.answers_json)
    out_dir = Path(args.out_dir) if args.out_dir else answers_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(answers_path.read_text(encoding="utf-8"))
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    scorer = BERTScorer(
        model_type=args.model_type,
        num_layers=args.num_layers,
        lang="en",
        batch_size=args.batch_size,
        rescale_with_baseline=False,
        device=device,
    )

    rows = []
    per_sample = []
    for q in data["questions"]:
        refs = [clean(x) for x in q["reference"]]
        for method, cands_raw in q["candidates"].items():
            cands = [clean(x) for x in cands_raw]
            P, R, F1 = scorer.score(cands, refs)
            row = {
                "question_id": q["id"],
                "category": q.get("category", ""),
                "method": method,
                "BERT-P": float(P.mean().item()),
                "BERT-R": float(R.mean().item()),
                "BERT-F1": float(F1.mean().item()),
                "BERT-P_std": float(P.std(unbiased=False).item()),
                "BERT-R_std": float(R.std(unbiased=False).item()),
                "BERT-F1_std": float(F1.std(unbiased=False).item()),
                "n": len(cands),
            }
            rows.append(row)
            for i, (p, r, f) in enumerate(zip(P.tolist(), R.tolist(), F1.tolist())):
                per_sample.append(
                    {
                        "question_id": q["id"],
                        "method": method,
                        "index": i,
                        "BERT-P": float(p),
                        "BERT-R": float(r),
                        "BERT-F1": float(f),
                        "reference": refs[i],
                        "candidate": cands[i],
                    }
                )

    methods = sorted({row["method"] for row in rows})
    for method in methods:
        vals = [row for row in rows if row["method"] == method]
        rows.append(
            {
                "question_id": "Average",
                "category": "average over questions",
                "method": method,
                "BERT-P": float(np.mean([v["BERT-P"] for v in vals])),
                "BERT-R": float(np.mean([v["BERT-R"] for v in vals])),
                "BERT-F1": float(np.mean([v["BERT-F1"] for v in vals])),
                "BERT-P_std": float(np.mean([v["BERT-P_std"] for v in vals])),
                "BERT-R_std": float(np.mean([v["BERT-R_std"] for v in vals])),
                "BERT-F1_std": float(np.mean([v["BERT-F1_std"] for v in vals])),
                "n": vals[0]["n"] if vals else 0,
            }
        )

    metrics = {"metadata": data.get("metadata", {}), "bertscore_model": args.model_type, "rows": rows}
    (out_dir / "interaction_bertscore_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "question_id",
        "category",
        "method",
        "BERT-P",
        "BERT-R",
        "BERT-F1",
        "BERT-P_std",
        "BERT-R_std",
        "BERT-F1_std",
        "n",
    ]
    with (out_dir / "interaction_bertscore_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "interaction_bertscore_samples.json").open("w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)

    md = [
        "# NeuroLens Interaction BERTScore",
        "",
        f"- Answers: `{answers_path}`",
        f"- BERTScore model: `{args.model_type}`",
        "",
        "| Question | Method | BERT-P | BERT-R | BERT-F1 | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['question_id']} | {row['method']} | {row['BERT-P']:.4f} | {row['BERT-R']:.4f} | {row['BERT-F1']:.4f} | {row['n']} |"
        )
    (out_dir / "interaction_bertscore_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "interaction_bertscore_metrics.md")


if __name__ == "__main__":
    main()
