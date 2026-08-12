"""Marketplace product moderation — text-only baseline.

Verbalized product text -> frozen LLM embeddings -> ClassificationHead.
Reports accuracy, F1, PR-AUC. Run:
  uv run python experiments/moderation/text_baseline.py --backbone smol17
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from recgen import ClassificationHead, FrozenEncoder

from common import record
from data import load, stratified_split, verbalize

BACKBONES = {
    "smol360": "models/SmolLM2-360M",
    "smol17": "models/SmolLM2-1.7B",
}


def main(backbone: str = "smol17", batch_size: int = 32, epochs: int = 40):
    model_dir = BACKBONES[backbone]
    cache_dir = f".cache/moderation/{backbone}"

    df = load()
    rows = df.to_dicts()
    texts = [verbalize(r) for r in rows]
    y = df["eval_decision"].to_numpy()
    print(f"rows={len(rows)} rejected={y.mean():.3f}")

    tr_idx, va_idx, te_idx = stratified_split(df)

    encoder = FrozenEncoder(model_dir, pooling="mean", batch_size=batch_size, max_length=1024)
    H = encoder.encode_cached(texts, f"{cache_dir}/text_emb.npy")

    for tag, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        print(f"{tag}: {len(idx)} (rejected {y[idx].mean():.3f})")

    head = ClassificationHead(hidden=(256, 128), epochs=epochs, batch_size=128, patience=6, random_state=0)
    head.fit(H[tr_idx], y[tr_idx])
    p = head.predict(H[te_idx])
    proba = head.predict_proba(H[te_idx])[:, 1]

    acc = accuracy_score(y[te_idx], p)
    f1 = f1_score(y[te_idx], p)
    auc = roc_auc_score(y[te_idx], proba)
    prec, rec, _ = precision_recall_curve(y[te_idx], proba)
    pr_auc = float(np.trapezoid(rec, prec))
    print(f"text[{backbone}]: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} pr_auc={pr_auc:.4f}")
    record(f"moderation_text_{backbone}", acc=acc, f1=f1, auc=auc, pr_auc=pr_auc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="smol17", choices=list(BACKBONES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    main(args.backbone, batch_size=args.batch_size, epochs=args.epochs)
