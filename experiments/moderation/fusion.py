"""Moderation v2: text + OCR + image embeddings fused by a head.

- H_text: SmolLM2-1.7B mean-pooled embeddings of product text (cached)
- H_ocr: SmolLM2-1.7B embeddings of OCR text extracted from images (short,
  fast to encode; cache)
- I_img: SigLIP2 pooled image embeddings (cached)
Concat -> ClassificationHead. Reports acc/F1/AUC/PR-AUC with val-tuned
threshold.

Run: uv run python experiments/moderation/fusion.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from recgen import ClassificationHead, FrozenEncoder

from common import record
from data import load, stratified_split

OCR_INSTRUCTION = "Extract the facts visible in the product images (brand names, model, specs, labels, text on packaging)."


def main(backbone: str = "smol17", epochs: int = 50):
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tr_idx, va_idx, te_idx = stratified_split(df)

    ocr_cache = f".cache/moderation/{backbone}/ocr_emb.npy"
    ocr_texts = np.load(".cache/moderation/ocr_text.npy", allow_pickle=True)
    encoder = FrozenEncoder(f"models/SmolLM2-1.7B", pooling="mean", batch_size=32, max_length=512)
    texts = [f"{OCR_INSTRUCTION}\n{t}" if t else OCR_INSTRUCTION for t in ocr_texts]
    H_ocr = encoder.encode_cached(texts, ocr_cache)

    H_text = np.load(f".cache/moderation/{backbone}/text_emb.npy")
    I_img = np.load(".cache/moderation/siglip/img_emb.npy")

    for name, X in [
        ("text+ocr", np.concatenate([H_text, H_ocr], axis=1)),
        ("text+ocr+img", np.concatenate([H_text, H_ocr, I_img], axis=1)),
    ]:
        head = ClassificationHead(hidden=(256, 128), epochs=epochs, batch_size=128, patience=6, random_state=0)
        head.fit(X[tr_idx], y[tr_idx])
        pva = head.predict_proba(X[va_idx])[:, 1]
        pte = head.predict_proba(X[te_idx])[:, 1]
        prec, rec, th = precision_recall_curve(y[va_idx], pva)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_th = th[f1s.argmax()] if f1s.argmax() < len(th) else 0.5
        for tag, t in [("0.5", 0.5), ("tuned", best_th)]:
            p = (pte >= t).astype(int)
            acc = accuracy_score(y[te_idx], p)
            f1 = f1_score(y[te_idx], p)
            auc = roc_auc_score(y[te_idx], pte)
            print(f"{name} @th={tag}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
            if tag == "tuned":
                record(f"moderation_{name.replace('+', '_')}", acc=acc, f1=f1, auc=auc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="smol17")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    main(args.backbone, epochs=args.epochs)
