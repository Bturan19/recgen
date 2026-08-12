"""Marketplace product moderation — multimodal (text + image embeddings).

Text: frozen SmolLM2-1.7B mean-pooled embeddings (reused from text_baseline).
Images: frozen SigLIP (400M) mean-pooled patch embeddings, averaged over up
to 4 images per product. Concat [text, img] -> ClassificationHead.

Run: uv run python experiments/moderation/multimodal.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from transformers import AutoProcessor, AutoModel

from recgen import ClassificationHead, FrozenEncoder

from common import record
from data import IMG_DIR, image_paths, load, stratified_split, verbalize

TEXT_CACHE = ".cache/moderation/smol17/text_emb.npy"
IMG_CACHE = ".cache/moderation/siglip/img_emb.npy"
SIGLIP = "google/siglip2-so400m-patch14-384"
MAX_IMGS = 4


def load_siglip(device="mps"):
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP).to(device).eval()
    return proc, model


def encode_images(paths, proc, model, device, batch=8):
    """Mean-pooled patch embeddings per image."""
    if not paths:
        return None
    embs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        imgs = []
        for p in chunk:
            from PIL import Image

            imgs.append(Image.open(p).convert("RGB"))
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        h = out.pooler_output.float()  # (B, D) pooled feature
        embs.append(h.cpu().numpy())
    return np.concatenate(embs, axis=0)


def main(epochs: int = 50, batch_size: int = 32):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tr_idx, va_idx, te_idx = stratified_split(df)

    H_text = np.load(TEXT_CACHE)
    print(f"text embeddings: {H_text.shape}")

    proc, model = load_siglip(device)
    img_emb = np.load(IMG_CACHE) if os.path.exists(IMG_CACHE) else None
    if img_emb is None:
        paths_per_product = [image_paths(r["ProductId"], MAX_IMGS) for r in rows]
        img_emb = np.zeros((len(rows), 1152), dtype=np.float32)
        n_with_img = 0
        for j, paths in enumerate(paths_per_product):
            if paths:
                e = encode_images(paths, proc, model, device)
                img_emb[j] = e.mean(axis=0)
                n_with_img += 1
            if (j + 1) % 200 == 0:
                print(f"  encoded images {j + 1}/{len(rows)} ({n_with_img} with images)")
        print(f"images encoded: {n_with_img}/{len(rows)} products with images")
        os.makedirs(os.path.dirname(IMG_CACHE), exist_ok=True)
        np.save(IMG_CACHE, img_emb)

    print("=== ablations ===")
    for name, X in [
        ("text-only", H_text),
        ("img-only", img_emb),
        ("text+img", np.concatenate([H_text, img_emb], axis=1)),
    ]:
        head = ClassificationHead(hidden=(256, 128), epochs=epochs, batch_size=128, patience=6, random_state=0)
        head.fit(X[tr_idx], y[tr_idx])
        p = head.predict(X[te_idx])
        proba = head.predict_proba(X[te_idx])[:, 1]
        acc = accuracy_score(y[te_idx], p)
        f1 = f1_score(y[te_idx], p)
        auc = roc_auc_score(y[te_idx], proba)
        prec, rec, _ = precision_recall_curve(y[te_idx], proba)
        pr_auc = float(np.trapezoid(rec, prec))
        print(f"{name}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} pr_auc={pr_auc:.4f}")
        record(f"moderation_{name.replace('+', '_')}", acc=acc, f1=f1, auc=auc, pr_auc=pr_auc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
