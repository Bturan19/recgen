"""Marketplace product-moderation data: verbalization + splits.

Task: predict eval_decision (Onaylandı / Reddedildi) from product text
(title, category, brand, description, attributes) and images.
Labels were produced by a Gemini-3.1-flash-lite eval run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import polars as pl

DATA = "data/qwen_dataset/sample_4k.parquet"
IMG_DIR = "data/qwen_images"
SEED = 0

INSTRUCTION = "Given a product listing, decide whether it should be approved or rejected for the marketplace."


def load(parquet: str = DATA):
    df = pl.read_parquet(parquet)
    df = df.with_columns(pl.col("eval_decision").map_elements(lambda d: 1 if d == "Reddedildi" else 0, return_dtype=pl.Int8))
    return df


def parse_attributes(attrs_json):
    if not attrs_json:
        return []
    try:
        attrs = json.loads(attrs_json)
    except Exception:
        return []
    if not isinstance(attrs, list):
        return []
    out = []
    for a in attrs:
        if isinstance(a, dict):
            k = a.get("key") or a.get("name")
            v = a.get("value")
            if k and v is not None:
                out.append(f"{k}: {v}")
    return out


def verbalize(row: dict, max_desc_chars: int = 700) -> str:
    parts = []
    if row.get("DisplayName"):
        parts.append(f"Title: {row['DisplayName']}")
    if row.get("BrandName"):
        parts.append(f"Brand: {row['BrandName']}")
    if row.get("CategoryHierarchy"):
        parts.append(f"Category: {row['CategoryHierarchy']}")
    attrs = parse_attributes(row.get("AttributesJson"))
    if attrs:
        parts.append("Attributes: " + " | ".join(attrs[:12]))
    desc = (row.get("Description") or "").strip()
    if desc:
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."
        parts.append(f"Description: {desc}")
    body = "\n".join(parts)
    return f"{INSTRUCTION}\n{body}"


def image_paths(product_id: str, max_imgs: int = 4) -> list[str]:
    d = os.path.join(IMG_DIR, str(product_id))
    if not os.path.isdir(d):
        return []
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    return [os.path.join(d, f) for f in files[:max_imgs]]


def stratified_split(df: pl.DataFrame, tr=0.8, va=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    y = df["eval_decision"].to_numpy()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_tr_p, n_va_p = int(len(pos) * tr), int(len(pos) * va)
    n_tr_n, n_va_n = int(len(neg) * tr), int(len(neg) * va)
    tr_idx = np.concatenate([pos[:n_tr_p], neg[:n_tr_n]])
    va_idx = np.concatenate([pos[n_tr_p : n_tr_p + n_va_p], neg[n_tr_n : n_tr_n + n_va_n]])
    te_idx = np.concatenate([pos[n_tr_p + n_va_p :], neg[n_tr_n + n_va_n :]])
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    rng.shuffle(te_idx)
    return tr_idx, va_idx, te_idx
