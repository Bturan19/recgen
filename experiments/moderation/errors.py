"""Moderation error analysis: where does a trained head fail vs Gemini's eval?

Loads test predictions (saved by the head) and groups errors by rejection
tag / category / text signals. Run after text_baseline.py / multimodal.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from data import load, stratified_split


def main(preds_path: str):
    df = load()
    y = df["eval_decision"].to_numpy()
    tags = df["eval_rejection_tag"].to_list()
    cats = df["CategoryHierarchy"].to_list()
    tr_idx, va_idx, te_idx = stratified_split(df)
    preds = np.load(preds_path)
    assert len(preds) == len(te_idx)

    yt = y[te_idx]
    acc = (preds == yt).mean()
    print(f"test acc: {acc:.4f}")

    # false negatives (missed rejections) by tag
    fn = np.where((yt == 1) & (preds == 0))[0]
    fp = np.where((yt == 0) & (preds == 1))[0]
    print(f"\nfalse negatives (missed rejections): {len(fn)} of {int(yt.sum())}")
    print("missed-by-tag:")
    from collections import Counter

    cnt = Counter()
    for i in fn:
        for t in tags[te_idx[i]]:
            if t:
                cnt[t] += 1
    for t, c in cnt.most_common(10):
        print(f"  {c:4d}  {t}")
    print(f"\nfalse positives (wrongly rejected): {len(fp)} of {int((yt == 0).sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".cache/moderation/test_preds.npy")
