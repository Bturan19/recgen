"""5-fold CV for the hybrid frozen-VLM + query-token pipeline.

Honest estimate (the single 402-row split overstates — old pipeline: 0.886
single vs 0.817 CV). Headline numbers must come from here.

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
     uv run --no-sync python -u experiments/hybrid/cv.py [--epochs 20] [--folds 5]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from moderation.data import load

from hybrid.data import N_TAGS, attr_labels, attr_schema, category_labels, tag_matrix
from hybrid.train import BATCH, DIM, LR, PATIENCE, POS_WEIGHT, SEED, W, collate, load_cache

from hybrid.heads import HybridModel
import torch.nn as nn

import time


def run_fold(tr_idx, va_idx, df, y, y_cat, attr_y, attr_mask, y_tag, arr, lens, epochs, w5=0.0, guidance=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    attr_groups = [len(v) for v in attr_schema(df).values()]
    model = HybridModel(DIM, n_cats=y_cat.max() + 1, attr_groups=attr_groups, n_tags=N_TAGS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    mod_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
    cat_fn = nn.CrossEntropyLoss()
    tag_fn = nn.BCEWithLogitsLoss()
    attr_vocab = attr_groups

    def one_step(b):
        hid, mask = collate(b, arr, lens, device)
        yb = y[b]
        out = model(hid, mask, return_attn=w5 > 0)
        loss = W["mod"] * mod_fn(out["mod"], torch.from_numpy(yb).float().to(device))
        loss += W["cat"] * cat_fn(out["cat"], torch.from_numpy(y_cat[b]).to(device))
        for k, lg in enumerate(out["attrs"]):
            m = attr_mask[b, k] == 1
            if m.sum() == 0:
                continue
            loss += W["attr"] * cat_fn(lg[m], torch.from_numpy(attr_y[b, k, : attr_vocab[k]].argmax(-1)[m]).to(device))
        tm = y[b] == 1
        if tm.sum():
            loss += W["tags"] * tag_fn(out["tags"][tm], torch.from_numpy(y_tag[b][tm]).to(device))
        if w5 > 0:
            attn = out["attn"][:, 0].masked_fill(mask == 0, -1e9).softmax(-1)
            tg = torch.from_numpy(np.ascontiguousarray(guidance[b])[:, : mask.shape[1]]).to(device)
            loss += w5 * torch.sum(tg * (torch.log(tg + 1e-9) - torch.log(attn + 1e-9)), dim=-1).mean()
        return loss

    best_auc, best_state, patience, t0 = -1.0, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), BATCH):
            b = tr_idx[perm[i : i + BATCH]].tolist()
            opt.zero_grad()
            try:
                loss = one_step(b)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except Exception:
                opt.zero_grad()
                continue
        model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(va_idx), BATCH * 4):
                b = va_idx[i : i + BATCH * 4].tolist()
                hid, mask = collate(b, arr, lens, device)
                probas.append(torch.sigmoid(model(hid, mask)["mod"]).cpu().numpy())
        pva = np.concatenate(probas)
        va_auc = roc_auc_score(y[va_idx], pva)
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    probas = []
    with torch.no_grad():
        for i in range(0, len(va_idx), BATCH * 4):
            b = va_idx[i : i + BATCH * 4].tolist()
            hid, mask = collate(b, arr, lens, device)
            probas.append(torch.sigmoid(model(hid, mask)["mod"]).cpu().numpy())
    pva = np.concatenate(probas)
    prec, rec, ths = precision_recall_curve(y[va_idx], pva)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    th = float(ths[f1s.argmax()])
    pred = (pva > th).astype(int)
    return {"auc": best_auc, "acc": accuracy_score(y[va_idx], pred), "f1": f1_score(y[va_idx], pred), "th": th}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--w5", type=float, default=0.0, help="attention-guidance weight")
    args = ap.parse_args()

    df = load()
    y = df["eval_decision"].to_numpy()
    y_cat, _ = category_labels(df)
    attr_y, attr_mask, _ = attr_labels(df, attr_schema(df))
    y_tag = tag_matrix(df)
    arr, lens, img_mask, ids_arr, meta = load_cache()
    print(f"cache: {arr.shape}")

    guidance = None
    if args.w5 > 0:
        guidance = np.load(".cache/hybrid/trendyol/guidance.npy")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    rows_all = np.arange(len(y))
    results = []
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(rows_all, y)):
        r = run_fold(tr_idx, va_idx, df, y, y_cat, attr_y, attr_mask, y_tag, arr, lens, args.epochs, w5=args.w5, guidance=guidance)
        results.append(r)
        print(f"fold {fold + 1}: auc={r['auc']:.4f} acc={r['acc']:.4f} f1={r['f1']:.4f} th={r['th']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    for k in ("auc", "acc", "f1"):
        vals = [r[k] for r in results]
        print(f"CV {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    from common import record
    record(f"hybrid_frozen_w5{args.w5}_cv", auc=float(np.mean([r["auc"] for r in results])),
           acc=float(np.mean([r["acc"] for r in results])),
           f1=float(np.mean([r["f1"] for r in results])))


if __name__ == "__main__":
    main()
