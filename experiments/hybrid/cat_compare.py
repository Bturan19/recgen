"""Category head comparison: flat CE vs hyperbolic (Poincaré ball).

Trains the hybrid pipeline on a given cache with the category head swapped,
reports test accuracy + hierarchical closeness (pred shares GT's parent path).

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
     uv run --no-sync python -u experiments/hybrid/cat_compare.py \
       --cache .cache/hybrid/trendyol_blind [--head flat|hyper] [--epochs 15]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from moderation.data import load, stratified_split

from hybrid.data import attr_labels, attr_schema, category_labels, tag_matrix
from hybrid.hyperbolic import FlatCategoryHead, HyperbolicCategoryHead
from hybrid.train import BATCH, DIM, POS_WEIGHT, SEED, W, collate, load_cache

import json

CACHE_DIR = ".cache/hybrid/trendyol"


def cat_head_variant(head, dim, n_cats):
    if head == "hyper":
        return HyperbolicCategoryHead(dim, n_cats)
    return FlatCategoryHead(dim, n_cats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE_DIR)
    ap.add_argument("--head", choices=["flat", "hyper"], default="flat")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    global CACHE
    CACHE = args.cache

    df = load()
    y = df["eval_decision"].to_numpy()
    y_cat, cat_vocab = category_labels(df)
    attr_y, attr_mask, attr_keys = attr_labels(df, attr_schema(df))
    attr_groups = [len(v) for v in attr_schema(df).values()]
    y_tag = tag_matrix(df)
    arr, lens, _, _, meta = load_cache()
    tr, va, te = stratified_split(df)
    n_cats = y_cat.max() + 1
    print(f"cache={args.cache} {arr.shape} head={args.head} train={len(tr)} val={len(va)} test={len(te)}")

    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from hybrid.heads import CrossAttnQueryBlock, ModerationHead, AttributeHeads, TagHead

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = CrossAttnQueryBlock(DIM, n_queries=4)
            self.cat_head = cat_head_variant(args.head, DIM, n_cats)
            self.mod_head = ModerationHead(DIM)
            self.attr_heads = AttributeHeads(DIM, attr_groups)
            self.tag_head = TagHead(DIM, 23)

        def forward(self, hidden, mask):
            q, _ = self.queries(hidden, mask)
            cat_out = self.cat_head(q[:, 1])
            if args.head == "hyper":
                dist, _ = cat_out
                cat_logits = -dist
            else:
                cat_logits, _ = cat_out
            return {
                "mod": self.mod_head(q[:, 0]),
                "cat": cat_logits,
                "attrs": self.attr_heads(q[:, 2]),
                "tags": self.tag_head(q[:, 3]),
            }

    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    mod_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
    cat_fn = nn.CrossEntropyLoss()
    tag_fn = nn.BCEWithLogitsLoss()

    def one_step(b):
        hid, mask = collate(b, arr, lens, device)
        out = model(hid, mask)
        loss = W["mod"] * mod_fn(out["mod"], torch.from_numpy(y[b]).float().to(device))
        loss += W["cat"] * cat_fn(out["cat"], torch.from_numpy(y_cat[b]).to(device))
        for k, lg in enumerate(out["attrs"]):
            m = attr_mask[b, k] == 1
            if m.sum() == 0:
                continue
            loss += W["attr"] * cat_fn(lg[m], torch.from_numpy(attr_y[b, k, : attr_groups[k]].argmax(-1)[m]).to(device))
        tm = y[b] == 1
        if tm.sum():
            loss += W["tags"] * tag_fn(out["tags"][tm], torch.from_numpy(y_tag[b][tm]).to(device))
        return loss

    best_acc, best_state, patience, t0 = -1.0, None, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = tr[perm[i : i + BATCH]].tolist()
            opt.zero_grad()
            try:
                loss = one_step(b)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            except Exception as e:
                print(f"  !! {type(e).__name__}: {e}", flush=True)
                opt.zero_grad()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(va), BATCH * 4):
                b = va[i : i + BATCH * 4].tolist()
                hid, mask = collate(b, arr, lens, device)
                preds.append(model(hid, mask)["cat"].argmax(-1).cpu().numpy())
        pva = np.concatenate(preds)
        va_acc = accuracy_score(y_cat[va], pva)
        print(f"  ep{ep + 1} loss={tot / max(len(perm) // BATCH, 1):.3f} val cat_acc={va_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(te), BATCH * 4):
            b = te[i : i + BATCH * 4].tolist()
            hid, mask = collate(b, arr, lens, device)
            preds.append(model(hid, mask)["cat"].argmax(-1).cpu().numpy())
    pte = np.concatenate(preds)
    acc = accuracy_score(y_cat[te], pte)
    print(f"TEST cat_acc ({args.head}, {os.path.basename(args.cache)}): {acc:.4f}")

    # hierarchical closeness: does pred share the GT's parent path?
    hier = [""] * len(cat_vocab)
    rows = df.to_dicts()
    for i, r in enumerate(rows):
        hier[y_cat[i]] = str(r["CategoryHierarchy"])
    close = 0
    for i, p in zip(te, pte):
        gt_path = hier[y_cat[i]]
        if p == y_cat[i]:
            close += 1
            continue
        pred_path = hier[p]
        gt_parts = [s.strip() for s in gt_path.split(">")]
        pred_parts = [s.strip() for s in pred_path.split(">")]
        shared = len(set(gt_parts) & set(pred_parts))
        close += 1 if shared >= max(1, len(gt_parts) - 1) else 0
    print(f"leaf-or-parent-correct: {close}/{len(te)} = {close / len(te):.4f}")

    from common import record
    record(f"hybrid_{args.head}cat_{os.path.basename(args.cache)}", cat_acc=acc, hier=close / len(te))


if __name__ == "__main__":
    main()
