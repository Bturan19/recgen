"""Evaluate the frozen hybrid's category / attribute heads + correction signals.

Answers:
  1. Query tokens vs mean-pooled probes: does q_cat/q_attr beat a pooled head
     on the same cached hidden states (category 645-CE, grouped attrs)?
  2. Category correction signal: when the head disagrees with the seller's
     declared category, is Gemini's eval_incorrect_category more likely?
  3. Attribute correction signal: when the head disagrees with the listed
     value, is eval_incorrect_attribute more likely?

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
     uv run --no-sync python -u experiments/hybrid/eval_heads.py [--epochs 15]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from moderation.data import load, stratified_split

from hybrid.data import attr_labels, attr_schema, category_labels, tag_matrix
from hybrid.heads import HybridModel
from hybrid.train import (DIM, POS_WEIGHT, SEED, W, collate, load_cache)

BATCH = 32


def mean_pool_baseline(tr_idx, te_idx, df, y_cat, attr_y, attr_mask, attr_groups, arr, lens):
    """Old-pipeline-style: MLP probe on mean-pooled hidden states."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pooled = np.zeros((len(df), DIM), dtype=np.float32)
    for i in range(len(df)):
        t = int(lens[i])
        pooled[i] = arr[i, :t].mean(axis=0)

    cat_probe = nn.Sequential(nn.Linear(DIM, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, y_cat.max() + 1)).to(device)
    cat_opt = torch.optim.AdamW(cat_probe.parameters(), lr=1e-4)
    cat_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(pooled).to(device)
    yc = torch.from_numpy(y_cat).to(device)
    rng = np.random.default_rng(SEED)
    for ep in range(15):
        cat_probe.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), 64):
            b = tr_idx[perm[i : i + 64]]
            cat_opt.zero_grad()
            loss = cat_fn(cat_probe(X[b]), yc[b])
            loss.backward()
            cat_opt.step()
    cat_probe.eval()
    with torch.no_grad():
        pte = cat_probe(X[te_idx]).argmax(-1).cpu().numpy()
    cat_acc = accuracy_score(y_cat[te_idx], pte)

    shared = nn.Sequential(nn.Linear(DIM, 256), nn.GELU(), nn.Dropout(0.2)).to(device)
    heads = nn.ModuleList([nn.Linear(256, v).to(device) for v in attr_groups])
    attr_opt = torch.optim.AdamW(list(shared.parameters()) + list(heads.parameters()), lr=1e-4)
    ay = torch.from_numpy(attr_y).to(device)
    am = torch.from_numpy(attr_mask).to(device)
    for ep in range(15):
        shared.train()
        for h in heads:
            h.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), 64):
            b = tr_idx[perm[i : i + 64]]
            attr_opt.zero_grad()
            hsh = shared(X[b])
            loss = torch.tensor(0.0, device=device)
            for k, hd in enumerate(heads):
                m = am[b, k] == 1
                if m.sum() == 0:
                    continue
                loss += W["attr"] * cat_fn(hd(hsh[m]), ay[b, k, : attr_groups[k]].argmax(-1)[m])
            loss.backward()
            attr_opt.step()
    shared.eval()
    for h in heads:
        h.eval()
    with torch.no_grad():
        hsh = shared(X[te_idx])
    n_hit = n_tot = 0
    per_key = {}
    for k, hd in enumerate(heads):
        m = am[te_idx, k] == 1
        if m.sum() == 0:
            continue
        pred = hd(hsh[m]).argmax(-1).cpu().numpy()
        gt = ay[te_idx, k, : attr_groups[k]].argmax(-1)[m].cpu().numpy()
        hit = (pred == gt).sum()
        n_hit += hit
        n_tot += len(gt)
        per_key[k] = hit / len(gt)
    attr_acc = n_hit / max(n_tot, 1)
    return cat_acc, attr_acc, per_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--cache", default=".cache/hybrid/trendyol", help="hidden-state cache dir")
    args = ap.parse_args()

    import train as train_mod
    train_mod.CACHE = args.cache

    df = load()
    y = df["eval_decision"].to_numpy()
    y_cat, cat_vocab = category_labels(df)
    attr_y, attr_mask, attr_keys = attr_labels(df, attr_schema(df))
    attr_groups = [len(v) for v in attr_schema(df).values()]
    y_tag = tag_matrix(df)
    arr, lens, _, _, meta = load_cache()
    tr, va, te = stratified_split(df)
    print(f"cache={args.cache} {arr.shape}, train={len(tr)} val={len(va)} test={len(te)}")

    # ---------- 1. hybrid query-token heads ----------
    from hybrid.train import train_model
    model, best = train_model(tr, va, df, y, y_cat, attr_y, attr_mask, y_tag, arr, lens,
                              epochs=args.epochs, tag="eval_heads")
    device = next(model.parameters()).device
    model.eval()
    mod_pts, cat_pts, attr_preds = [], [], {k: [] for k in range(len(attr_keys))}
    with torch.no_grad():
        for i in range(0, len(te), BATCH * 4):
            b = te[i : i + BATCH * 4].tolist()
            hid, mask = collate(b, arr, lens, device)
            out = model(hid, mask)
            mod_pts.append(torch.sigmoid(out["mod"]).cpu().numpy())
            cat_pts.append(out["cat"].argmax(-1).cpu().numpy())
            for k, lg in enumerate(out["attrs"]):
                attr_preds[k].append(lg.argmax(-1).cpu().numpy())
    mod_pts = np.concatenate(mod_pts)
    cat_pts = np.concatenate(cat_pts)
    cat_acc = accuracy_score(y_cat[te], cat_pts)
    print(f"\n== hybrid query-token heads (test) ==")
    print(f"category acc (645, seller-declared GT): {cat_acc:.4f}")

    n_hit = n_tot = 0
    print("attr per-key acc (present keys only):")
    for k in range(len(attr_keys)):
        m = attr_mask[te, k] == 1
        if m.sum() == 0:
            print(f"  {attr_keys[k]:12s}: n/a")
            continue
        pred = np.concatenate(attr_preds[k])
        gt = attr_y[te, k, : attr_groups[k]].argmax(-1)[m]
        acc = (pred[m] == gt).mean()
        n_hit += (pred[m] == gt).sum()
        n_tot += m.sum()
        print(f"  {attr_keys[k]:12s}: {acc:.4f} (n={m.sum()})")
    print(f"attr overall: {n_hit / max(n_tot, 1):.4f}")

    # ---------- 2. mean-pool baselines ----------
    print("\n== mean-pool probes (old-pipeline style, same cache) ==")
    bp_cat, bp_attr, bp_keys = mean_pool_baseline(tr, te, df, y_cat, attr_y, attr_mask, attr_groups, arr, lens)
    print(f"category acc: {bp_cat:.4f}")
    print(f"attr per-key: { {attr_keys[k]: round(v, 4) for k, v in bp_keys.items()} }")
    print(f"attr overall: {bp_attr:.4f}")

    # ---------- 3. category correction signal ----------
    print("\n== category correction signal (test) ==")
    inc = df["eval_incorrect_category"].to_numpy()
    inc_te = (inc[te] == "Evet").astype(int)
    disagree = (cat_pts != y_cat[te]).astype(int)
    print(f"head-seller disagreement rate: {disagree.mean():.3f}")
    if disagree.sum() > 0:
        p_wrong_given_disagree = inc_te[disagree == 1].mean()
        p_wrong_given_agree = inc_te[disagree == 0].mean()
        print(f"P(Gemini says category WRONG | head disagrees) = {p_wrong_given_disagree:.3f}")
        print(f"P(Gemini says category WRONG | head agrees)    = {p_wrong_given_agree:.3f}")
    # how many incorrect rows does disagreement catch?
    tp = (disagree == 1) & (inc_te == 1)
    print(f"incorrect rows caught by disagreement: {tp.sum()}/{inc_te.sum()}")
    # Gemini suggested category vs head prediction for incorrect rows (token level)
    sugg = df["eval_suggested_category"].to_list()
    cat_names_all = df["CategoryName"].to_list()
    sugg_te = [sugg[i] for i in te]

    def token_overlap(a, b):
        sa, sb = set(str(a).lower().replace(",", " ").split()), set(str(b).lower().split())
        return len(sa & sb)

    n_ov_pred = n_ov_seller = n_sugg = 0
    for i, (s, cp, cg) in enumerate(zip(sugg_te, cat_pts, inc_te)):
        if s in ("N/A",):
            continue
        n_sugg += 1
        n_ov_pred += token_overlap(cat_vocab[cp], s) > 0
        n_ov_seller += token_overlap(str(cat_names_all[te[i]]), s) > 0
    print(f"head-pred shares token with Gemini suggestion: {n_ov_pred}/{n_sugg}")
    print(f"seller-cat shares token with Gemini suggestion: {n_ov_seller}/{n_sugg}")

    # ---------- 4. attribute correction signal ----------
    print("\n== attribute correction signal (test) ==")
    inc_attr = df["eval_incorrect_attribute"].to_numpy()
    inc_attr_te = (inc_attr[te] == "Evet").astype(int)
    rows = df.to_dicts()
    dis_flags, agr_flags = [], []
    for row_pos, i in enumerate(te):
        r = rows[i]
        from moderation.data import parse_attributes
        listed = dict(kv.split(": ", 1) for kv in parse_attributes(r["AttributesJson"]))
        for k, key in enumerate(attr_keys):
            v_gt = listed.get(key)
            if v_gt is None:
                continue
            vocab = attr_schema(df)[key]
            if v_gt not in vocab:
                continue
            pred = np.concatenate(attr_preds[k])[row_pos]
            if pred == vocab.index(v_gt):
                agr_flags.append(inc_attr_te[row_pos])
            else:
                dis_flags.append(inc_attr_te[row_pos])
    dis_flags, agr_flags = np.array(dis_flags), np.array(agr_flags)
    print(f"attr disagreements: {len(dis_flags)} vs agreements: {len(agr_flags)}")
    if len(dis_flags):
        print(f"P(Gemini says attr WRONG | head disagrees w/ listing) = {dis_flags.mean():.3f}")
        print(f"P(Gemini says attr WRONG | head agrees w/ listing)    = {agr_flags.mean():.3f}")

    # reason-based: rows whose Gemini reason mentions key K — does the head
    # disagree with the listing more on those rows (i.e. is it catching the error)?
    print("\n== per-key error detection (reasons mention key) ==")
    reasons = df["eval_incorrect_attribute_reason"].to_list()
    for k, key in enumerate(attr_keys):
        vocab = attr_schema(df)[key]
        rows_flagged = []  # (row_pos in test, listed value in vocab)
        for row_pos, i in enumerate(te):
            r = rows[i]
            listed = dict(kv.split(": ", 1) for kv in parse_attributes(r["AttributesJson"]))
            v_gt = listed.get(key)
            if v_gt is None or v_gt not in vocab:
                continue
            if key.lower() in str(reasons[i]).lower():
                rows_flagged.append((row_pos, v_gt))
        if not rows_flagged:
            continue
        dis = sum(1 for rp, v in rows_flagged if np.concatenate(attr_preds[k])[rp] != vocab.index(v))
        print(f"  {key:12s}: head disagrees with listing on {dis}/{len(rows_flagged)} of reason-flagged rows "
              f"({dis / len(rows_flagged):.2f})")

if __name__ == "__main__":
    main()
