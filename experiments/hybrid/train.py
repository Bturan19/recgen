"""Phase 1: frozen-VLM + learned query tokens + 4 heads (multi-task).

Trains ONLY the query block + heads on the cached hidden states
(.cache/hybrid/trendyol/), keeping the VLM frozen. This isolates the
query-token contribution vs the old mean-pooled head pipeline.

Loss: w1*BCE(mod, pos_weight) + w2*CE(cat) + w3*grouped-softmax(attr)
      + w4*BCE(tags, masked) [+ w5*KL(attn || guidance)]

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
     uv run --no-sync python -u experiments/hybrid/train.py [--epochs 30] [--w5 0]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from moderation.data import load, stratified_split

from hybrid.data import (attr_labels, attr_schema, build_guidance_targets, category_labels,
                         tag_matrix, N_TAGS)
from hybrid.heads import HybridModel

CACHE = ".cache/hybrid/trendyol"
DIM = 1024
POS_WEIGHT = 3.6  # 21.5% reject rate
W = {"mod": 1.0, "cat": 0.5, "attr": 0.25, "tags": 0.5, "guidance": 0.0}
LR = 1e-4
BATCH = 32
PATIENCE = 5
SEED = 0


def load_cache():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    arr = np.load(os.path.join(CACHE, "hidden.npy"), mmap_mode="r")
    lens = np.load(os.path.join(CACHE, "lens.npy"))
    img_mask = np.load(os.path.join(CACHE, "img_mask.npy"))
    ids = np.load(os.path.join(CACHE, "input_ids.npy"))
    return arr, lens, img_mask, ids, meta


def to_tensor(x, device):
    return torch.from_numpy(x).to(device)


def collate(indices, arr, lens, device):
    n = len(indices)
    T = int(lens[indices].max())
    hid = torch.zeros(n, T, DIM, dtype=torch.float32, device=device)
    mask = torch.zeros(n, T, dtype=torch.float32, device=device)
    for j, i in enumerate(indices):
        t = int(lens[i])
        hid[j, :t] = torch.from_numpy(np.ascontiguousarray(arr[i, :t])).float().to(device)
        mask[j, :t] = 1.0
    return hid, mask


def build_guidance(rows, y, tags_mat, ids_arr, lens, img_mask):
    out = os.path.join(CACHE, "guidance.npy")
    if os.path.exists(out):
        return np.load(out)
    print("building attention-guidance targets (1x) ...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Trendyol-Vision-Flash", trust_remote_code=True, use_fast=False)
    max_len = ids_arr.shape[1]
    per_row = [ids_arr[i, : lens[i]] for i in range(len(rows))]
    t = build_guidance_targets(rows, y, tags_mat, tokenizer, per_row, img_mask, max_len)
    np.save(out, t)
    print("guidance targets saved", flush=True)
    return t


def train_model(tr_idx, va_idx, df, y, y_cat, attr_y, attr_mask, y_tag, arr, lens,
                w5=0.0, guidance=None, epochs=30, tag=""):
    torch.manual_seed(SEED)
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
        loss = W["mod"] * mod_fn(out["mod"], to_tensor(y[b], device).float())
        loss += W["cat"] * cat_fn(out["cat"], to_tensor(y_cat[b], device))
        for k, lg in enumerate(out["attrs"]):
            m = attr_mask[b, k] == 1
            if m.sum() == 0:
                continue
            loss += W["attr"] * cat_fn(lg[m], to_tensor(attr_y[b, k, : attr_vocab[k]].argmax(-1)[m], device))
        tm = y[b] == 1
        if tm.sum():
            loss += W["tags"] * tag_fn(out["tags"][tm], to_tensor(y_tag[b][tm], device))
        if w5 > 0:
            attn = out["attn"][:, 0]  # q_mod (B, T)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(attn, dim=-1)
            tg = to_tensor(np.ascontiguousarray(guidance[b])[:, : mask.shape[1]], device)
            loss += w5 * torch.sum(tg * (torch.log(tg + 1e-9) - torch.log(attn + 1e-9)), dim=-1).mean()
        return loss

    def predict_proba(idx, th=None):
        model.eval()
        probas, cat_preds = [], []
        with torch.no_grad():
            for i in range(0, len(idx), BATCH * 4):
                b = idx[i : i + BATCH * 4].tolist()
                hid, mask = collate(b, arr, lens, device)
                out = model(hid, mask)
                probas.append(torch.sigmoid(out["mod"]).cpu().numpy())
                cat_preds.append(out["cat"].argmax(-1).cpu().numpy())
        return np.concatenate(probas), np.concatenate(cat_preds)

    best = {"auc": -1.0, "state": None, "th": 0.5, "acc": 0.0, "f1": 0.0}
    patience = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        tot_loss = 0.0
        for i in range(0, len(perm), BATCH):
            b = tr_idx[perm[i : i + BATCH]].tolist()
            opt.zero_grad()
            try:
                loss = one_step(b)
                if not torch.isfinite(loss):
                    print("  !! NaN loss, skip", flush=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot_loss += loss.item()
            except Exception as e:
                print(f"  !! step error {type(e).__name__}: {e}", flush=True)
                opt.zero_grad()
                continue
        pva, _ = predict_proba(va_idx)
        va_auc = roc_auc_score(y[va_idx], pva)
        prec, rec, ths = precision_recall_curve(y[va_idx], pva)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_th = float(ths[f1s.argmax()])
        va_acc = accuracy_score(y[va_idx], (pva > best_th).astype(int))
        va_f1 = f1_score(y[va_idx], (pva > best_th).astype(int))
        print(f"  ep{ep + 1} loss={tot_loss / max(len(perm) // BATCH, 1):.4f} "
              f"val auc={va_auc:.4f} acc@{best_th:.2f}={va_acc:.4f} f1={va_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if va_auc > best["auc"]:
            best = {"auc": va_auc, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "th": best_th, "acc": va_acc, "f1": va_f1}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early stop", flush=True)
                break
    model.load_state_dict(best["state"])
    print(f"{tag} best val: auc={best['auc']:.4f} acc={best['acc']:.4f} f1={best['f1']:.4f} th={best['th']:.2f}")
    return model, best


def per_tag_recall(df, te_idx, y, proba, th):
    tags_mat = tag_matrix(df)
    pred = (proba > th).astype(int)
    te = list(te_idx)
    print("  per-tag recall (test rejected rows):")
    for j, t in enumerate(["Marka Uyumsuzluğu", "Başlık/Resim/Açıklama Arasında Büyük Bir Uyuşmazlık",
                           "Cinsellik", "Sağlık Beyanı", "İletişim ve Yönlendirme"]):
        hit = correct = 0
        for i in te:
            if tags_mat[i, j] == 1:
                correct += 1
                if pred[te.index(i)] == 1:
                    hit += 1
        print(f"    {t}: {hit}/{correct} = {hit / max(correct, 1):.3f}")


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--w5", type=float, default=0.0, help="attention-guidance weight")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--cache", default=CACHE, help="hidden-state cache dir")
    ap.add_argument("--save", default=None, help="save best state dict to this path")
    args = ap.parse_args()

    CACHE = args.cache

    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    y_cat, _ = category_labels(df)
    attr_y, attr_mask, attr_keys = attr_labels(df, attr_schema(df))
    y_tag = tag_matrix(df)

    arr, lens, img_mask, ids_arr, meta = load_cache()
    print(f"cache: {meta['Tmax']}x{meta['D']} = {arr.shape}")

    tr, va, te = stratified_split(df)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    guidance = build_guidance(rows, y, y_tag, ids_arr, lens, img_mask) if args.w5 > 0 else None

    model, best = train_model(tr, va, df, y, y_cat, attr_y, attr_mask, y_tag, arr, lens,
                              w5=args.w5, guidance=guidance, epochs=args.epochs, tag="hybrid")

    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"saved -> {args.save}")

    if args.skip_test:
        return
    device = next(model.parameters()).device
    model.eval()
    pte, cte = [], []
    with torch.no_grad():
        for i in range(0, len(te), BATCH * 4):
            b = te[i : i + BATCH * 4].tolist()
            hid, mask = collate(b, arr, lens, device)
            out = model(hid, mask)
            pte.append(torch.sigmoid(out["mod"]).cpu().numpy())
            cte.append(out["cat"].argmax(-1).cpu().numpy())
    pte = np.concatenate(pte)
    cte = np.concatenate(cte)
    pred = (pte > best["th"]).astype(int)
    acc = accuracy_score(y[te], pred)
    f1 = f1_score(y[te], pred)
    auc = roc_auc_score(y[te], pte)
    cat_acc = accuracy_score(y_cat[te], cte)
    print(f"TEST w5={args.w5}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} cat_acc={cat_acc:.4f} (th={best['th']:.2f})")
    per_tag_recall(df, te, y, pte, best["th"])

    from common import record
    record(f"hybrid_frozen_w5{args.w5}", acc=acc, f1=f1, auc=auc, cat_acc=cat_acc, th=best["th"])


if __name__ == "__main__":
    main()
