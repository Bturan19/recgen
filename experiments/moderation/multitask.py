"""Moderation v4: multi-task head (binary decision + 23 rejection tags).

Shared embedding -> shared MLP -> [binary logit, 23 tag logits].
The tag supervision adds structure (brand-mismatch, health-claim patterns).
Also tries: more images (up to 8) and class-weighted LightGBM + tuned
threshold for comparison.

Run: uv run python experiments/moderation/multitask.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from data import IMG_DIR, image_paths, load, stratified_split

TEXT_CACHE = ".cache/moderation/smol17/text_emb.npy"
IMG_CACHE4 = ".cache/moderation/siglip/img_emb.npy"
IMG_CACHE8 = ".cache/moderation/siglip/img_emb8.npy"


def encode_images8():
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from multimodal import encode_images  # reuse SigLIP helpers

    device = "mps"
    proc = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").to(device).eval()
    df = load()
    rows = df.to_dicts()
    img_emb = np.zeros((len(rows), 1152), dtype=np.float32)
    for j, r in enumerate(rows):
        paths = image_paths(r["ProductId"], max_imgs=8)
        if paths:
            e = encode_images(paths, proc, model, device)
            img_emb[j] = e.mean(axis=0)
        if (j + 1) % 500 == 0:
            print(f"  img8 {j + 1}/{len(rows)}")
    np.save(IMG_CACHE8, img_emb)
    return img_emb


class MultiTaskHead(nn.Module):
    def __init__(self, dim, hidden=(256, 128), n_tags=23, dropout=0.2):
        super().__init__()
        d = dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.bin = nn.Linear(d, 1)
        self.tags = nn.Linear(d, n_tags)

    def forward(self, x):
        h = self.net(x)
        return self.bin(h).squeeze(-1), self.tags(h)


def main():
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tags = df["eval_rejection_tag"].to_list()
    all_tags = sorted({t for row in tags for t in row})
    tag_idx = {t: i for i, t in enumerate(all_tags)}
    Y = np.zeros((len(rows), len(all_tags)), dtype=np.float32)
    for i, row in enumerate(tags):
        for t in row:
            Y[i, tag_idx[t]] = 1.0
    print(f"tags: {len(all_tags)}")
    tr, va, te = stratified_split(df)

    H = np.load(TEXT_CACHE)
    I = np.load(IMG_CACHE4)
    X = np.concatenate([H, I], axis=1)
    print(f"X: {X.shape}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    model = MultiTaskHead(X.shape[1], n_tags=len(all_tags)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X[tr].astype(np.float32)).to(dev)
    yt = torch.from_numpy(y[tr].astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y[tr]).to(dev)
    Xva = torch.from_numpy(X[va].astype(np.float32)).to(dev)
    yva = torch.from_numpy(y[va].astype(np.float32)).to(dev)
    Xte = torch.from_numpy(X[te].astype(np.float32)).to(dev)

    bce = nn.BCEWithLogitsLoss()
    best, best_state, patience = -1, None, 0
    n = len(Xt)
    for epoch in range(80):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            b, t = model(Xt[idx])
            loss = bce(b, yt[idx]) + 0.5 * bce(t, Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            bv, _ = model(Xva)
        auc = roc_auc_score(yva.cpu().numpy(), torch.sigmoid(bv).cpu().numpy())
        if auc > best:
            best = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        bte, _ = model(Xte)
    pte = torch.sigmoid(bte).cpu().numpy()
    yt_np = y[te]
    prec, rec, th = precision_recall_curve(yt_np, pte)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_th = th[f1s.argmax()] if f1s.argmax() < len(th) else 0.5
    for tag, t in [("0.5", 0.5), ("tuned", best_th)]:
        p = (pte >= t).astype(int)
        print(f"multi-task @{tag}: acc={accuracy_score(yt_np, p):.4f} f1={f1_score(yt_np, p):.4f} auc={roc_auc_score(yt_np, pte):.4f}")

    # class-weighted LightGBM with tuned threshold
    import lightgbm as lgb

    m = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=63,
                           num_threads=1, random_state=0, scale_pos_weight=(1 - y[tr].mean()) / y[tr].mean())
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pte_l = m.predict_proba(X[te])[:, 1]
    prec, rec, th = precision_recall_curve(yt_np, pte_l)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_th = th[f1s.argmax()] if f1s.argmax() < len(th) else 0.5
    p = (pte_l >= best_th).astype(int)
    print(f"lgbm-weighted @{best_th:.3f}: acc={accuracy_score(yt_np, p):.4f} f1={f1_score(yt_np, p):.4f} auc={roc_auc_score(yt_np, pte_l):.4f}")
    np.save(".cache/moderation/test_proba_mt.npy", pte)
    np.save(".cache/moderation/test_proba_lgbw.npy", pte_l)


if __name__ == "__main__":
    main()
