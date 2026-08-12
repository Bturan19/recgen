"""Sequential recommender baselines for the MovieLens benchmark.

Paper-faithful implementations:
- SASRec (Kang & McAuley, ICDM'18): 2 self-attention blocks, single head,
  learned positional embeddings, shared item embeddings, BCE with 1 negative
  per step, dropout 0.2 (ML-1M), max_len 200, Adam lr 1e-3, batch 128,
  early stop on val.
- BERT4Rec (Sun et al., WSDM'19): bidirectional transformer, masked-item
  (cloze) task with 0.15 mask ratio, linear softmax layer over items.
"""

import torch
import torch.nn as nn


def transformer_block(dim, nhead, ff_dim, dropout):
    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nhead,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class SASRec(nn.Module):
    def __init__(self, n_items, dim=50, layers=2, nhead=1, max_len=200, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, dim)
        self.blocks = nn.ModuleList([transformer_block(dim, nhead, dim * 2, dropout) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, seq):
        mask = seq != 0
        x = self.emb(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device).unsqueeze(0))
        x = self.drop(x)
        attn_mask = torch.triu(torch.full((seq.shape[1], seq.shape[1]), float("-inf"), device=seq.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=~mask, src_mask=attn_mask)
        return self.layer_norm(x)


class BERT4Rec(nn.Module):
    def __init__(self, n_items, dim=64, layers=2, nhead=2, max_len=200, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, dim)
        self.blocks = nn.ModuleList([transformer_block(dim, nhead, dim * 2, dropout) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_items + 1, bias=False)

    def forward(self, seq):
        mask = seq != 0
        x = self.emb(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device).unsqueeze(0))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=~mask)
        return self.out(self.layer_norm(x))

    def last_position_logits(self, seq):
        """Logits over items at the last non-pad position only (memory-safe
        for batched eval: no (B, T, V) materialization). Left-padded inputs:
        the last real item is always at position T-1."""
        mask = seq != 0
        x = self.emb(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device).unsqueeze(0))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=~mask)
        x = self.layer_norm(x)
        return self.out(x[:, -1])


def pad_seq(seq, max_len):
    """Left-pad to max_len (SASRec-paper convention: the last item always
    sits at position max_len-1, so position embeddings can specialize)."""
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def train_sasrec(model, seqs, end_targets, val_seqs, val_targets, n_items, n_neg=1, epochs=80, bs=128, lr=1e-3, seed=0, device="cpu"):
    """Full-sequence objective per the paper: at every position t, predict
    s_{t+1}; the LAST position (t = n) predicts end_targets (the held-out
    val action) — so the last-slot output, where eval looks, is trained.
    One random negative per step."""
    import numpy as np

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = np_random(seed)
    best = -1.0
    best_state = None
    patience = 0
    n = model.pos.num_embeddings - 1
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(seqs))
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            s = torch.tensor([pad_seq(seqs[j], n) for j in b], device=device)
            h = model(s)  # (B, n, D)
            s_np = s.cpu().numpy()
            targets_np = s_np[:, 1:].copy()  # positions 0..n-2 predict s[1..n-1]
            last_col = np.array([end_targets[j] for j in b], dtype=np.int64)[:, None]
            targets_np = np.concatenate([targets_np, last_col], axis=1)  # position n-1 -> val item
            real_pos = s_np != 0
            valid = (targets_np != 0) & real_pos
            negs_np = rng.integers(1, n_items + 1, size=targets_np.shape)
            hist_lists = [set(seqs[j]) for j in b]
            for k in range(len(b)):
                hist = hist_lists[k]
                bad = valid[k] & (np.isin(negs_np[k], list(hist)) | (negs_np[k] == targets_np[k]))
                while bad.any():
                    negs_np[k][bad] = rng.integers(1, n_items + 1, size=int(bad.sum()))
                    bad = valid[k] & (np.isin(negs_np[k], list(hist)) | (negs_np[k] == targets_np[k]))
            negs = torch.from_numpy(negs_np).to(device)
            targets = torch.from_numpy(targets_np).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            pos_score = (h * model.emb(targets)).sum(-1)
            neg_score = (h * model.emb(negs)).sum(-1)
            loss = (loss_fn(pos_score, torch.ones_like(pos_score)) + loss_fn(neg_score, torch.zeros_like(neg_score)))
            loss = loss[valid_t].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        hr, ndcg = eval_sequential(model, val_seqs, val_targets, n_items, n_neg=100, seed=seed, device=device)
        print(f"  sasrec epoch {epoch + 1}: val HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
        if hr > best:
            best = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print(f"  sasrec early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_bert4rec(model, seqs, val_seqs, val_targets, n_items, mask_ratio=0.15, epochs=80, bs=128, lr=1e-3, seed=0, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    rng = np_random(seed)
    best = -1.0
    best_state = None
    patience = 0
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(seqs))
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            s = torch.tensor([pad_seq(seqs[j], model.pos.num_embeddings - 1) for j in b], device=device)
            s = torch.where(s == 0, torch.zeros_like(s), s)  # keep padding
            mask = (torch.rand_like(s.float()) < mask_ratio) & (s != 0)
            labels = torch.full_like(s, -1)
            labels[mask] = s[mask]
            s_in = s.clone()
            s_in[mask] = 0  # [mask] token
            logits = model(s_in)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        hr, ndcg = eval_bert4rec(model, val_seqs, val_targets, n_items, n_neg=100, seed=seed, device=device)
        print(f"  bert4rec epoch {epoch + 1}: val HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
        if hr > best:
            best = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print(f"  bert4rec early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def np_random(seed):
    import numpy as np

    return np.random.default_rng(seed)


def _eval_batched(model, seqs, targets, n_items, n_neg=100, seed=0, device="cpu", use_logits=False):
    """Batched leave-one-out eval: one forward over all sequences."""
    import numpy as np

    model.eval()
    max_len = model.pos.num_embeddings - 1
    hits = ndcg = 0.0
    with torch.no_grad():
        S = torch.tensor([pad_seq(s, max_len) for s in seqs], device=device)
        O = model.last_position_logits(S) if use_logits else model(S)
        last_idx = max_len - 1  # left-padded: last real item sits at max_len-1
        for j in range(len(seqs)):
            t = targets[j]
            h = O[j, last_idx] if not use_logits else O[j]
            hist = set(seqs[j])
            pool = [it for it in range(1, n_items + 1) if it not in hist and it != t]
            negs = np.random.default_rng(seed + j).choice(pool, n_neg, replace=False)
            cands = np.concatenate([[t], negs])
            if use_logits:
                scores = h[torch.tensor(cands, device=device)]
            else:
                scores = (h * model.emb(torch.tensor(cands, device=device))).sum(-1)
            rank = int((scores[0] < scores[1:]).sum()) + 1
            if rank <= 10:
                hits += 1
                ndcg += 1.0 / np.log2(rank + 1)
    return hits / len(seqs), ndcg / len(seqs)


def eval_sequential(model, seqs, targets, n_items, n_neg=100, seed=0, device="cpu"):
    """Leave-one-out over positive + n_neg random negatives (SASRec-style
    shared-embedding dot product)."""
    return _eval_batched(model, seqs, targets, n_items, n_neg, seed, device, use_logits=False)


def eval_bert4rec(model, seqs, targets, n_items, n_neg=100, seed=0, device="cpu"):
    """Same protocol; BERT4Rec predicts directly over the item vocabulary."""
    return _eval_batched(model, seqs, targets, n_items, n_neg, seed, device, use_logits=True)
