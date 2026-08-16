"""Hybrid heads: learned query tokens + cross-attention + 4 specialized heads.

BLIP-2/Q-Former pattern: 4 learnable query vectors (moderation, category,
attribute, tag) cross-attend to the VLM's cached full token sequence
(visual + text interleaved). Each head then consumes its own query vector —
no mean-pooling, no 7500-token prompt.

  q_mod  → sigmoid BCE      (P(Reddedildi))
  q_cat  → linear → 645 CE  (CategoryId leaf)
  q_attr → grouped softmax  (Renk / Beden-Yaş / Materyal / ... )
  q_tag  → linear → 22 BCE  (rejection tags, masked to rejected rows)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnQueryBlock(nn.Module):
    """4 learned queries cross-attending to the full token sequence.

    Pre-norm residual blocks: cross-attn (queries<-sequence) + self-attn +
    FFN. Returns (B, nq, D) query vectors and the last cross-attn weights
    (B, nq, T) for attention guidance.
    """

    def __init__(self, dim: int, n_queries: int = 4, n_heads: int = 8,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.seq_norm = nn.LayerNorm(dim)
        blocks = []
        for _ in range(layers):
            blocks.append(_QFormerLayer(dim, n_heads, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, hidden, mask):
        """hidden: (B, T, D); mask: (B, T) 1=valid (float)."""
        B, T, D = hidden.shape
        seq = self.seq_norm(hidden)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, nq, D)
        attn = None
        for blk in self.blocks:
            q, attn = blk(q, seq, mask)
        return self.out_norm(q), attn


class _QFormerLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, seq, mask):
        h = self.n1(q)
        a, w = self.cross_attn(h, seq, seq, key_padding_mask=(mask == 0).bool(), need_weights=True,
                               average_attn_weights=True)
        q = q + self.drop(a)
        h = self.n2(q)
        a, _ = self.self_attn(h, h, h)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.n3(q)))
        return q, w


class ModerationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, q):
        return self.net(q).squeeze(-1)


class CategoryHead(nn.Module):
    def __init__(self, dim, n_cats):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Linear(256, n_cats)

    def forward(self, q):
        return self.out(self.net(q))


class AttributeHeads(nn.Module):
    """Grouped softmax: one linear head per attribute key (Renk, Beden, ...)."""

    def __init__(self, dim, groups):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList([nn.Linear(256, v) for v in groups])

    def forward(self, q):
        h = self.shared(q)
        return [hd(h) for hd in self.heads]


class TagHead(nn.Module):
    def __init__(self, dim, n_tags):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
        )
        self.out = nn.Linear(128, n_tags)

    def forward(self, q):
        return self.out(self.net(q))


class HybridModel(nn.Module):
    """Query block + 4 heads. dim must match the VLM hidden size (1024)."""

    def __init__(self, dim, n_cats, attr_groups, n_tags=22, n_heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.queries = CrossAttnQueryBlock(dim, n_queries=4, n_heads=n_heads, layers=layers, dropout=dropout)
        self.mod_head = ModerationHead(dim)
        self.cat_head = CategoryHead(dim, n_cats)
        self.attr_heads = AttributeHeads(dim, attr_groups)
        self.tag_head = TagHead(dim, n_tags)

    def forward(self, hidden, mask, return_attn=False):
        q, attn = self.queries(hidden, mask)
        out = dict(
            mod=self.mod_head(q[:, 0]),
            cat=self.cat_head(q[:, 1]),
            attrs=self.attr_heads(q[:, 2]),
            tags=self.tag_head(q[:, 3]),
        )
        if return_attn:
            out["attn"] = attn
        return out
