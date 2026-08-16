"""Hyperbolic (Poincaré ball) category head.

Per the spec doc: the taxonomy is a tree, so embed each leaf node in a
Poincaré ball and classify by distance — the hyperbolic geometry preserves
the parent-child hierarchy instead of treating leaves as a flat bag.

distance(u, v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
logits = -distance(q_cat_proj, leaf_embeddings); CE over leaves.
"""

import torch
import torch.nn as nn


class HyperbolicCategoryHead(nn.Module):
    def __init__(self, dim, n_leaves, hyper_dim=128):
        super().__init__()
        self.hyper_dim = hyper_dim
        self.to_hyperbolic = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, hyper_dim)
        )
        self.leaf_embeddings = nn.Parameter(torch.randn(n_leaves, hyper_dim) * 0.05)

    def poincare_dist(self, u, v):
        u_n = torch.clamp((u ** 2).sum(-1, keepdim=True), max=0.99)
        v_n = torch.clamp((v ** 2).sum(-1, keepdim=True), max=0.99)
        d2 = ((u - v) ** 2).sum(-1, keepdim=True)
        num = 2 * d2
        den = (1 - u_n) * (1 - v_n)
        return torch.acosh(1 + num / (den + 1e-8)).squeeze(-1)

    def forward(self, q_cat):
        pt = torch.tanh(self.to_hyperbolic(q_cat))  # inside the ball
        dist = self.poincare_dist(pt.unsqueeze(1), self.leaf_embeddings.unsqueeze(0))  # (B, L)
        return dist, pt


class FlatCategoryHead(nn.Module):
    """Flat CE counterpart (handoff option A)."""

    def __init__(self, dim, n_cats):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Linear(256, n_cats)

    def forward(self, q_cat):
        return self.out(self.net(q_cat)), None
