"""Shared audit helpers: leakage-free evaluation utilities for e-commerce benchmarks.

Leakage definition: a user whose held-out (last) purchase also appears in
their history (repeat purchase) leaks the answer into the model input. Such
users are excluded from training and evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ecommerce.data import build_splits, prep_data

N_USERS = 25000
SEED = 0


def load_splits(min_hist: int = 5, max_history: int = 10, n_users: int = N_USERS, seed: int = SEED):
    reviews, meta = prep_data()
    splits = build_splits(reviews, max_history=max_history)
    splits = [s for s in splits if len(s[1]) >= min_hist]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(splits))
    splits = [splits[i] for i in perm[:n_users]]
    return splits, meta


def leak_mask(splits):
    """True where the test item appears in the user's history (label leakage)."""
    return np.array([any(it == t for it, _ in hist) for _, hist, t in splits])


def split_sets(splits, n_tr_ratio=0.8, n_va_ratio=0.1):
    n = len(splits)
    n_tr = int(n * n_tr_ratio)
    n_va = int(n * n_va_ratio)
    train = splits[:n_tr]
    val = splits[n_tr : n_tr + n_va]
    test = splits[n_tr + n_va :]
    return train, val, test
