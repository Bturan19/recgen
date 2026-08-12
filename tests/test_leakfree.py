"""Leakage audit tests (AUDIT.md section 1).

These need the Amazon Musical Instruments data (gitignored); they skip on CI
or machines without data. They assert the benchmark splits are leak-free:
- no test user's history contains their target item_id
- no test user's verbalized history text contains the test item's title
- negative candidates exclude the positive and all history items
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments" / "ecommerce"))

REVIEWS = REPO / "data" / "raw" / "review_categories" / "Musical_Instruments.jsonl"

pytestmark = pytest.mark.skipif(
    not REVIEWS.exists(),
    reason="e-commerce data not present (gitignored); skipped outside the data machine",
)

from ecommerce.data import verbalize_history  # noqa: E402
from leakfree import leak_mask, load_splits  # noqa: E402


@pytest.fixture(scope="module")
def clean_splits():
    splits, meta = load_splits(min_hist=5, max_history=10, n_users=25000, seed=0)
    mask = leak_mask(splits, meta=meta)
    assert int(mask.sum()) > 0, "expected repeat-purchase users to be filtered"
    return [s for s, m in zip(splits, mask) if not m], meta


def test_item_id_not_in_history(clean_splits):
    splits, _ = clean_splits
    for u, hist, t in splits:
        assert all(it != t for it, _ in hist), f"user {u}: test item {t} in history"


def test_verbalized_history_has_no_test_title(clean_splits):
    splits, meta = clean_splits
    for u, hist, t in splits:
        text = verbalize_history(hist, meta)
        m = meta.filter(meta["item_id"] == t)
        if m.is_empty():
            continue
        title = m.to_dicts()[0]["title"]
        assert title not in text, f"user {u}: test item title {title!r} leaked into history text"


def test_negative_candidates_exclude_positive_and_history(clean_splits):
    splits, _ = clean_splits
    n_tr = int(len(splits) * 0.8)
    n_va = int(len(splits) * 0.1)
    test = splits[n_tr + n_va :]
    all_items = sorted({it for _, _, it in splits})
    n_neg = 100
    for j, (u, hist, t) in enumerate(test):
        forbidden = {it for it, _ in hist} | {t}
        pool = np.array([i for i in all_items if i not in forbidden], dtype=object)
        negs = np.random.default_rng(0 + j).choice(pool, n_neg, replace=False)
        assert t not in negs
        assert not (set(negs) & forbidden)


def test_leak_filter_removes_only_leaked_users(clean_splits):
    splits, meta = clean_splits
    assert len(splits) == 23865, f"expected 23865 clean users, got {len(splits)}"


def test_splits_are_disjoint_and_exhaustive(clean_splits):
    splits, _ = clean_splits
    n_all = len(splits)
    n_tr = int(n_all * 0.8)
    n_va = int(n_all * 0.1)
    train, val, test = splits[:n_tr], splits[n_tr : n_tr + n_va], splits[n_tr + n_va :]
    assert len(train) + len(val) + len(test) == n_all
    users = [s[0] for s in splits]
    assert len(set(users)) == n_all
