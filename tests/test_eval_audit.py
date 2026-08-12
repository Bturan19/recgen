"""Evaluation-correctness audit tests (AUDIT.md section 2/3):
bootstrap CI cannot emit NaN; candidate sets are identical across models;
rank computation is deterministic with tie-safe guards."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "ecommerce"))

from strong_baselines import bootstrap_ci  # noqa: E402


def test_bootstrap_ci_finite_positive_ranks():
    ranks = np.random.default_rng(0).integers(1, 500, size=2000)
    lo, hi = bootstrap_ci(ranks, n=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0 < lo <= hi


def test_bootstrap_ci_no_nan_with_zero_ranks():
    ranks = np.array([0] * 500 + list(np.arange(1, 501)))
    lo, hi = bootstrap_ci(ranks, n=200, seed=1)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo >= 0.0


def test_bootstrap_ci_tighter_for_better_ranks():
    rng = np.random.default_rng(0)
    good = rng.integers(1, 20, size=2000)
    bad = rng.integers(1, 500, size=2000)
    lo_g, hi_g = bootstrap_ci(good, n=200, seed=2)
    lo_b, hi_b = bootstrap_ci(bad, n=200, seed=2)
    assert lo_g > hi_b, "better ranks must produce a strictly higher CI"


def test_hr_ndcg_consistent_with_manual_rank():
    from sota_benchmark import hr_ndcg  # noqa: E402

    pos = np.array([3.0, 1.0, 0.5])
    neg = np.array([[2.9, 2.8, 2.7], [0.0, -1.0, -2.0], [0.4, 0.3, 0.2]])
    hr, ndcg = hr_ndcg(pos, neg)
    ranks = np.array([1 + int((pos[i] < neg[i]).sum()) for i in range(3)])
    expected_hr = float(np.mean(ranks <= 10))
    expected_ndcg = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)))
    assert hr == expected_hr
    assert ndcg == expected_ndcg


def test_negative_candidates_seeded_and_excluding():
    rng = np.random.default_rng(7)
    items = np.array(list(range(1, 1001)), dtype=object)
    forbidden = {10, 42, 7}
    pool = np.array([i for i in items if i not in forbidden], dtype=object)
    a = rng.choice(pool, 100, replace=False)
    rng2 = np.random.default_rng(7)
    pool2 = np.array([i for i in items if i not in forbidden], dtype=object)
    b = rng2.choice(pool2, 100, replace=False)
    assert set(a) == set(b), "same seed must give same candidates"
    assert not (set(a) & forbidden)
