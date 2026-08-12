"""Encoder/cache audit tests (AUDIT.md section 5): pooling correctness,
non-finite sanitization, cache staleness. CPU-only, no model download."""

import numpy as np
import torch

from recgen.cache import EmbeddingCache
from recgen.encoder import pool_hidden, sanitize


def test_pooling_mean_masked():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  # 3 attended tokens
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 attended tokens
        ]
    )
    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out = pool_hidden(hidden, attn, "mean")
    np.testing.assert_allclose(out[0], [(1 + 3 + 5) / 3, (2 + 4 + 6) / 3])
    np.testing.assert_allclose(out[1], [1.5, 1.5])


def test_pooling_last_nonpad():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out = pool_hidden(hidden, attn, "last")
    np.testing.assert_allclose(out[0], [5.0, 6.0])
    np.testing.assert_allclose(out[1], [2.0, 2.0])


def test_sanitize_zeroes_nonfinite_elements():
    H = np.array([[1.0, 2.0], [np.nan, 5.0], [3.0, np.inf]])
    out = sanitize(H)
    np.testing.assert_allclose(out[0], [1.0, 2.0])
    np.testing.assert_allclose(out[1], [0.0, 5.0])
    np.testing.assert_allclose(out[2], [3.0, 0.0])
    assert out is not H  # sanitize must not mutate the input in place


def test_cache_staleness_same_length_different_texts(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npy"))
    cache.store(["one", "two"], np.zeros((2, 3)))
    assert cache.load(["one", "two"]) is not None
    assert cache.load(["two", "one"]) is None  # same length, different order
    assert cache.load(["one", "three"]) is None  # same length, different text
    assert cache.load(["one", "two", "three"]) is None  # different length


def test_text_hash_order_sensitive():
    from recgen import FrozenEncoder

    enc = FrozenEncoder  # class; text_hash is a static method
    h1 = enc.text_hash(None, ["a", "b"])
    h2 = enc.text_hash(None, ["b", "a"])
    assert h1 != h2
