"""Framework-completeness audit tests (AUDIT.md section 4):
RecgenPipeline must handle polars AND pandas frames and cache embeddings
across fit/transform/predict calls (no re-tokenization per call)."""

import numpy as np
import polars as pl
import pytest

from recgen import RecgenPipeline, TemplateVerbalizer


class StubEncoder:
    pooling = "mean"
    dim = 4

    def __init__(self):
        self.calls = 0
        self._cache = {}

    def encode(self, texts, progress=True):
        self.calls += 1
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)

    def encode_cached(self, texts, cache_path=None, progress=True):
        key = tuple(texts)
        if key in self._cache:
            return self._cache[key]
        H = self.encode(texts, progress=progress)
        self._cache[key] = H
        return H


def _make_pipe(tmp_path, head="classifier"):
    verb = TemplateVerbalizer(fields=["a", "b"], instruction="Task.")
    enc = StubEncoder()
    pipe = RecgenPipeline(enc, verb, head=head, cache_dir=str(tmp_path / "cache"))
    return pipe, enc


def _df_pandas(n=40):
    import pandas as pd

    return pd.DataFrame({"a": np.arange(n), "b": np.arange(n) % 3})


def _df_polars(n=40):
    return pl.DataFrame({"a": np.arange(n), "b": np.arange(n) % 3})


@pytest.mark.parametrize("make_df", [_df_pandas, _df_polars], ids=["pandas", "polars"])
def test_pipeline_polars_and_pandas(tmp_path, make_df):
    pipe, enc = _make_pipe(tmp_path)
    X = make_df()
    y = (X.to_pandas()["a"] > 20).astype(int).to_numpy() if isinstance(X, pl.DataFrame) else (X["a"] > 20).astype(int).to_numpy()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),)
    H1 = pipe.transform(X)
    assert H1.shape == (len(X), 4)


def test_pipeline_encodes_once_when_cached(tmp_path):
    pipe, enc = _make_pipe(tmp_path)
    X = _df_pandas(20)
    y = (X["a"] > 10).astype(int).to_numpy()
    pipe.fit(X, y)
    n_calls_after_fit = enc.calls
    pipe.transform(X)
    pipe.predict(X)
    assert enc.calls == n_calls_after_fit, "transform/predict must hit the cache, not re-encode"


def test_pipeline_text_list_input(tmp_path):
    pipe, enc = _make_pipe(tmp_path)
    X = ["good movie", "bad movie"] * 10
    y = np.array([1, 0] * 10)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (20,)
