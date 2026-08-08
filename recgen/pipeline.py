import numpy as np
import polars as pl
import pandas as pd

from .encoder import FrozenEncoder
from .heads import ClassificationHead, RegressionHead
from .verbalizers.base import Verbalizer


class RecgenPipeline:
    def __init__(
        self,
        encoder: FrozenEncoder,
        verbalizer: Verbalizer | None = None,
        head: str = "classifier",
        head_kwargs: dict | None = None,
        cache_dir: str | None = ".cache",
    ):
        self.encoder = encoder
        self.verbalizer = verbalizer
        self.cache_dir = cache_dir
        self.head = (ClassificationHead if head == "classifier" else RegressionHead)(
            **(head_kwargs or {})
        )

    def _to_texts(self, X) -> list[str]:
        if self.verbalizer is None:
            if isinstance(X, (pd.DataFrame, pl.DataFrame)):
                raise ValueError("verbalizer is required for DataFrame input")
            return list(X)
        self.verbalizer.fit(X)
        return self.verbalizer.transform(X)

    def _encode(self, texts: list[str], cache_path: str | None = None) -> np.ndarray:
        return self.encoder.encode_cached(texts, cache_path)

    def fit(self, X, y):
        texts = self._to_texts(X)
        H = self._encode(texts, self._cache_path())
        self.head.fit(H, y)
        return self

    def transform(self, X) -> np.ndarray:
        texts = self._to_texts(X)
        return self._encode(texts, self._cache_path())

    def predict(self, X):
        texts = self._to_texts(X)
        H = self._encode(texts, self._cache_path())
        return self.head.predict(H)

    def predict_proba(self, X):
        texts = self._to_texts(X)
        H = self._encode(texts, self._cache_path())
        return self.head.predict_proba(H)

    def _cache_path(self) -> str | None:
        if self.cache_dir is None:
            return None
        return f"{self.cache_dir}/emb_{self.encoder.pooling}.npy"
