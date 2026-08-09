import os
import threading
import time

import numpy as np

from recgen import FrozenEncoder, TemplateVerbalizer
from recgen.cache import EmbeddingCache

from . import config


class EncoderService:
    def __init__(self):
        self._lock = threading.Lock()
        self._encoder = None
        self._heads = {}
        self._verbalizers = {}
        self._hits = 0
        self._misses = 0
        self._t0 = time.time()

    def _get_encoder(self) -> FrozenEncoder:
        if self._encoder is None:
            self._encoder = FrozenEncoder(
                config.MODEL_DIR,
                pooling=config.POOLING,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
            )
        return self._encoder

    def encode(self, texts: list[str], cache_key: str | None = None) -> list[list[float]]:
        enc = self._get_encoder()
        if cache_key is not None:
            path = f"{config.CACHE_DIR}/{cache_key}.npy"
            cache = EmbeddingCache(path)
            cached = cache.load(texts)
            if cached is not None:
                self._hits += 1
                return cached.tolist()
        H = enc.encode(texts, progress=False)
        if cache_key is not None:
            cache.store(texts, H)
        self._misses += 1
        return H.tolist()

    def encode_rows(self, rows: list[dict], fields: list[str] | None, instruction: str = "") -> list[list[float]]:
        verb = TemplateVerbalizer(fields=fields, instruction=instruction).fit(rows[0] if rows else {})
        texts = verb.transform_rows(rows)
        return self.encode(texts)

    def rank(self, context: str, items: list[str], head_key: str | None = None, top_k: int = 10):
        enc = self._get_encoder()
        ctx_emb = np.asarray(self.encode([context]))[0]
        item_embs = np.asarray(self.encode(items))
        scores = ctx_emb @ item_embs.T
        if head_key is not None and head_key in self._heads:
            head = self._heads[head_key]
            scores = head.predict_scores(ctx_emb.reshape(1, -1), item_embs)[0]
        else:
            ctx_n = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-9)
            item_n = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-9)
            scores = ctx_n @ item_n.T
        order = np.argsort(-scores)
        return [
            {"item": items[i], "score": float(scores[i])}
            for i in order[:top_k]
        ]

    def rank_catalog(self, context: str, catalog_items: list[str], cached_embs: np.ndarray, top_k: int = 10):
        """GenRec-style multi-output scoring: ONE forward pass on the context,
        then a single matmul emits scores for the entire catalog at once."""
        enc = self._get_encoder()
        ctx_emb = np.asarray(self.encode([context]))[0]
        scores = ctx_emb @ cached_embs.T
        order = np.argsort(-scores)[::-1]
        return [
            {"item": catalog_items[i], "score": float(scores[i])}
            for i in order[:top_k]
        ]

    def status(self) -> dict:
        return {
            "model": config.MODEL_DIR,
            "pooling": config.POOLING,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "uptime_s": round(time.time() - self._t0),
            "dim": self._encoder.dim if self._encoder else None,
        }


service = EncoderService()
