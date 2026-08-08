import os

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        self.hash_path = path + ".sha256"

    def load(self, texts: list[str]) -> np.ndarray | None:
        if not os.path.exists(self.path) or not os.path.exists(self.hash_path):
            return None
        with open(self.hash_path) as f:
            stored_hash = f.read().strip()
        from .encoder import FrozenEncoder

        if stored_hash != FrozenEncoder.text_hash(self, texts):
            print("[EmbeddingCache] hash mismatch, ignoring cache")
            return None
        return np.load(self.path)

    def store(self, texts: list[str], H: np.ndarray):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.save(self.path, H)
        from .encoder import FrozenEncoder

        with open(self.hash_path, "w") as f:
            f.write(FrozenEncoder.text_hash(self, texts))
        print(f"[EmbeddingCache] stored {H.shape} -> {self.path}")
