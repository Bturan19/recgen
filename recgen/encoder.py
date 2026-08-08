import hashlib
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import EmbeddingCache

POOLINGS = ("last", "mean")


class FrozenEncoder:
    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        pooling: str = "last",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling}")
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[FrozenEncoder] loading {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def encode(self, texts: list[str], progress: bool = True) -> np.ndarray:
        all_h = []
        t0 = time.time()
        n = len(texts)
        for i in range(0, n, self.batch_size):
            batch = texts[i : i + self.batch_size]
            ids = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**ids, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
            attn = ids["attention_mask"]
            if self.pooling == "last":
                seq_lens = attn.sum(dim=1).long()
                idx = torch.arange(hidden.shape[0], device=hidden.device)
                h = hidden[idx, seq_lens - 1]
            else:
                h = (hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(
                    dim=1, keepdim=True
                )
            all_h.append(h.cpu())
            if progress and (i // self.batch_size) % 10 == 0:
                print(
                    f"  encoded {min(i + self.batch_size, n)}/{n} "
                    f"({time.time() - t0:.0f}s elapsed)"
                )
        H = torch.cat(all_h).numpy().astype(np.float32)
        n_bad = int((~np.isfinite(H)).any(axis=1).sum())
        if n_bad:
            print(f"[FrozenEncoder] WARNING: {n_bad}/{n} rows had non-finite values; zeroed them")
            H[~np.isfinite(H)] = 0.0
        if progress:
            print(
                f"[FrozenEncoder] encoded {n} prompts in {time.time() - t0:.0f}s "
                f"({n / max(time.time() - t0, 1e-9):.1f}/s), shape={H.shape}"
            )
        return H

    def encode_cached(
        self, texts: list[str], cache_path: str | None, progress: bool = True
    ) -> np.ndarray:
        if cache_path is None:
            return self.encode(texts, progress=progress)
        cache = EmbeddingCache(cache_path)
        H = cache.load(texts)
        if H is not None:
            print(f"[FrozenEncoder] cache hit: {cache_path}")
            return H
        H = self.encode(texts, progress=progress)
        cache.store(texts, H)
        return H

    def text_hash(self, texts: list[str]) -> str:
        return hashlib.sha256("\n".join(texts).encode()).hexdigest()
