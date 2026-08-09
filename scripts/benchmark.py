"""Benchmark + serving projection for recgen on this machine.

Measures:
1. encode throughput (prompts/s) vs sequence length
2. ranking head scoring throughput (items scored per second)
3. full-request latency: encode(history) + score(catalog)
4. projected cost model (M1 now, GPU estimate)

Run: uv run python scripts/benchmark.py [--model smol360|smol17]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from recgen import CatalogRankingHead, FrozenEncoder

MODELS = {
    "smol360": "models/SmolLM2-360M",
    "smol17": "models/SmolLM2-1.7B",
}


def synthetic_texts(n, tokens, prefix="music "):
    word = prefix * 8
    texts = []
    for i in range(n):
        j = i % max(tokens // len(word), 1)
        texts.append(word * (j + 1))
    return texts


def bench_encode(encoder, tokens, n=64):
    texts = synthetic_texts(n, tokens)
    t0 = time.time()
    encoder.encode(texts, progress=False)
    dt = time.time() - t0
    return n / dt


def main(model_name="smol360"):
    enc = FrozenEncoder(MODELS[model_name], pooling="mean", batch_size=32, max_length=1024)
    print(f"\n== {model_name} encode throughput (fp16, MPS) ==")
    rows = []
    for tokens in (50, 150, 300, 500):
        rate = bench_encode(enc, tokens)
        rows.append((tokens, rate))
        print(f"  {tokens:>4} tokens: {rate:5.1f} prompts/s")
        time.sleep(2)

    print("\n== ranking head scoring (catalog-aware, one pass over all items) ==")
    dim = enc.dim
    for n_items in (1_000, 10_000, 100_000):
        E = torch.randn(n_items, dim, dtype=torch.float32)
        H = torch.randn(1, dim, dtype=torch.float32)
        t0 = time.time()
        for _ in range(3):
            scores = (H @ E.T)
        dt = (time.time() - t0) / 3
        print(f"  {n_items:>7,} items scored in {dt * 1000:6.1f} ms -> {n_items / dt:10,.0f} items/s")

    print("\n== full request: encode 300-token history + score 100k-item catalog ==")
    h_texts = synthetic_texts(16, 300)
    t0 = time.time()
    H = enc.encode(h_texts, progress=False)
    dt = (time.time() - t0) / 16
    E = torch.randn(100_000, dim, dtype=torch.float32)
    t0 = time.time()
    (torch.from_numpy(H).float() @ E.T)
    dt_score = time.time() - t0
    per = (dt + dt_score) * 1000
    print(f"  encode: {dt * 1000:6.1f} ms/request, score 100k items: {dt_score * 1000:6.1f} ms")
    print(f"  total: {per:6.1f} ms/request -> {1000 / per:5.1f} requests/s on this M1")

    print("\n== cost projection (estimate) ==")
    requests_per_s_m1 = 1000 / per
    daily_m1 = requests_per_s_m1 * 86400
    gpu_factor = 20  # A10 vs M1 fp16, conservative
    print(f"  M1 (this machine): {requests_per_s_m1:.1f} req/s -> {daily_m1:,.0f} req/day")
    print(f"  A10 GPU (est x{gpu_factor}): {requests_per_s_m1 * gpu_factor:.0f} req/s")
    gpu_hr = 1.5  # $/hr A10 spot
    gpu_req = 3600 * requests_per_s_m1 * gpu_factor
    print(f"  GPU cost per 1M requests: ${1e6 / gpu_req * gpu_hr:.2f}")
    gpu_emb = 100_000 * 1000 / (requests_per_s_m1 * gpu_factor * 3600)
    print(f"  catalog embed (100k items, 1x): {100_000 / (requests_per_s_m1 * gpu_factor) / 3600:.2f} GPU-hr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smol360", choices=list(MODELS))
    args = ap.parse_args()
    main(args.model)
