"""Download product images from the marketplace CDN (URLs read from the dataset), downscale to 512px, store
locally as JPEG. Parallel workers, retries, missing images recorded."""

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from PIL import Image

SRC = "data/qwen_dataset/sample_4k.parquet"
OUT = "data/qwen_images"
MAX_SIZE = 512
WORKERS = 16


def fetch_one(args):
    product_id, url, idx = args
    dest = os.path.join(OUT, product_id, f"{idx}.jpg")
    if os.path.exists(dest):
        return product_id, idx, "cached", 0
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return product_id, idx, f"http_{r.status_code}", 0
            img = Image.open(io.BytesIO(r.content))
            img.thumbnail((MAX_SIZE, MAX_SIZE))
            if img.mode != "RGB":
                img = img.convert("RGB")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            img.save(dest, "JPEG", quality=85)
            return product_id, idx, "ok", os.path.getsize(dest)
        except Exception as e:
            if attempt == 2:
                return product_id, idx, f"err_{type(e).__name__}", 0
    return product_id, idx, "err", 0


def main():
    df = pl.read_parquet(SRC).select(["ProductId", "ImageUrlsPipeSeparated"])
    tasks = []
    for row in df.to_dicts():
        urls = [u for u in str(row["ImageUrlsPipeSeparated"]).split("|") if u]
        for i, u in enumerate(urls):
            tasks.append((row["ProductId"], u.strip(), i))
    print(f"total images to fetch: {len(tasks)}")

    stats = {}
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_one, t) for t in tasks]
        for f in as_completed(futs):
            pid, idx, status, size = f.result()
            stats[status] = stats.get(status, 0) + 1
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} {stats}")
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
