"""Download product images via headless Chrome (passes Cloudflare bot checks
that block curl/requests). Parallel pages, downscale to 512px, resumable.

Run: uv run --no-sync python scripts/download_images_browser.py
"""

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from PIL import Image
from playwright.sync_api import sync_playwright

SRC = "data/qwen_dataset/sample_4k.parquet"
OUT = "data/qwen_images"
MAX_SIZE = 512
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
WORKERS = 5


def main():
    df = pl.read_parquet(SRC).select(["ProductId", "ImageUrlsPipeSeparated"])
    tasks = []
    for row in df.to_dicts():
        urls = [u for u in str(row["ImageUrlsPipeSeparated"]).split("|") if u]
        for i, u in enumerate(urls):
            tasks.append((str(row["ProductId"]), u.strip(), i))
    print(f"total images: {len(tasks)}", flush=True)

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    stats = {"ok": 0, "skip": 0, "fail": 0}

    def worker(urls):
        from playwright.sync_api import sync_playwright as _sp

        with _sp() as p:
            b = p.chromium.launch(channel="chrome", headless=True)
            ctx = b.new_context(user_agent=UA)
            page = ctx.new_page()
            for product_id, url, idx in urls:
                dest = os.path.join(OUT, product_id, f"{idx}.jpg")
                if os.path.exists(dest):
                    stats["skip"] += 1
                    continue
                try:
                    r = page.goto(url, timeout=45000)
                    if r is None or r.status != 200:
                        stats["fail"] += 1
                        continue
                    img = Image.open(io.BytesIO(r.body()))
                    img.thumbnail((MAX_SIZE, MAX_SIZE))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    img.save(dest, "JPEG", quality=85)
                    stats["ok"] += 1
                except Exception:
                    stats["fail"] += 1
            b.close()

    chunks = [tasks[i :: WORKERS] for i in range(WORKERS)]
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(worker, c) for c in chunks]
        for f in as_completed(futs):
            f.result()
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
