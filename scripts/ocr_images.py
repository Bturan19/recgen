"""Apple Vision OCR over product images (macOS native, fast, Turkish-capable).

Extracts all recognized text per image into a cache; the text is added to
the product verbalization so the LLM encoder can see brand names / spec
labels printed on boxes and product photos.

Run: uv run python scripts/ocr_images.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from experiments.moderation.data import DATA, IMG_DIR

OUT = ".cache/moderation/ocr_text.npy"

os.environ.setdefault("PYOBJC_DISABLE_BYPASS", "0")


def ocr_image(path: str) -> list[str]:
    import Vision
    from Foundation import NSURL
    from Quartz import CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex

    url = NSURL.fileURLWithPath_(path)
    src = CGImageSourceCreateWithURL(url, None)
    if src is None:
        return []
    img = CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return []
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(["tr-TR", "en-US"])
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    texts = []
    for obs in req.results() or []:
        c = obs.topCandidates_(1)
        if c and c.count() > 0:
            texts.append(str(c[0].string()))
    return texts


def main():
    df = pl.read_parquet(DATA)
    rows = df["ProductId"].to_list()
    results = []
    done = 0
    for pid in rows:
        d = os.path.join(IMG_DIR, str(pid))
        texts = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".jpg"):
                    texts.extend(ocr_image(os.path.join(d, f)))
        results.append(" | ".join(texts)[:600])
        done += 1
        if done % 200 == 0:
            print(f"  ocr {done}/{len(rows)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, np.array(results, dtype=object), allow_pickle=True)
    n_any = sum(1 for t in results if t)
    print(f"done: {n_any}/{len(rows)} products with OCR text")


if __name__ == "__main__":
    main()
