"""Hybrid VLM hidden-state extraction (frozen encoder, cached for Phase 1).

One forward per product (up to 3 images, 448px) through Trendyol-Vision-Flash
(InternVL3.5-1B, transformers 4.56.2 overlay), caching last_hidden_state for
the FULL token sequence plus token masks. This is what the query-token block
cross-attends to — no mean pooling.

Cache (`.cache/hybrid/trendyol/`):
  hidden.npy     (n, Tmax, 1024) fp16  — last_hidden_state, padded
  lens.npy       (n,)             int32 — real sequence length per row
  img_mask.npy   (n, Tmax)        uint8 — 1 = vision token position
  input_ids.npy  (n, Tmax)        int32 — padded with pad id (for guidance)
  meta.json                        Tmax / D / order (aligned to data.py load())

Run (model load ~1min + ~1s/row → ~70min for 4000):
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  uv run --with "transformers==4.56.2" python -u experiments/hybrid/vlm_hidden.py
Resumable: rows already extracted are skipped.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from moderation.data import load

MODEL_ID = "models/Trendyol-Vision-Flash"
CACHE = ".cache/hybrid/trendyol"
INPUT_SIZE = 448
MAX_IMGS = 3
MAX_DESC_CHARS = 350
IMG_CONTEXT_TOKEN_ID = 151671
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_text(r, img_token_str, blind=False):
    parts = [img_token_str]
    if r.get("DisplayName"):
        parts.append("Ürün Başlığı: " + r["DisplayName"])
    if r.get("BrandName"):
        parts.append("Marka: " + r["BrandName"])
    if not blind and r.get("CategoryHierarchy"):
        parts.append("Kategori: " + r["CategoryHierarchy"])
    attrs = r.get("_attrs") or []
    if not blind and attrs:
        parts.append("Özellikler: " + " | ".join(attrs[:10]))
    desc = (r.get("Description") or "").strip()
    if desc:
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[:MAX_DESC_CHARS] + "..."
        parts.append("Açıklama: " + desc)
    return "\n".join(parts)


def image_paths(product_id):
    d = os.path.join("data/qwen_images", str(product_id))
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))[:MAX_IMGS]


def build_transform():
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def main(blind=False):
    df = load()
    rows = df.to_dicts()
    n = len(rows)
    print(f"rows: {n} (blind={blind})", flush=True)

    # attributes per row (cached on the dict so build_text can use them)
    from moderation.data import parse_attributes
    for r in rows:
        r["_attrs"] = parse_attributes(r.get("AttributesJson"))

    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    os.makedirs(CACHE, exist_ok=True)
    meta_path = os.path.join(CACHE, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        tmax = meta["Tmax"]
        d = meta["D"]
        print(f"resuming: meta found, Tmax={tmax} D={d}", flush=True)
    else:
        print(f"pass 1: tokenizing to find Tmax... (blind={blind})", flush=True)
        t0 = time.time()
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "right"
        max_len = 0
        all_ids = []
        for i, r in enumerate(rows):
            img_tokens = "<img>" + "<IMG_CONTEXT>" * (256 * len(image_paths(r["ProductId"]))) + "</img>"
            if not image_paths(r["ProductId"]):
                img_tokens = ""
            text = build_text(r, img_tokens, blind=blind)
            ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=2048)["input_ids"]
            max_len = max(max_len, len(ids))
            all_ids.append(ids)
            if (i + 1) % 500 == 0:
                print(f"  tokenized {i + 1}/{n} (max {max_len})", flush=True)
        tmax = max_len
        d = 1024
        meta = {"Tmax": tmax, "D": d, "model": MODEL_ID, "n": n, "note": "row order == data.load() order"}
        json.dump(meta, open(meta_path, "w"))
        print(f"pass 1 done: Tmax={tmax} ({time.time() - t0:.0f}s)", flush=True)
        np.save(os.path.join(CACHE, "input_ids.npy"), np.zeros((n, tmax), dtype=np.int32))

    hid_path = os.path.join(CACHE, "hidden.npy")
    if not os.path.exists(hid_path):
        arr = np.lib.format.open_memmap(hid_path, mode="w+", dtype=np.float16, shape=(n, tmax, d))
        arr.flush()
    else:
        arr = np.lib.format.open_memmap(hid_path, mode="r+")
    lens = np.load(os.path.join(CACHE, "lens.npy")) if os.path.exists(os.path.join(CACHE, "lens.npy")) else np.zeros(n, dtype=np.int32)
    img_mask = np.load(os.path.join(CACHE, "img_mask.npy")) if os.path.exists(os.path.join(CACHE, "img_mask.npy")) else np.zeros((n, tmax), dtype=np.uint8)
    ids_arr = np.load(os.path.join(CACHE, "input_ids.npy"))

    print("loading model...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, use_flash_attn=False,
    ).eval().to("mps")
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
    print("loaded", flush=True)
    tf = build_transform()
    from PIL import Image

    t0 = time.time()
    done = 0
    for i, r in enumerate(rows):
        if lens[i] > 0:
            done += 1
            continue
        paths = image_paths(r["ProductId"])
        if not paths:
            continue
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
        pixel_values = torch.stack(imgs).to(dtype=torch.float16, device="mps")
        img_tokens = "<img>" + "<IMG_CONTEXT>" * (256 * len(paths)) + "</img>"
        text = build_text(r, img_tokens, blind=blind)
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=tmax).input_ids.to("mps")
        flags = torch.ones(len(paths), dtype=torch.long, device="mps")
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=pixel_values, image_flags=flags, output_hidden_states=True)
        hs = out.hidden_states[-1].float().cpu().numpy()[0]  # (T, D)
        T = hs.shape[0]
        ids_cpu = ids[0].cpu().numpy()
        arr[i, :T] = hs
        lens[i] = T
        img_mask[i, :T] = (ids_cpu == IMG_CONTEXT_TOKEN_ID).astype(np.uint8)
        ids_arr[i, :T] = ids_cpu
        done += 1
        if done % 100 == 0:
            arr.flush()
            np.save(os.path.join(CACHE, "lens.npy"), lens)
            np.save(os.path.join(CACHE, "img_mask.npy"), img_mask)
            np.save(os.path.join(CACHE, "input_ids.npy"), ids_arr)
            el = time.time() - t0
            print(f"  {done}/{n} ({el / done:.1f}s/row, {el:.0f}s elapsed, eta {(n - done) * el / max(done, 1) / 60:.0f}min)", flush=True)

    arr.flush()
    np.save(os.path.join(CACHE, "lens.npy"), lens)
    np.save(os.path.join(CACHE, "img_mask.npy"), img_mask)
    np.save(os.path.join(CACHE, "input_ids.npy"), ids_arr)
    print(f"\nextraction done: {done}/{n} rows cached to {CACHE}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", action="store_true", help="omit category + attributes lines from the VLM text")
    args = ap.parse_args()
    if args.blind:
        CACHE = ".cache/hybrid/trendyol_blind"
    main(blind=args.blind)
