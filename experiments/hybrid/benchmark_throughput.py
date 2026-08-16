"""Throughput benchmark: VLM prefill + heads, batch sizes, MPS vs CPU.

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
     uv run --with "transformers==4.56.2" python -u experiments/hybrid/benchmark_throughput.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from moderation.data import load

MODEL_ID = "models/Trendyol-Vision-Flash"
INPUT_SIZE = 448
MAX_IMGS = 3
MAX_DESC_CHARS = 350
IMG_CONTEXT_TOKEN_ID = 151671
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_paths(product_id):
    d = os.path.join("data/qwen_images", str(product_id))
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))[:MAX_IMGS]


def build_text(r):
    from moderation.data import parse_attributes
    parts = []
    if r.get("DisplayName"):
        parts.append("Ürün Başlığı: " + r["DisplayName"])
    if r.get("BrandName"):
        parts.append("Marka: " + r["BrandName"])
    desc = (r.get("Description") or "").strip()
    if desc:
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[:MAX_DESC_CHARS] + "..."
        parts.append("Açıklama: " + desc)
    return "\n".join(parts)


def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main():
    import numpy as np
    from PIL import Image

    df = load()
    rows = df.to_dicts()
    samples = [r for r in rows if len(image_paths(r["ProductId"])) == MAX_IMGS][:8]
    print(f"samples: {len(samples)} (3 images each)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16 if device == "mps" else torch.float32,
        low_cpu_mem_usage=True, use_flash_attn=False,
    ).eval().to(device)
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    tf = build_transform()

    # precompute inputs
    inputs = []
    for r in samples:
        paths = image_paths(r["ProductId"])
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
        pv = torch.stack(imgs).to(dtype=torch.float16 if device == "mps" else torch.float32, device=device)
        img_tokens = "<img>" + "<IMG_CONTEXT>" * (256 * len(imgs)) + "</img>"
        text = img_tokens + "\n" + build_text(r)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        inputs.append((pv, ids))

    def run_batch(m, batch):
        pvs = [x[0] for x in batch]
        idss = [x[1] for x in batch]
        pixel_values = torch.cat(pvs, dim=0)
        T_ = max(x.shape[1] for x in idss)
        input_ids = torch.zeros(len(batch), T_, dtype=idss[0].dtype, device=device)
        for k, x in enumerate(idss):
            input_ids[k, : x.shape[1]] = x[0]
        flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            out = m(input_ids=input_ids, pixel_values=pixel_values, image_flags=flags,
                    output_hidden_states=True)
        return out.hidden_states[-1]

    for bs in (1, 2, 4):
        reps = 3 if bs == 1 else 2
        t0 = time.time()
        for _ in range(reps):
            for i in range(0, len(samples), bs):
                run_batch(model, inputs[i : i + bs])
        dt = (time.time() - t0) / (reps * len(samples))
        print(f"MPS batch={bs}: {dt * 1000:.0f} ms/product -> {1 / dt:.2f} products/s", flush=True)

    # CPU comparison (float32, 2 products)
    if device == "mps":
        print("CPU comparison (float32, 3 products)...", flush=True)
        del model
        torch.mps.empty_cache()
        cpu_model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.float32,
            low_cpu_mem_usage=False, use_flash_attn=False,
        ).eval().float()
        cpu_model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
        cpu_inputs = []
        for r in samples[:3]:
            paths = image_paths(r["ProductId"])
            imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
            pv = torch.stack(imgs).to("cpu")
            img_tokens = "<img>" + "<IMG_CONTEXT>" * (256 * len(imgs)) + "</img>"
            text = img_tokens + "\n" + build_text(r)
            ids = tokenizer(text, return_tensors="pt").input_ids.to("cpu")
            cpu_inputs.append((pv, ids))
        t0 = time.time()
        run_batch(cpu_model, cpu_inputs)
        run_batch(cpu_model, cpu_inputs)
        dt = (time.time() - t0) / (2 * len(cpu_inputs))
        print(f"CPU batch=1: {dt * 1000:.0f} ms/product -> {1 / dt:.2f} products/s", flush=True)


if __name__ == "__main__":
    main()
