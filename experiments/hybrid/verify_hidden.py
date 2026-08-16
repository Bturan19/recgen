"""One-shot verification: Trendyol-Vision-Flash output_hidden_states=True.

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run --with "transformers==4.56.2" python -u experiments/hybrid/verify_hidden.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from moderation.data import image_paths, load

MODEL_ID = "models/Trendyol-Vision-Flash"
INPUT_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform():
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def main():
    df = load()
    rows = df.to_dicts()
    with_imgs = [r for r in rows if image_paths(r["ProductId"], 3)]
    print(f"products with images: {len(with_imgs)}")
    r = with_imgs[0]
    print(f"product {r['ProductId']} title={r['DisplayName'][:60]}")

    print("loading model...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, use_flash_attn=False,
    ).eval().to("mps")
    model.img_context_token_id = 151671
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    print("loaded", flush=True)

    tf = build_transform()
    paths = image_paths(r["ProductId"], 3)
    from PIL import Image
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    pixel_values = torch.stack(imgs).to(dtype=torch.float16, device="mps")

    n_img_tokens = model.num_image_token
    img_tokens = "<img>" + "<IMG_CONTEXT>" * (n_img_tokens * pixel_values.shape[0]) + "</img>"
    text = (
        img_tokens + "\nÜrün Başlığı: " + (r["DisplayName"] or "")
        + "\nMarka: " + (r["BrandName"] or "")
        + "\nKategori: " + (r["CategoryHierarchy"] or "")
        + "\nAçıklama: " + (r["Description"] or "")[:300]
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to("mps")
    flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device="mps")

    print(f"input_ids: {ids.shape}, n_img_ctx: {n_img_tokens}, images: {pixel_values.shape[0]}")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, pixel_values=pixel_values, image_flags=flags, output_hidden_states=True)
    dt = time.time() - t0

    hs = out.hidden_states[-1]
    img_pos = (ids[0] == 151671).nonzero().squeeze(-1).tolist()
    print(f"hidden_states[-1]: {hs.shape} ({dt:.1f}s)")
    print(f"matches input len: {hs.shape[1] == ids.shape[1]}")
    print(f"num image-token positions: {len(img_pos)}, first: {img_pos[:3]}, last: {img_pos[-3:]}")
    print(f"hidden dtype: {hs.dtype}")

    print("--- double check with eval_reason keyword in text ---")
    print(f"eval_reason: {str(r.get('eval_reason'))[:150]}")


if __name__ == "__main__":
    main()
