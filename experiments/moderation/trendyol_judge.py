"""Trendyol-Vision-Flash zero-shot moderation judge (ready for when images exist).

Native InternVL model.chat() API + torchvision preprocessing (from the
Trendyol README). Evaluates on the moderation test split.

Run: uv run --with "transformers==4.56.2" python -u experiments/moderation/trendyol_judge.py
Requires data/qwen_images/{ProductId}/*.jpg to be present.
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

from data import image_paths, load, stratified_split

MODEL_ID = "models/Trendyol-Vision-Flash"
OUT = ".cache/moderation/trendyol_judge.jsonl"
MAX_IMGS = 2
INPUT_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Trendyol's own content-safety prompt pattern
PROMPT = (
    "<image>\nÜrün Başlığı: {title}\nMarka: {brand}\nKategori: {category}\n"
    "Açıklama: {desc}\n\n"
    "Bu ürün görselini ve başlığını inceleyerek moderasyon kararı ver. "
    "Ürün kurallara uygunsa ve bilgiler tutarlıysa \"Onaylandı\", aksi halde "
    "\"Reddedildi\" yaz. Sadece kararı yaz."
)


def build_transform(input_size=INPUT_SIZE):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def generate_decision(model, tokenizer, pixel_values, question, max_new=16):
    """Manual greedy decode: the model's custom chat()/generate() is broken
    on transformers 4.56 (mixed-version custom code). InternVL's forward
    splices vision features at <IMG_CONTEXT> positions."""
    img_tokens = "<img>" + "<IMG_CONTEXT>" * (model.num_image_token * pixel_values.shape[0]) + "</img>"
    q = question.replace("<image>", img_tokens)
    ids = tokenizer(q, return_tensors="pt").input_ids.to(pixel_values.device)
    flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
    n_in = ids.shape[1]
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=ids, pixel_values=pixel_values, image_flags=flags)
            nxt = out.logits[:, -1].argmax(-1).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0][n_in:], skip_special_tokens=True)


def main(limit=None, start=0):
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tr, va, te = stratified_split(df)
    te = sorted(te.tolist())[start : (start + limit) if limit else None]

    missing = sum(1 for i in te if not image_paths(rows[i]["ProductId"], 1))
    print(f"test rows: {len(te)}, missing images: {missing}", flush=True)
    if missing:
        print("IMAGES MISSING — download them first (scripts/download_images.py on a network without the CDN block)", flush=True)

    print("loading model...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval().to("mps")
    model.img_context_token_id = 151671
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    print("loaded", flush=True)
    tf = build_transform()

    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                done[d["idx"]] = d
            except Exception:
                pass

    out_f = open(OUT, "a")
    t0 = time.time()
    n_correct = n_total = 0
    for idx in te:
        if idx in done:
            continue
        from PIL import Image

        r = rows[idx]
        paths = image_paths(r["ProductId"], MAX_IMGS)
        if not paths:
            continue
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
        pixel_values = torch.stack(imgs).to(dtype=torch.float16, device="mps")
        q = PROMPT.format(
            title=r["DisplayName"], brand=r["BrandName"], category=r["CategoryHierarchy"],
            desc=(r["Description"] or "")[:250],
        )
        ans = generate_decision(model, tokenizer, pixel_values, q)
        gt = "Reddedildi" if r["eval_decision"] == 1 else "Onaylandı"
        pred = "Reddedildi" if "Reddedildi" in ans else "Onaylandı"
        n_total += 1
        n_correct += pred == gt
        out_f.write(json.dumps({"idx": idx, "gt": gt, "pred": pred, "raw": ans}, ensure_ascii=False) + "\n")
        out_f.flush()
        el = time.time() - t0
        print(f"[{idx}] {n_correct}/{n_total} acc={n_correct/n_total:.3f} GT={gt} pred={pred} ({el:.0f}s, {n_total/el:.2f}/s)", flush=True)
    print(f"\njudge done: acc={n_correct/n_total:.4f} on {n_total} rows")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    main(limit=args.limit, start=args.start)
