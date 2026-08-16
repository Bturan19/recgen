"""Live prediction on a single product: all 4 heads + correction signals.

Loads the frozen Trendyol-Vision-Flash + the trained query-token heads
(checkpoint from train.py --save), runs ONE prefill, prints every output:

  - moderation: P(reject), decision at the val-tuned threshold, top tags
  - category: top-5 predicted leaves + correction vs seller's declared path
  - attributes: per-key predicted value + confidence, vs seller's listing
  - attention: what the q_mod query actually focused on (image vs text)

Run: OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
     uv run --with "transformers==4.56.2" python -u experiments/hybrid/predict_product.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from hybrid.data import N_TAGS, TAG_ORDER, attr_schema, category_labels
from hybrid.heads import HybridModel

MODEL_ID = "models/Trendyol-Vision-Flash"
CKPT = "checkpoints/hybrid_blind.pt"
INPUT_SIZE = 448
MAX_IMGS = 3
MAX_DESC_CHARS = 350
IMG_CONTEXT_TOKEN_ID = 151671
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLD = 0.62  # val-tuned (blind run)


def build_text(title, brand, desc, blind=True):
    parts = []
    if title:
        parts.append("Ürün Başlığı: " + title)
    if brand:
        parts.append("Marka: " + brand)
    if desc:
        d = desc if len(desc) <= MAX_DESC_CHARS else desc[:MAX_DESC_CHARS] + "..."
        parts.append("Açıklama: " + d)
    return "\n".join(parts)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--brand", default="")
    ap.add_argument("--desc", default="")
    ap.add_argument("--seller-category", default="", help="declared category path (for correction display)")
    ap.add_argument("--seller-attrs", default="{}", help='declared attrs JSON, e.g. {"Renk":"Kahverengi"}')
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    from PIL import Image

    seller_attrs = json.loads(args.seller_attrs)

    # ---------- labels / vocab (must match training) ----------
    from moderation.data import load
    df = load()
    _, cat_vocab = category_labels(df)
    cat_name = dict(zip(df["CategoryId"].to_list(), df["CategoryName"].to_list()))
    schema = attr_schema(df)
    attr_keys = list(schema)

    # ---------- VLM ----------
    print("loading VLM...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, use_flash_attn=False,
    ).eval().to("mps")
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    tf = build_transform()

    imgs = [tf(Image.open(p).convert("RGB")) for p in args.images[:MAX_IMGS]]
    pixel_values = torch.stack(imgs).to(dtype=torch.float16, device="mps")
    img_tokens = "<img>" + "<IMG_CONTEXT>" * (256 * len(imgs)) + "</img>"
    text = img_tokens + "\n" + build_text(args.title, args.brand, args.desc)
    ids = tokenizer(text, return_tensors="pt").input_ids.to("mps")
    flags = torch.ones(len(imgs), dtype=torch.long, device="mps")
    with torch.no_grad():
        out = model(input_ids=ids, pixel_values=pixel_values, image_flags=flags, output_hidden_states=True)
    hid = out.hidden_states[-1].float()  # (1, T, 1024)
    img_pos = set((ids[0] == IMG_CONTEXT_TOKEN_ID).nonzero().squeeze(-1).tolist())
    print(f"hidden: {hid.shape} ({len(img_pos)} image tokens, {ids.shape[1] - len(img_pos)} text tokens)\n")

    # ---------- heads ----------
    heads = HybridModel(1024, n_cats=len(cat_vocab), attr_groups=[len(v) for v in schema.values()], n_tags=N_TAGS)
    heads.load_state_dict(torch.load(args.ckpt, map_location="mps"))
    heads.eval().to("mps")
    mask = torch.ones(1, ids.shape[1], device="mps")
    with torch.no_grad():
        q, attn = heads.queries(hid, mask)
        out_mod = heads.mod_head(q[:, 0])
        out_cat = heads.cat_head(q[:, 1])
        out_attrs = heads.attr_heads(q[:, 2])
        out_tags = heads.tag_head(q[:, 3])

    p_rej = torch.sigmoid(out_mod).item()
    print("=" * 70)
    print("1) MODERATION")
    print(f"   P(Reddedildi) = {p_rej:.3f}  ->  {'REDDEDILDIR' if p_rej > THRESHOLD else 'ONAYLANIR'} (th={THRESHOLD})")
    tag_probs = torch.sigmoid(out_tags).squeeze(0)
    top_tags = tag_probs.topk(4)
    print("   top rejection tags (if rejected):")
    for i, s in zip(top_tags.indices.tolist(), top_tags.values.tolist()):
        print(f"     {TAG_ORDER[i]:45s} p={s:.3f}")

    print("\n2) CATEGORY (predicted from pixels + title/brand/desc)")
    probs = torch.softmax(out_cat.squeeze(0), dim=-1)
    top5 = probs.topk(5)
    for i, s in zip(top5.indices.tolist(), top5.values.tolist()):
        mark = " <-- seller" if args.seller_category and cat_name[cat_vocab[i]] == args.seller_category else ""
        print(f"     {cat_name[cat_vocab[i]]:55s} p={s:.3f}{mark}")
    if args.seller_category:
        pred = cat_name[cat_vocab[top5.indices[0].item()]]
        print(f"   seller declared: {args.seller_category}")
        print(f"   correction: {'SUGGEST DIFFERENT CATEGORY' if pred != args.seller_category else 'agrees with seller'}")

    print("\n3) ATTRIBUTES")
    for k, lg in enumerate(out_attrs):
        vp = torch.softmax(lg.squeeze(0), dim=-1)
        best = vp.argmax().item()
        val = schema[attr_keys[k]][best]
        conf = vp[best].item()
        listed = seller_attrs.get(attr_keys[k])
        mark = ""
        if listed:
            same = listed.lower() == val.lower()
            mark = " (matches listing)" if same else f" vs listed: {listed}  *** CORRECT OR CONFLICT ***"
        elif conf > 0.4:
            mark = " (not in listing -> ENRICHMENT)"
        print(f"     {attr_keys[k]:12s}: {val:30s} conf={conf:.2f}{mark}")

    # ---------- attention ----------
    print("\n4) ATTENTION (what q_mod focuses on)")
    a = attn[0, 0]  # q_mod over tokens
    img_w = a[list(img_pos)].sum().item() if img_pos else 0.0
    print(f"   mass on image tokens: {img_w:.3f}, text: {1 - img_w:.3f}")
    top_idx = a.topk(8).indices.tolist()
    toks = tokenizer.batch_decode(ids[0][[t for t in top_idx]], skip_special_tokens=True)
    for t, ti in zip(toks, top_idx):
        where = "IMAGE" if ti in img_pos else "text"
        print(f"     {a[ti].item():.3f}  [{where:5s}] {t.strip()[:40]}")


if __name__ == "__main__":
    main()
