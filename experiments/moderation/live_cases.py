"""Live-site moderation case study with Trendyol-Vision-Flash.

4 problematic products from the live site (not in the dataset): fetch page
metadata, download images via headless Chrome (Cloudflare), then run the
Trendyol model with 4 prompt types: decision, rejection tag, category
matching, attribute extraction.

Run: uv run --with "transformers==4.56.2" python -u experiments/moderation/live_cases.py
"""

import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "models/Trendyol-Vision-Flash"
OUT_DIR = "data/live_cases"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

CASES = [
    {
        "id": "nipple-pads",
        "url": "https://www.pazarama.com/yuksek-kaliteli-kadin-ic-giyim-gogus-ucu-meme-kapatan-gizleyen-100-silikon-ten-nipple-ped-1-cift-p-89654868560",
        "cat": "Anne Bebek",
    },
    {
        "id": "canvas-toy",
        "url": "https://www.pazarama.com/nu-kadin-deri-ic-giyim-oyuncak-yatay-tekli-kanvas-duvar-tablosu-salon-yatak-odasi-ofis-cafe-p-Art-002196",
        "cat": "Ev, Yaşam, Yapı Market > Dekorasyon",
    },
    {
        "id": "bikini",
        "url": "https://www.pazarama.com/bikini-kadin-ic-giyim-p-TE11203941939",
        "cat": "Moda",
    },
    {
        "id": "pareo",
        "url": "https://www.pazarama.com/najmaddin-kadin-pareo-kahve-payetli-bohem-el-yapimi-l-xl-najmaddin0ef53f8457cfc737ed3708-com-p-Najmaddin0EF53F8457CFC737ED3708-com",
        "cat": "Moda",
    },
]

PROMPTS = {
    "decision": (
        "<image>\nÜrün Başlığı: {title}\nKategori: {cat}\n\n"
        "Bu ürünü marketplace kurallarına göre değerlendir: marka uyumsuzluğu, "
        "başlık/resim/açıklama uyuşmazlığı, cinsellik, sağlık beyanı, yasa dışı madde, "
        "yanıltıcı ürün, iletişim bilgisi. Ürün kurallara uygunsa ve bilgiler tutarlıysa "
        "\"Onaylandı\", aksi halde \"Reddedildi\" yaz. Sadece kararı yaz."
    ),
    "tag": (
        "<image>\nÜrün Başlığı: {title}\nKategori: {cat}\n\n"
        "Bu ürün reddedildiyse hangi kuralı ihlal ediyor? Seçenekler: Marka Uyumsuzluğu, "
        "Başlık/Resim/Açıklama Arasında Büyük Bir Uyuşmazlık, Cinsellik, Sağlık Beyanı, "
        "İletişim ve Yönlendirme, Yasa Dışı/Kontrollü Maddeler, Aldatıcı Ürün, "
        "Kategori Hatası, Diğer. Sadece etiketi yaz."
    ),
    "category": (
        "<image>\nÜrün Başlığı: {title}\nMevcut Kategori: {cat}\n\n"
        "Bu ürünün görseline göre mevcut kategorisi doğru mu? Değilse doğru kategoriyi öner. "
        "Cevap formatı: \"Uygun\" veya \"Önerilen Kategori: <kategori>\"."
    ),
    "attributes": (
        "<image>\nÜrün Başlığı: {title}\n\n"
        "Bu ürünün görselinden temel öznitelikleri çıkar (renk, beden, malzeme, model vb.). "
        "Kısa liste halinde yaz."
    ),
}


def fetch_page(url):
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": UA})
    html = r.text
    title = ""
    m = re.search(r'"name":"([^"]{10,150})","@type":"Product"', html)
    if not m:
        m = re.search(r'<h1[^>]*>([^<]{10,150})</h1>', html)
    if m:
        title = m.group(1)
    imgs = sorted(set(re.findall(r'https://img\.pzrmcdn\.com/asset/[^" ]+\.(?:jpg|jpeg|png|webp)', html)))
    return title, imgs


def download_images(case_id, urls):
    import requests
    from PIL import Image
    from playwright.sync_api import sync_playwright

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, case_id)
    os.makedirs(dest, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context(user_agent=UA)
        page = ctx.new_page()
        for i, url in enumerate(urls[:3]):
            fp = os.path.join(dest, f"{i}.jpg")
            if os.path.exists(fp):
                continue
            try:
                r = page.goto(url, timeout=45000)
                if r is None or r.status != 200:
                    continue
                img = Image.open(io.BytesIO(r.body()))
                img.thumbnail((512, 512))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(fp, "JPEG", quality=85)
            except Exception:
                pass
        b.close()
    return sorted(os.listdir(dest))


def main():
    tf = T.Compose(
        [
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attn=False
    ).eval().to("mps")
    model.img_context_token_id = 151671
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    def ask(pixel_values, question, max_new=40):
        img_tokens = "<img>" + "<IMG_CONTEXT>" * (model.num_image_token * pixel_values.shape[0]) + "</img>"
        q = question.replace("<image>", img_tokens)
        ids = tok(q, return_tensors="pt").input_ids.to("mps")
        flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device="mps")
        n_in = ids.shape[1]
        with torch.no_grad():
            for _ in range(max_new):
                out = model(input_ids=ids, pixel_values=pixel_values, image_flags=flags)
                nxt = out.logits[:, -1].argmax(-1).unsqueeze(0)
                ids = torch.cat([ids, nxt], dim=1)
                if nxt.item() == tok.eos_token_id:
                    break
        return tok.decode(ids[0][n_in:], skip_special_tokens=True)

    for case in CASES:
        print(f"\n{'='*70}\nCASE: {case['id']} ({case['url'][:60]}...)", flush=True)
        title, imgs = fetch_page(case["url"])
        print(f"  title: {title[:100]}")
        print(f"  category (from page): {case['cat']}")
        print(f"  images found: {len(imgs)}")
        files = download_images(case["id"], imgs)
        print(f"  downloaded: {files}")
        paths = [os.path.join(OUT_DIR, case["id"], f) for f in files]
        if not paths:
            print("  !! no images, skipping")
            continue
        from PIL import Image

        pixel_values = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(
            dtype=torch.float16, device="mps"
        )
        for name, prompt in PROMPTS.items():
            t0 = time.time()
            ans = ask(pixel_values, prompt.format(title=title, cat=case["cat"]))
            print(f"  [{name}] ({time.time()-t0:.0f}s): {ans[:200]}", flush=True)


if __name__ == "__main__":
    main()
