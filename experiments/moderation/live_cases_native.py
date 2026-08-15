"""Live-site cases with Trendyol-Vision-Flash using its NATIVE task formats
(from the model README): content safety 0/1/2, brand detection, product
caption, attribute extraction (JSON), title generation.

Run: uv run --with "transformers==4.56.2" python -u experiments/moderation/live_cases_native.py
"""

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
CASES_DIR = "data/live_cases"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CASES = [
    {
        "id": "nipple-pads",
        "title": "Yüksek Kaliteli Kadın İç Giyim Göğüs Ucu Meme Kapatan Gizleyen %100 Silikon Ten Nipple Ped 1 Çift",
        "cat": "Anne Bebek",
        "brand": "N/A",
    },
    {
        "id": "canvas-toy",
        "title": "Nü Kadın Deri İç Giyim Oyuncak Yatay Tekli Kanvas Duvar Tablosu Salon Yatak Odası Ofis Cafe",
        "cat": "Ev, Yaşam, Yapı Market > Dekorasyon > Tablo > Kanvas Tablo",
        "brand": "N/A",
    },
    {
        "id": "bikini",
        "title": "Bikini Kadın İç Giyim",
        "cat": "Moda",
        "brand": "N/A",
    },
    {
        "id": "pareo",
        "title": "Beru Kadın Kahve El Yapımı Plaj Pareosu–payet Detaylı Bohem Etek,plaj Ve Tatil Kombini,iç Giyim Najmaddin0EF53F8457CFC737ED3708-com",
        "cat": "Moda",
        "brand": "N/A",
    },
]


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

    def ask(pixel_values, question, max_new=48):
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
        return tok.decode(ids[0][n_in:], skip_special_tokens=True).strip()

    for case in CASES:
        from PIL import Image

        print(f"\n{'='*72}\nCASE: {case['id']} | cat: {case['cat']}", flush=True)
        paths = sorted(os.listdir(os.path.join(CASES_DIR, case["id"])))
        pixel_values = torch.stack(
            [tf(Image.open(os.path.join(CASES_DIR, case["id"], f)).convert("RGB")) for f in paths[:3]]
        ).to(dtype=torch.float16, device="mps")

        # P1: content safety (native 0/1/2)
        t0 = time.time()
        q1 = f"<image>\nÜrün Başlığı: {case['title']}\n\nBu ürün görselini ve başlığını inceleyerek moderasyon sınıflandırması yap. Sonucu 0 (Forbidden), 1 (Fantasy) veya 2 (Safe) olarak ver."
        print(f"  [safety 0/1/2] ({time.time()-t0:.0f}s): {ask(pixel_values, q1, 12)}", flush=True)

        # P2: brand detection (native)
        t0 = time.time()
        q2 = f"<image>\nGörsellerden, {case['cat'].split('>')[0].strip()} kategorisinde yer alan ürünün markasını çıkar.\nSadece verilen görsellerde doğrulanabilen bilgilere dayan.\nEmin olmadığında \"Unknown\" şeklinde cevap ver.\nSadece marka adını döndür."
        print(f"  [brand] ({time.time()-t0:.0f}s): {ask(pixel_values, q2, 12)}", flush=True)

        # P3: product caption (native, English) — reveals what the image actually shows
        t0 = time.time()
        q3 = f"<image>\nWithout speculating about details you cannot see, describe this product based on the image and the information provided.\nProduct title: {case['title']}\nBrand: {case['brand']}\nFirst decide which object is the product, review OCR for brand/model/title clues, then analyze colors, shape, material, pattern, and other grounded details."
        print(f"  [caption] ({time.time()-t0:.0f}s): {ask(pixel_values, q3, 60)}", flush=True)

        # P4: attribute extraction (native JSON)
        t0 = time.time()
        q4 = f"<image>\nBu görseldeki, başlığı '{case['title'][:120]}' olan ürünün Renk, Beden/Ölçü, Malzeme, Model, Kategori bilgilerini json formatında çıkarır mısın?"
        print(f"  [attributes] ({time.time()-t0:.0f}s): {ask(pixel_values, q4, 60)}", flush=True)

        # P5: title generation (native) — judges title quality vs image
        t0 = time.time()
        q5 = f"<image>\nÜrün fotoğrafı ile '{case['title']}' bilgisini karşılaştırıp, Trendyol katalog moderasyon kurallarına göre yanıltıcı ifadelerden kaçınarak net bir başlık üret; çıktıyı düz metin ver."
        print(f"  [title-gen] ({time.time()-t0:.0f}s): {ask(pixel_values, q5, 40)}", flush=True)


if __name__ == "__main__":
    main()
