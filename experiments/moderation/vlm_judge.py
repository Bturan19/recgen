"""Marketplace VLM judge: Qwen2.5-VL-3B moderates products (image + text).

Resumable: answers are saved incrementally to a JSONL cache. Parses the
JSON decision from the model output; falls back to text search.

Usage: uv run python experiments/moderation/vlm_judge.py [--limit N] [--start I]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import image_paths, load, stratified_split

MODEL_DIR = "models/Qwen2.5-VL-3B"
OUT = ".cache/moderation/vlm_judge.jsonl"
MAX_IMGS = 3
MAX_NEW = 100

SYS = (
    "Sen bir e-ticaret platformu moderasyon uzmanısın. Ürün görsellerini ve metin bilgilerini dikkatlice incele. "
    "Reddetme sebepleri şunlar olabilir: (1) Marka uyumsuzluğu: görsellerdeki marka ile ürün bilgisindeki marka farklı. "
    "(2) Başlık/Resim/Açıklama arasında büyük uyuşmazlık: görseldeki ürün, başlık veya açıklamadaki teknik özelliklerle çelişiyor "
    "(ör. başlıkta 35W yazıyor ama ürün üzerinde 20W yazıyor). (3) Cinsellik içerikli görsel veya metin. "
    "(4) Sağlık beyanı: ürünün tedavi edici/iyileştirici iddiaları. (5) Yasa dışı/kontrollü maddeler, silah, kesici alet. "
    "(6) Yanıltıcı/aldatıcı ürün. Eğer ürün temiz ve bilgileri tutarlıysa ONAYLA. "
    "Sadece geçerli JSON döndür, başka hiçbir şey yazma: "
    '{"decision": "Onaylandı" veya "Reddedildi", "reason": "kısa gerekçe"}'
)


def parse_answer(ans: str):
    m = re.search(r"\{.*\}", ans, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return d.get("decision"), d.get("reason")
        except Exception:
            pass
    if "reddedildi" in ans.lower():
        return "Reddedildi", ans[:150]
    if "onaylandı" in ans.lower():
        return "Onaylandı", ans[:150]
    return None, ans[:150]


def main(limit: int = None, start: int = 0):
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tr, va, te = stratified_split(df)
    te = sorted(te.tolist())
    if limit:
        te = te[start : start + limit]
    else:
        te = te[start:]

    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                done[d["idx"]] = d
            except Exception:
                pass

    print(f"loading {MODEL_DIR}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.float16, torch_dtype=torch.float16).to("mps").eval()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    print("loaded", flush=True)

    out_f = open(OUT, "a")
    t0 = time.time()
    n_correct = 0
    n_total = 0
    for idx in te:
        if idx in done:
            continue
        from PIL import Image

        r = rows[idx]
        paths = image_paths(r["ProductId"], MAX_IMGS)
        imgs = [Image.open(p).convert("RGB") for p in paths] or [Image.new("RGB", (16, 16), "white")]
        text = (
            f"Ürün başlığı: {r['DisplayName']}\n"
            f"Marka: {r['BrandName']}\n"
            f"Kategori: {r['CategoryHierarchy']}\n"
            f"Öznitelikler: {r['AttributesJson']}\n"
            f"Açıklama: {(r['Description'] or '')[:500]}"
        )
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "image", "image": i} for i in imgs] + [{"type": "text", "text": text}]},
        ]
        t_prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[t_prompt], images=[imgs], return_tensors="pt").to("mps")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW)
        ans = proc.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        decision, reason = parse_answer(ans)
        gt = "Reddedildi" if r["eval_decision"] == 1 else "Onaylandı"
        ok = decision == gt
        n_total += 1
        n_correct += ok
        rec = {"idx": idx, "gt": gt, "decision": decision, "reason": reason, "raw": ans}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        elapsed = time.time() - t0
        rate = (n_total) / elapsed
        print(f"[{idx}] {n_correct}/{n_total} acc={n_correct/n_total:.3f} | GT={gt} VLM={decision} | {elapsed:.0f}s ({rate:.2f}/s)", flush=True)
    print(f"\njudge done: acc={n_correct/n_total:.4f} on {n_total} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    main(limit=args.limit, start=args.start)
