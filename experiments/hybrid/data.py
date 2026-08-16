"""Hybrid VLM + query-token pipeline: multi-task label plumbing.

Reuses experiments/moderation/data.py (load / verbalize / stratified_split)
and adds the label builders for the 4 specialized heads:
  - moderation: eval_decision (binary)
  - category:   CategoryId leaf (645 classes)
  - attribute:  grouped softmax over top attribute keys (Renk, Beden, ...)
  - tag:        multi-label BCE over rejection tags (masked to rejected rows)
plus the weak attention-guidance targets (from eval_reason keyword matching).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from moderation.data import DATA, IMG_DIR, SEED, image_paths, load, parse_attributes, stratified_split  # noqa: F401

N_TAGS = 23
TAG_ORDER = [
    "Marka Uyumsuzluğu",
    "Başlık/Resim/Açıklama Arasında Büyük Bir Uyuşmazlık",
    "Cinsellik",
    "Sağlık Beyanı",
    "İletişim ve Yönlendirme",
    "Yetersiz Bilgi",
    "Yasa Dışı/Kontrollü Maddeler",
    "Kesici Aletler",
    "Aldatıcı Ürün/Hizmet",
    "Diğer Platform İsmi",
    "Organik Ürün",
    "Beşeri Tıbbi Ürün",
    "Tıbbi Cihaz",
    "Geçici Görsel",
    "Yazar Bilgisi Eksikliği veya Hatası",
    "Alakasız Resim",
    "Siyasi İçerik",
    "Şiddet ve Nefret Söylemi",
    "Uygunsuz Dil",
    "Patlayıcı ve Yanıcı Maddeler",
    "Çocuk Manken",
    "Tester/Numune Ürünler",
    "Diğer",
]

# Weak attention-guidance keywords, per rejection tag (Turkish, lowercase).
# Matched against the VLM's *text* tokens (title/brand/category/desc).
GUIDANCE_KEYWORDS = {
    "Marka Uyumsuzluğu": None,  # dynamic: quoted brand strings from eval_reason
    "Sağlık Beyanı": ["sağlık", "şifa", "tedavi", "hastalık", "ilaç", "fayda", "iyileştir", "vitamin", "doktor"],
    "Cinsellik": ["seks", "cinsel", "erotik", "porno", "vibratör", "dildo"],
    "Yasa Dışı/Kontrollü Maddeler": ["yasa dışı", "kontrollü", "uyuşturucu", "kenevir", "esrar", "bonzai"],
    "Tıbbi Cihaz": ["tıbbi", "cihaz", "steril"],
    "Beşeri Tıbbi Ürün": ["ilaç", "tıbbi ürün", "reçete"],
    "İletişim ve Yönlendirme": ["whatsapp", "telefon", "iletişim", "gsm", "tel:"],
    "Diğer Platform İsmi": ["amazon", "hepsiburada", "n11", "ebay", "aliexpress", "sahibinden"],
    "Aldatıcı Ürün/Hizmet": ["sahte", "orijinal", "kopya", "aldatıcı"],
    "Organik Ürün": ["organik", "eko sertifikalı"],
    "Yetersiz Bilgi": ["bilgi eksik", "açıklama eksik", "yetersiz"],
    "Şiddet ve Nefret Söylemi": ["şiddet", "nefret", "ırkçı"],
    "Siyasi İçerik": ["siyasi", "parti", "seçim"],
    "Uygunsuz Dil": ["küfür", "argo", "hakaret"],
    "Kesici Aletler": ["bıçak", "kılıç", "jilet", "balta"],
    "Patlayıcı ve Yanıcı Maddeler": ["patlayıcı", "yanıcı", "barut", "fişek"],
    "Çocuk Manken": ["çocuk manken"],
}

# Attribute schema: top keys by frequency, top values per key (grouped softmax).
ATTR_KEYS = ["Renk", "Beden/Yaş", "Materyal", "Cinsiyet", "Ürün Tipi", "Uyumlu Model", "Uyumlu Marka"]
ATTR_VOCAB_SIZE = 48


def tags_list(val):
    if not val:
        return []
    try:
        return json.loads(val) if isinstance(val, str) else list(val)
    except Exception:
        return []


def tag_matrix(df) -> np.ndarray:
    """(n, 23) binary tag labels, aligned with TAG_ORDER."""
    mat = np.zeros((len(df), N_TAGS), dtype=np.float32)
    for i, tags in enumerate(df["eval_rejection_tag"].to_list()):
        for t in tags_list(tags):
            if t in TAG_ORDER:
                mat[i, TAG_ORDER.index(t)] = 1.0
    return mat


def attr_schema(df):
    """Top ATTR_KEYS with a fixed value vocab (first ATTR_VOCAB_SIZE + UNK)."""
    schema = {}
    for k in ATTR_KEYS:
        schema[k] = []
    for a in df["AttributesJson"].drop_nulls().to_list():
        for kv in parse_attributes(a):
            key, val = kv.split(": ", 1)
            if key in schema and val not in schema[key]:
                schema[key].append(val)
    out = {}
    for k, vals in schema.items():
        out[k] = vals[:ATTR_VOCAB_SIZE]
    return out


def attr_labels(df, schema):
    """(n, K, V) one-hot for present keys + (n, K) presence mask."""
    keys = list(schema)
    K, V = len(keys), max(len(v) for v in schema.values())
    y = np.zeros((len(df), K, V), dtype=np.float32)
    mask = np.zeros((len(df), K), dtype=np.float32)
    idx = {k: {v: i for i, v in enumerate(schema[k])} for k in keys}
    for i, a in enumerate(df["AttributesJson"].to_list()):
        for kv in parse_attributes(a):
            key, val = kv.split(": ", 1)
            if key not in idx or val not in idx[key]:
                continue
            y[i, keys.index(key), idx[key][val]] = 1.0
            mask[i, keys.index(key)] = 1.0
    return y, mask, keys


def category_labels(df) -> np.ndarray:
    """(n,) indices into the 645 leaf CategoryId vocabulary."""
    cats = df["CategoryId"].to_list()
    vocab = sorted(set(cats))
    return np.array([vocab.index(c) for c in cats], dtype=np.int64), vocab


def build_guidance_targets(rows, y, tags_mat, tokenizer, text_per_row, img_mask, max_len):
    """Weak attention targets over VLM token positions.

    For rejected rows with a tag: keyword-match (or quoted-brand strings from
    eval_reason) against the decoded text tokens; fallback = uniform over
    image tokens. Approved rows: all-zero (not supervised).

    rows: list of row dicts (needs eval_reason). text_per_row: (n, T) input_ids
    (padded). img_mask: (n, T) uint8. Returns (n, max_len) float32
    row-wise-normalized target distributions.
    """
    n = len(rows)
    targets = np.zeros((n, max_len), dtype=np.float32)

    def token_char_offsets(ids):
        """Per-token start offset in the decoded string."""
        offsets = []
        pos = 0
        texts = tokenizer.batch_decode(ids, clean_up_tokenization_spaces=False)
        for t in texts:
            offsets.append(pos)
            pos += len(t)
        return offsets

    for i, r in enumerate(rows):
        if y[i] != 1:
            continue
        if tags_mat[i].sum() == 0:
            continue
        tag = TAG_ORDER[tags_mat[i].argmax()]
        ids = [int(x) for x in text_per_row[i] if x != tokenizer.pad_token_id]
        if not ids:
            continue
        offsets = token_char_offsets(ids)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        text_lc = text.lower()
        spans = []  # (start, end) char spans
        if tag == "Marka Uyumsuzluğu":
            quoted = re.findall(r"['\"]([^'\"]{2,40})['\"]", r.get("eval_reason") or "")
            for q in quoted:
                ql = q.lower()
                start = 0
                while True:
                    p = text_lc.find(ql, start)
                    if p < 0:
                        break
                    spans.append((p, p + len(ql)))
                    start = p + 1
        else:
            for kw in GUIDANCE_KEYWORDS.get(tag, []):
                start = 0
                while True:
                    p = text_lc.find(kw, start)
                    if p < 0:
                        break
                    spans.append((p, p + len(kw)))
                    start = p + 1
        if spans:
            for s, e in spans:
                for j, off in enumerate(offsets):
                    if off < e and off + len(tokenizer.decode([ids[j]], clean_up_tokenization_spaces=False)) > s:
                        targets[i, j] += 1.0
        else:
            img_pos = np.flatnonzero(img_mask[i])
            if len(img_pos):
                targets[i, img_pos] = 1.0
        s = targets[i].sum()
        if s > 0:
            targets[i] /= s
    return targets


if __name__ == "__main__":
    df = load()
    y, mask, keys = attr_labels(df, attr_schema(df))
    print("attr schema:", {k: len(v) for k, v in attr_schema(df).items()})
    print("attr mask coverage:", mask.sum(0))
    print("tags:", tag_matrix(df).sum(0).astype(int).tolist())
    print("categories:", len(category_labels(df)[1]))
