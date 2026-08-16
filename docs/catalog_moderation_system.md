# Catalog Moderation & Enrichment System — Production Design Specification

**Goal:** a single, small fine-tuned VLM + learned query tokens + specialized heads that, in **one prefill per product** (no verbalization, no generation), decides listing suitability, corrects the category, fixes/enriches attributes, and emits an anomaly score — all strictly aligned with the marketplace catalog (4,000+ categories, per-category attribute schemas with predefined values).

**Status:** pilot-validated on 4k products (see §1). This document is the build spec: an agent reading it end-to-end should be able to build, train, evaluate, and serve the production system. Reference implementation: `experiments/hybrid/` in this repo (Phase 0-1 completed, measured).

**Date:** 2026-08-16.

---

## 0. Executive summary

```
[5 images + title + brand + description]      <- NO seller category, NO seller attributes in text
        │
        ▼
small VLM (1-2B, fine-tuned) ── ONE PREFILL ──► hidden states [h1..hN]
        │                                          (full token sequence)
        ▼
5 learned query tokens: q_mod q_cat q_attr q_tag q_recon
        │   (cross-attend to [h1..hN], learn what to focus on)
        ▼
5 heads:  sigmoid │ flat-CE over catalog │ grouped softmax │ BCE │ MSE
        ▼
Onay/Red + tag │ CategoryId + path │ attrs per catalog │ tag │ anomaly
```

Design rules that make it work in production:

1. **Blind input.** The VLM text contains only what the *marketplace knows*: title, brand, description (never the seller's declared category/attributes). Every head must infer from pixels + title + desc. This is what makes correction possible — a head that sees the seller's declaration will copy it (measured: 0/135 corrected categories matched Gemini's suggestion when the seller's category was in the text; identical category accuracy with/without it in the input, so the blind design costs nothing).
2. **Catalog is the output vocabulary.** The category head is a classifier over the *catalog's leaf set*. Attribute heads are grouped softmaxes over the *catalog's allowed values per key*. The model can never emit a value that does not exist in the catalog. (Pilot failure: 48-value data-frequency vocab truncated real values like `Elastan - Polyamid`, `EVA`, `80B` — the model literally could not say the truth. Catalog wiring fixes this by construction.)
3. **Category gates attributes.** After the category head runs, only the keys the catalog defines for that category are evaluated, with that category's value list. (Pilot failure: a dress got `Uyumlu Marka: Apple` — irrelevant keys were always scored.)
4. **Confidence-abstain.** No head output is authoritative below a calibrated threshold; low-confidence outputs are emitted as "inconclusive", not as confident guesses. (Pilot failure: `Materyal: Hakiki Deri` at 0.88 confidence was wrong — a frequency-prior artifact of noisy labels. Thresholds + abstention convert this into "inconclusive".)
5. **No generation.** Autoregressive decoding is banned in the hot path; throughput is prefill-bound (measured: ~60-100 products/s per A100 for a 1B model; the naive "1000 products/s" estimate is physically impossible for a dense 1-2B prefill — see §7).

---

## 1. Why this architecture (pilot evidence, all measured)

Baseline: 4,000 products, 645 leaf categories, Gemini-3.1-flash-lite labels. Frozen VLM (Trendyol-Vision-Flash, 1B) + query tokens + heads trained on cached hidden states. Single stratified split (seed 0) + 5-fold CV.

| task | pilot result | old pooled-embedding pipeline |
|---|---|---|
| moderation (CV auc) | **0.834 ± 0.020** | 0.820 |
| moderation (single test) | 0.80-0.85 acc / 0.86 auc / 0.62 f1 | 0.88 acc / 0.88 auc |
| category (645-way, blind input) | **0.82 acc / 0.88 leaf-or-parent** | 0.63 (mean-pool probe) |
| attributes (grouped softmax, blind input) | **0.80 overall** (Renk .76, Cinsiyet .94, Materyal .86) | 0.59 (mean-pool probe) |
| hyperbolic category head | 0.62 — **flat CE wins**, use flat | — |

Measured failure modes that shaped the design (each fixed by a design rule above):
- vocab truncation (rule 2), irrelevant-key noise (rule 3), high-confidence wrong values from label priors (rule 4), correction impossible with seller fields in text (rule 1).
- attention guidance (KL on q_mod attention vs keyword targets) was neutral on CV — keep it as an optional aux loss, not a headline feature.
- attribute *correction* (detect a wrong declared value) requires training on correction labels (the Gemini reason texts); value prediction from the listing alone cannot detect its own errors.

---

## 2. Catalog as the source of truth

Two JSON artifacts define the entire output space. They are the only thing the heads may emit.

### 2.1 taxonomy.json — category tree (4,000+ leaves)

```json
{
  "node_id": "cat_giyim_kadin_ic_giyim_fantezi_ic_camasiri_takimlari",
  "name": "Fantezi İç Çamaşırı, Takımları",
  "path": "Moda > Kadın Giyim > Kadın İç Giyim, Uyku Giyim > Fantezi Giyim > Fantezi İç Çamaşırı, Takımları",
  "parent": "cat_fantezi_giyim",
  "is_leaf": true,
  "attribute_keys": ["Cinsiyet", "Renk", "Beden", "Desen", "Kumaş Tipi", "Takım Sayısı", "Balenli"]
}
```

Requirements:
- Every node has a stable `node_id` (never reuse; deprecate instead).
- Leaves are the classification target of the category head. Internal nodes are used only for: hierarchical evaluation (leaf-or-parent metric), cold-start (new leaf = parent embedding init), and evaluation breakdown.
- The category head is **flat CE over leaves** (§5). A hyperbolic head is optional; the pilot measured it 20pt worse than flat at 4k rows — revisit only with >100k rows or when hierarchy-guaranteed outputs are legally required.
- **Cold start:** when a leaf is added, initialize its head logit row as the parent's row + small noise, and set its training weight to 0 until labeled products exist. Never let the head score unseen leaves above "reject-new-category" during the first N days.

### 2.2 attributes.json — per-category attribute schemas

```json
{
  "Renk": {
    "display_name": "Renk",
    "group": "visual",
    "applicable_categories": ["cat_giyim_kadin_...", "..."],
    "values": ["Siyah", "Kahverengi", "Kırmızı", "Dark Grey", "Çok Renkli"],
    "inferable_from": ["image", "title"],
    "missing_value_policy": "enrich"        // or "abstain"
  },
  "Beden": {
    "display_name": "Beden/Yaş",
    "group": "sizing",
    "applicable_categories": ["..."],
    "values": ["XS", "S", "M", "L", "XL", "80B", "39", "L - XL"],
    "inferable_from": ["title", "description"],
    "missing_value_policy": "abstain"       // size from an image is unreliable
  }
}
```

Rules:
- `values` is the *complete* allowed list — the grouped softmax head is built directly on it. Any catalog change = rebuild the head's output layer (a config+initialization change, no re-training of the VLM).
- `inferable_from` drives the enrichment task: keys marked `["image"]` are trained on visual evidence; keys marked `["title"]` mainly copy from text; keys with `missing_value_policy: "abstain"` are never guessed — the system emits "missing" instead of a low-confidence value.
- The **key gate** is per (category, key): a small sigmoid head on `q_attr` that predicts whether key K applies to this product, multiplied by the catalog's hard mask for the predicted category. Predicted value × gate × threshold → final output.

### 2.3 Alignment invariants (unit-tested)

1. Every head output value ∈ catalog vocab (softmax cannot emit otherwise).
2. Attributes are only emitted for keys in `attributes.json[K].applicable_categories` ∩ predicted-category's key list.
3. No attribute is emitted with confidence < calibration threshold (per key, calibrated on a held-out set).
4. Category predictions are always valid leaf `node_id`s with a valid `path`.
5. `is_category_corrected` = (predicted leaf ≠ seller's declared leaf) AND predicted confidence ≥ threshold.

---

## 3. Architecture

### 3.1 Encoder (one prefill, no generation)

- **Model:** 1-2B class VLM. Candidates, by preference:
  1. **Trendyol-Vision-Flash (InternVL3.5-1B)** — e-commerce-tuned, Turkish-first, smallest/fastest. Custom code; requires `transformers==4.56.2` pinned (broken on 4.46/4.49/5.14). Extract hidden states with `output_hidden_states=True` → `last_hidden_state` covers the full interleaved sequence (verified: image tokens at `<IMG_CONTEXT>` positions).
  2. SmolVLM2-2.2B — standard transformers API, fastest fine-tune on MPS.
  3. Qwen2-VL-2B — best OCR/text-in-image + Turkish; slowest.
- **Input text (blind):** `Ürün Başlığı: {title}\nMarka: {brand}\nAçıçlama: {desc}` (desc truncated ~350 chars; up to 3 images at 448px). No instruction, no rules, no seller fields.
- **Extraction:** one forward → `(B, T, D)` hidden states. Cache them per split (production: cache during ingestion, or run inline if prefill is cheap enough at your volume).

### 3.2 Query block (BLIP-2/Q-Former pattern)

- 5 learnable query vectors `q_mod, q_cat, q_attr, q_tag, q_recon` (D-dim each).
- 2-layer pre-norm cross-attention: queries attend to the full hidden sequence (key-padding-mask from lengths); residual self-attention + FFN between layers.
- Output: 5 vectors (D-dim) + last-layer attention weights `(B, 5, T)` (for attention guidance and interpretability).
- Size: ~35M params for D=1024. Trainable from scratch over the frozen VLM.

### 3.3 Heads

| head | input | architecture | output |
|---|---|---|---|
| Moderation | q_mod | MLP → sigmoid (BCE, pos_weight ≈ class ratio) | P(Reddedildi), decision at calibrated threshold |
| Category | q_cat | flat MLP → logits over catalog leaves (CE) | top-k leaf ids, paths, confidence |
| Attributes | q_attr | shared MLP + per-key linear → grouped softmax over catalog values per key; key gate per (category, key) | per-key value, confidence, missing/enrichment flags |
| Rejection tag | q_tag | MLP → 23 logits (BCE, masked to rejected rows only) | multi-label tags |
| Reconstruction (anomaly) | q_recon | MLP decoder → MSE vs reference encoding of the sequence (trained on approved-only rows) | anomaly score = reconstruction error |

### 3.4 Output contract (ProductOutput)

```python
@dataclass
class ProductOutput:
    product_id: str
    # moderation
    is_approved: bool
    rejection_tag: list[str]            # multi-label, empty if approved
    anomaly_score: float                # [0, 1], high = deviation from clean manifold
    # category
    predicted_category_id: str          # catalog leaf node_id
    predicted_category_path: str
    category_confidence: float
    is_category_corrected: bool
    # attributes
    predicted_attributes: dict[str, str]        # key -> catalog value
    attribute_confidence: dict[str, float]
    missing_attributes: list[str]               # keys the catalog requires, seller left empty
    corrected_attributes: list[str]             # keys where predicted != declared (conf ≥ threshold)
    # reconstruction
    reconstruction_error: float
```

---

## 4. Data requirements

Per task, minimum labeled rows (pilot data = 4k rows total):

| task | label source | min rows (pilot) | notes |
|---|---|---|---|
| moderation | binary decision | 4k (862 rejects) | stratified, class-weighted |
| category | catalog leaf id (seller-declared, verified) | 4k | 645 leaves; for 4k leaves target ≥10k rows |
| attributes | per-key catalog value | 3.3k (571 with Materyal) | noisy seller labels accepted; correct with reason texts |
| tag | rejection tag list | 862 rejects | mask approved rows |
| correction labels | reviewer reason texts | ~1.2k category, ~1.9k attribute | parse key+wrong-value+correct-value triples |
| anomaly | approved-only rows | ~3.1k | clean manifold |

Labeling strategy at production scale (2M products/day):
- **Auto-label tier:** current system + rule checks → confident labels (approve/reject, category).
- **Reviewer tier:** stratified sample (~1-2%) + all low-confidence outputs → human moderation queue, which doubles as the correction-label source.
- **LLM-judge tier:** the pilot's Gemini-flash-lite style judge is a *label generator*, not a serving component. Keep it only for labeling and audits; never in the hot path (latency, cost, and its judgment style differs from marketplace rules).

The pilot's biggest data lesson: **label noise is the ceiling** — the reviewer-correction labels (reason texts) matter more than model architecture for attribute tasks.

---

## 5. Training

### 5.1 Loss

```
L = w1·BCE(mod, pos_weight=3.6)
  + w2·CE(cat)                         # flat over catalog leaves
  + w3·Σ_keys grouped-softmax(attr)    # masked to present keys & applicable keys
  + w4·BCE(tags, masked to rejects)
  + w5·MSE(recon)                      # approved-only rows
  + w6·KL(q_mod attention || guidance) # optional, w6=0.1, neutral in pilot
```

Start weights: (1, 0.5, 0.25, 0.5, 0.5, 0.1).

### 5.2 Phases

1. **Phase 0 — frozen VLM baseline (the decisive cheap experiment).** Extract + cache hidden states once. Train ONLY query block + heads. This was the pilot's entire result set and it already beats the old pipeline. Must be reproduced before anything else.
2. **Phase 1 — LoRA the VLM** (r=8 on LM projections, vision tower frozen), joint with the query block. Expect 6-12h/epoch on MPS for a 2B model; epochs ≤ 2-3 with patience 1-2. If it overfits (it will at 4k rows), the Phase-0 baseline is the fallback.
3. **Phase 2 — correction heads.** Train the attribute head on reviewer-corrected values (key+value triples parsed from reason texts) and a dedicated "verify" mode: when the seller declared key K, the system predicts match/conflict/inconclusive for (declared value, image) instead of re-guessing the value.

### 5.3 Training hygiene (all pilot-verified lessons)

- MPS: fp16 fine-tuning is unstable — fp32 heads, clip 1.0, NaN-skip, lr 1e-4, gradient accumulation.
- **Checkpoint selection must be per-task.** The pilot lost 15pt of category accuracy by selecting the checkpoint on moderation AUC. Train once, checkpoint every task metric, ship per-task checkpoints (heads are tiny; multiple is cheap).
- Always `torch.manual_seed(seed)` + numpy seed; runs otherwise vary ±5pt.
- Threshold calibration on the val split only (never test).
- Report **both** the single split and 5-fold CV; CV is the headline (single-split overstated the pilot by 5-7pt).
- Attention guidance: keep at w6=0.1 or drop; it did not help CV in the pilot.

---

## 6. Evaluation protocol (production)

Per release:

1. **Moderation:** accuracy, F1, AUC, per-tag recall (the pilot's known weaknesses: Marka Uyumsuzluğu, Başlık/Resim Uyuşmazlığı), calibration curve, threshold-recall tradeoff for the compliance team.
2. **Category:** top-1 and top-5 accuracy, leaf-or-parent accuracy (hierarchical), per-department breakdown, and *correction precision* = P(category truly wrong | system says corrected), measured against reviewer labels.
3. **Attributes:** per-key accuracy for present keys (enrichment evaluation: rows where the key was absent in training but present in the catalog — simulate by masking, as the pilot's blind-cache did), per-key calibration, and *correction precision* per key.
4. **Anomaly:** separation of approved vs rejected products by reconstruction error; false-positive rate at the operating threshold.
5. **Safety:** golden set of "never approve" cases (Cinsellik, yasa dışı, health claims...) — regression-gated in CI.
6. **Catastrophic-failure review:** all false-approve predictions reviewed by a human in the first week after each release.

Acceptance gates (suggested): moderation CV-AUC ≥ pilot (0.83) at minimum, category top-1 ≥ 0.80 on the new taxonomy, per-key attr accuracy ≥ 0.75 for `["image"]` keys, correction precision ≥ 0.6 on the reviewer set.

---

## 7. Serving & throughput (measured + derived)

**The architecture's throughput story: one prefill, zero generation.** All heads together are <10ms.

| hardware | measured latency | throughput |
|---|---|---|
| M1 Pro MPS, 1B model, 3 imgs + text (~1100 tokens) | 1.43 s/product | 0.70 products/s (batching: no gain — compute-bound) |
| M1 Pro CPU (8 threads, fp32) | 3.96 s/product | 0.25 products/s |
| A100 80GB, 1B model (derived) | ~10-15 ms/product | **~60-100 products/s** (batch 64, flash-attn, padding bucketing, bf16) |
| A100, 2B model (derived) | ~20-30 ms/product | ~25-50 products/s |

FLOP floor for a 1B prefill of ~1100 tokens: `2·1e9·1100 ≈ 2.2 TFLOP`; A100 bf16 ≈ 312 TFLOP/s → even at 100% MFU it's ~7ms/product, i.e. ~140/s max. **Any claim above ~150/s per A100 for a dense 1B prefill is wrong** (the original "1000 products/s" estimate is off by ~10-20x). 2M products/day = 23 products/s sustained → **one A100 is sufficient with headroom; two for batch peaks.**

Cost drivers per product: images (3×448² through the ViT) ≈ 2/3 of the cost — reducing to 1 image at the loss of recall, or 336px, trades accuracy for throughput. `use_flash_attn=True` requires a model with flash-attn support (Trendyol's custom code does not support it on transformers 4.56).

Serving topology:

```
ingest (5 imgs + metadata) → VLM prefill (GPU, batch 64) → hidden states
   → query block + heads (CPU or GPU, trivial) → ProductOutput
   → catalog-validity validator (unit-tested invariants) → output topic
```

Backfill vs incremental: incremental only (2M/day); no full re-encode needed when heads or catalog change (heads are tiny; only catalog *vocab* changes need a head rebuild).

---

## 8. Implementation plan (build order for an agent)

Repository layout (production):

```
catalog/
  taxonomy.json, attributes.json            # source of truth
  validator.py                             # invariants §2.3 (unit-tested)
data/                                      # gitignored
  ingestion, labeling, splits
encoder/
  vlm_hidden.py                            # one-prefill extraction (transformers 4.56.2 pinned)
  cache_store.py                           # hidden-state cache (memmap per split)
model/
  heads.py                                 # query block + 5 heads
  hyperbolic.py                            # optional category head (pilot: flat wins)
  train.py                                 # multi-task trainer (per-task checkpointing)
  predict.py                               # single-product inference (demo + smoke tests)
eval/
  eval_heads.py                            # full battery §6
  cv.py                                    # 5-fold CV
serving/
  server.py                                # batched prefill worker
  validator wiring, metrics, thresholds.json
tests/                                     # catalog invariants, output schema, golden set
```

Build order with acceptance criteria:

1. **Catalog JSONs + validator** (day 1). All invariants green on unit tests. Nothing else starts before this.
2. **Encoder + cache** (days 1-2). One product through the VLM returns `(T, D)` hidden states covering image+text tokens; cache reproducible, resumable.
3. **Heads + frozen training** (days 2-5). Reproduce pilot ballpark on your data: moderation CV-AUC ≥ 0.83, category top-1 ≥ 0.80, attr overall ≥ 0.75 (blind input). This is the go/no-go.
4. **Correction + anomaly heads** (week 2). Correction labels parsed from reviewer reasons; verify-mode trained; anomaly head on approved-only rows.
5. **Serving worker + thresholds + golden CI** (week 2-3). Batched prefill, per-key calibrated thresholds, catastrophic-failure review loop.
6. **LoRA fine-tune** (week 3+, only if frozen baseline underdelivers on text-in-image cases like brand/OCR). Overnight runs, patience 1-2, fall back to frozen checkpoint.
7. **Scale-up** (month 2): taxonomy to 4k leaves, label pipeline, reviewer queue integration, cost monitoring.

Critical environment notes (pilot-verified): pin `transformers==4.56.2` for Trendyol-Vision-Flash; `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` on MPS; use `uv run --with "transformers==4.56.2"` or a dedicated venv.

---

## 9. Operations runbook

- **Rule changes** (policy updates): never touch the VLM. Change the thresholds/labels → retrain heads (minutes-hours) → ship. Products are not re-encoded.
- **Catalog changes:** new values/keys = rebuild head output layer (config); new leaves = cold-start init (§2.1); removed values = drop from softmax (predictions renormalize).
- **Drift monitoring:** per-day distributions of P(approve), top-category entropy, anomaly-score mean, per-key confidence means, threshold-crossing rates. Alert on >2σ shifts.
- **Review loop:** all low-confidence outputs + all corrections flow to the reviewer queue; reviewer verdicts become next week's correction labels (self-improving).
- **Model updates:** A/B on a shadow stream; compare decision agreement, correction precision, and catastrophic-failure counts before rollout.
- **Privacy:** product data is irreplaceable in this pilot (the 4k parquet's HF repo is gone) — data/ and models/ are gitignored; never commit them.

---

## 10. Known pitfalls (pilot log — do not relearn these)

1. Vocab truncation silently caps the model's expressiveness (fix: catalog vocab, §2).
2. Heads copy seller fields from text if present (fix: blind input, rule 1).
3. High-confidence wrong attribute values from label priors (fix: confidence-abstain + correction labels).
4. Per-task checkpoint selection matters (fix: per-task checkpoints, §5.3).
5. Batching gains nothing on MPS; everything on CPU is ~4x slower than MPS (throughput math must be per-target-hardware).
6. The custom Trendyol code breaks on most transformers versions; pin 4.56.2 and keep the manual greedy-decoder pattern out of the hot path.
7. Guidance loss (KL) is neutral on 4k rows — don't build the system around it.
8. "1000 products/s per A100" is a myth for dense 1-2B prefill (FLOP-bound, §7).
