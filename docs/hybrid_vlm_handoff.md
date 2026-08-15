# Handoff: Hybrid VLM + Learned Query Tokens + Specialized Heads

**Goal of the next experiment:** replace the pooled-embedding pipeline with a
small fine-tuned VLM that outputs a *hidden-state sequence*, attends to it via
*learned query tokens*, and feeds *4 specialized heads* (moderation, category,
attribute, rejection-tag) — with multi-task training including attention
guidance. This is the "kill the prompt, put the rules in the weights" study.

Write date: 2026-08-14. Everything below is verified on this machine.

---

## 0. TL;DR of the proposal (the new architecture)

```
5 images + title + desc + category
        │  (no verbalization of images — pixels go in directly)
        ▼
small VLM (2B class, fine-tuned)  ── ONE prefill ──►  hidden states [h1..hN]
        │                                            (sequence, NOT pooled)
        ▼
4 learned query tokens:  q_moderation  q_category  q_attribute  q_tag
        │  (each cross-attends to [h1..hN], learns to focus on
        │   violation/category/attribute/tag-relevant regions)
        ▼
4 parallel heads:  sigmoid │ hyperbolic lookup │ grouped softmax │ CE
        ▼
Onay/Red │ CategoryId │ attribute K-V pairs │ rejection tag
```

Key difference vs recgen: no mean-pooling (the bottleneck that lost
cross-modal detail), no 7500-token prompt (rules live in weights), and each
head sees a *targeted* representation instead of one averaged vector.

---

## 1. Machine & environment quirks (critical)

- M1 Pro 32GB, macOS, MPS. **Always run with:**
  `export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`
  and for big models: `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- `uv run` **auto-syncs uv.lock and reverts ad-hoc pip installs.** Use
  `uv run --no-sync python ...` after `uv pip install --python .venv/bin/python ...`,
  or `uv run --with "pkg==ver" python ...` for ephemeral overlays.
  (`uv pip install` WITHOUT `--python .venv/bin/python` installs into the
  conda base env — a known trap in this shell.)
- transformers in the project env is **5.14.1** (pinned via uv.lock).
  Trendyol-Vision-Flash needs 4.56.2 (`uv run --with "transformers==4.56.2"`).
- MPS fp16 LoRA/fine-tuning is unstable: 3B VLM training hit NaN (fixed with
  lr 1e-4 + grad clip + NaN-skip) and OOM at batch>2. fp16 fine-tuning a 2B
  VLM will be ~3-8s/sample → budget 6-12h per epoch. Plan small epochs.
- The image CDN (img.pzrmcdn.com) blocks curl/requests (Cloudflare bot
  management) — **headless Chrome passes**: see `scripts/download_images_browser.py`
  (Playwright + system Chrome + Chrome UA).

## 2. Assets on disk (do not delete)

| asset | path | notes |
|---|---|---|
| dataset parquet | `data/qwen_dataset/sample_4k.parquet` | 4000 rows; the HF repo is now PRIVATE/GONE — this local copy is the only one |
| product images | `data/qwen_images/{ProductId}/{i}.jpg` | all 11,283 re-downloaded (310MB), 512px, unique per product (no URL dupes → no leakage) |
| text embeddings | `.cache/moderation/smol17/text_emb.npy` | SmolLM2-1.7B mean-pool, (4000, 2048), on the verbalized listing |
| image embeddings | `.cache/moderation/siglip/img_emb.npy` | SigLIP2-SO400M pooler, (4000, 1152) |
| OCR text | `.cache/moderation/ocr_text.npy` | Apple Vision OCR of images (2073/4000 products have text) |
| OCR-in-context emb | `.cache/moderation/smol17/textocr_emb.npy` | 1.7B embeddings of product+OCR text |
| models | `models/SmolLM2-360M`, `SmolLM2-1.7B`, `SigLIP2-SO400M`, `Trendyol-Vision-Flash`, `SmolLM2-135M`, `Qwen2.5-0.5B` | SigLIP2+SmolLM2 needed for the old pipeline; Trendyol (2GB) optional for captioner use; the 0.5B/135M are tiny-FT leftovers |
| live case images | `data/live_cases/{nipple-pads,canvas-toy,bikini,pareo}/` | 4 problem products, 3 imgs each, fetched from the live site |
| results | `experiments/results/results.csv` (gitignored) | all recorded numbers |

## 3. Dataset deep-dive (the target)

- 4000 products, 40 cols. Labels from a single Gemini-3.1-flash-lite eval run.
- **Target:** `eval_decision` → 1="Reddedildi" (862, 21.5%), 0="Onaylandı".
- **Aux labels (multi-task material):**
  - `eval_rejection_tag` (list): 23 tags; top: Marka Uyumsuzluğu (348),
    Başlık/Resim/Açıklama Uyuşmazlığı (248), Cinsellik (125), Sağlık Beyanı (120),
    İletişim ve Yönlendirme (44), Yetersiz Bilgi (23), Yasa Dışı (11)...
  - `eval_reason` (Turkish text): the Gemini rationale (e.g., "görsellerde
    'Quota' markası görülüyor") — usable as weak localization signal for
    attention guidance (see §7.5) and as a secondary generation target.
  - `eval_suggested_category` (549 uniques), `eval_incorrect_category`,
    `eval_insufficient_description`, `eval_incorrect_attribute` flags.
- **Leakage:** ProductIds unique, image URLs unique per product (no dupes),
  no GroupCode training/test mixing in the standard split. Stratified
  80/10/10 split with seed 0 (`experiments/moderation/data.py:stratified_split`)
  — test = 402 rows (87 rejected / 315 approved). **Use this exact split for
  comparability with every number below.**
- **67% of rejection reasons are image-driven** (brand on box, specs printed
  on product, title/image mismatch, sexual imagery). This is why the old
  pipeline plateaued and why the new study must be pixel-first.
- The 4 live cases (data/live_cases/) are examples of moderation failures on
  the live site: wrong category (nipple pads under "Anne Bebek"), adult-toy
  canvas, generic bikini title, pareo with an ID/domain junk in the title.

## 4. Evidence base — what was already tried (all measured)

| approach | result | where |
|---|---|---|
| text-only 1.7B + MLP | 0.843 acc / 0.832 auc | text_baseline.py |
| text+SigLIP → MLP | 0.861 / 0.869 | multimodal.py |
| text+SigLIP → **weighted LGBM + tuned th** | **0.878-0.886 / 0.881** | fusion runs |
| same, 5-fold CV (honest) | **0.817 ± 0.034 / 0.820** | CV script in session |
| +OCR embeddings / multi-task tags / ensembles | no gain | fusion.py, multitask.py |
| Qwen2.5-VL-3B zero-shot judge | 0.55 (over-rejects) | vlm_judge.py |
| Trendyol-Vision-Flash zero-shot | 0.376 (recall 0.77, precision 0.23 — ultra-conservative) | trendyol_judge.py |
| SmolLM2-135M / Qwen2.5-0.5B full FT (gen) | ≈ majority (0.777 / 0.785) | tiny_finetune.py |
| Qwen2.5-VL-3B LoRA distillation | 0.76-0.79 val (overfits 4k; MPS ~7h/epoch) | vlm_lora.py |

**Key conclusions that motivate the new study:**
1. Mean-pooling is the bottleneck — the head pipeline misses exactly the
   fine cross-modal signals (brand/spec OCR, colors, category-vs-content).
2. Zero-shot VLMs disagree with Gemini's *style* (strict vs lenient-with-notes)
   → any VLM used as the decision-maker must be **distilled on the labels**.
3. Generative fine-tuning on 4k rows overfits/memorizes (loss→0 fast) —
   the new architecture mitigates with: fewer params than 3B, multi-task
   supervision (4 heads = more signal per row), and attention guidance.
4. VLM runtime on MPS is real but manageable for a 2B model.

## 5. The architecture — full spec

### 5.1 Encoder
- Small VLM, **≤2B class**, images direct (no verbalize). Candidates, in
  order of recommendation:
  1. **Trendyol-Vision-Flash (InternVL3.5-1B) — ALREADY ON DISK, start here.**
     - 1B params (smallest = fastest MPS fine-tune), fine-tuned on e-commerce
       catalog data (brand detection, attribute extraction, content safety,
       Turkish primary) — domain-specialized for exactly this task.
     - Gotchas: custom code is broken on transformers 5.14 and 4.49/4.46;
       **works on transformers 4.56.2 via the manual greedy decoder** in
       `experiments/moderation/trendyol_judge.py` (pattern: build
       `<img><IMG_CONTEXT>×256×n_img</img>` tokens, `model.img_context_token_id=151671`,
       `image_flags=ones(n_img)`, loop `model(input_ids, pixel_values, image_flags)`).
     - For the hybrid we need `output_hidden_states=True` on that forward —
       the signature accepts it; verify the returned sequence covers all
       text+image tokens (if InternVL's custom forward only returns decoder
       hidden states, take those; the query block just needs a token-level
       sequence).
     - Run under `uv run --with "transformers==4.56.2"` (project env is 5.14.1).
  2. **SmolVLM2-2.2B** (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`, download
     ~4.4GB) — standard transformers API, fastest on MPS (~2-3x vs
     Qwen-VL-2B), SigLIP tower + SmolLM2 text; weaker OCR than Qwen. Best
     LoRA fine-tune target for iteration speed.
  3. **Qwen2-VL-2B-Instruct** (download ~4.5GB) — strongest OCR/text-in-image
     + Turkish; quality tier, slower on MPS.
  4. PaliGemma-3B — slower on MPS; skip unless 1-3 fail.
- Extraction: run ONE forward, take `outputs.last_hidden_state` → (B, T, D)
  for the FULL sequence (visual+text tokens interleaved by the VLM).

### 5.2 Learned query tokens
- 4 learnable vectors (D-dim each): `q_mod`, `q_cat`, `q_attr`, `q_tag`.
- Cross-attention: queries attend to [h1..hN] (standard cross-attn block with
  q=queries, kv=hidden states, 2-3 layers, dropout) → 4 output vectors
  (D-dim each). BLIP-2/Q-Former pattern; can be trained from scratch on top
  of the frozen VLM first (cheap) then jointly.
- The cross-attention block must also be LoRA-able if fine-tuning the VLM.

### 5.3 Heads (4 parallel)
1. **Moderation:** `MLP(q_mod) → sigmoid` → P(Reddedildi).
2. **Category:** option A: linear head over `q_cat` → CE over 645 leaf
   categories; option B (the proposal): hyperbolic/Poincaré embedding of the
   category hierarchy, score = -distance(q_cat_proj, node_emb), argmin over
   nodes — hierarchical structure with explicit parent-child semantics.
   Start with A (flat CE) as baseline; B as the experiment.
3. **Attributes:** `MLP(q_attr)` → grouped softmax over the attribute schema
   (key per group: Renk, Beden, Malzeme...). The dataset has
   `AttributesJson` as noisy weak labels (717 nulls) — accept that.
4. **Rejection tag:** `MLP(q_tag)` → CE over 23 tags (multi-label with BCE,
   since tags are lists; majority-empty — use weight masking).

### 5.4 Training objective (multi-task)
```
loss = w1*BCE(mod) + w2*CatLoss(cat) + w3*GroupedSoftmax(attr)
     + w4*BCE(tags) + w5*AttentionGuidance
```
- Weights w1..w4: start (1, 0.5, 0.25, 0.5); tune on val.
- **AttentionGuidanceLoss (the differentiator):** for q_mod, take its
  attention distribution over the hidden sequence; the ground-truth "violation
  regions" are NOT labeled, so use weak proxies:
  (a) `eval_reason` keyword matching (e.g., tag=Sağlık Beyanı → attend to
  text tokens containing health keywords; tag=Marka Uyumsuzluğu → tokens of
  the quoted brand string in the reason);
  (b) fall back to uniform over image tokens when no keyword matches.
  Loss = KL(attn || target_dist). Start weak (w5=0.1) to avoid fighting the
  main losses.
- Class imbalance: 21.5% reject → use weighted BCE (scale_pos_weight ~3.6)
  as in the LGBM success; calibrate thresholds on val (the LGBM best
  threshold was ~0.28).

## 6. Implementation plan (files to create, in order)

Phase 0 — plumbing (0.5 day):
1. `experiments/hybrid/data.py` — reuse `experiments/moderation/data.py`
   (load, verbalize text fields, stratified_split, image_paths).
2. `experiments/hybrid/vlm_hidden.py` — encoder per §5.1. **Start with
   Trendyol-Vision-Flash** (on disk, 1B — fastest): one forward per product
   (up to 3 images, 448px, `output_hidden_states=True`), return
   last_hidden_state + the mask of image/text token positions (needed for
   attention guidance). Cache hidden states to
   `.cache/hybrid/trendyol/hidden_{split}.npy`. For SmolVLM2-2.2B/Qwen2-VL-2B
   (standard transformers) the same helper works via `AutoModelForImageTextToText`.
   Size watch: 4k rows × ~800 tokens × D × 2B — Trendyol D=1024 (≈6.5GB
   fp16); SmolVLM2 D=2048 (≈13GB) — store per-split chunks or subsample
   tokens.
3. `experiments/hybrid/heads.py` — the cross-attn query block + 4 heads.
4. `experiments/hybrid/train.py` — multi-task trainer (MPS, batch 2-4,
   grad accum, clip 1.0, lr 1e-4, early stop on val acc, NaN-skip,
   checkpoint every 200 steps — replicate the hardening lessons from
   vlm_lora.py).

Phase 1 — frozen-VLM baseline (the cheap first result, 1 day):
- Keep VLM frozen; train ONLY the query block + heads on the cached hidden
  states. This isolates the query-token contribution vs the old pooled head
  (expect it to already beat 0.817 CV if the hypothesis is right, because
  queries see the full sequence instead of the mean).
- Report: acc/F1/AUC on the SAME test split + 5-fold CV + per-tag error
  breakdown (copy the analysis script pattern).

Phase 2 — LoRA fine-tune of the VLM (2-4 days, overnight runs):
- LoRA r=8 on the LM projections (vision tower frozen), joint with the query
  block. 4k rows → keep epochs ≤ 2-3, patience 1-2, watch val.
- Add AttentionGuidanceLoss (weak), then ablate it.

Phase 3 — category & attributes deep-dive:
- Flat CE vs hyperbolic category head; attribute grouped softmax with
  AttributesJson supervision; report per-head metrics.

Phase 4 — serving projection:
- Reuse the MPS timings; the user's A100 math (~50ms prefill 2B, ~1200/s at
  batch 64 → ~100M/day, 2M/day target ≈ 2% of one A100) is directionally
  right — verify on a GPU if one becomes available.

## 7. Evaluation & reporting rules (keep honest)

- **Always report both single-split AND 5-fold CV.** The single split gave
  0.886 vs CV 0.817 — a 7pt gap from the 402-row test. All claims must quote
  the CV number as the headline.
- Keep the SAME stratified split (seed 0) so every number in §4 is comparable.
- Report per-tag recall for the moderation head (the old pipeline missed
  Marka Uyumsuzluğu + Başlık/Resim Uyuşmazlığı most — those must improve).
- Threshold tuning on val only (LGBM precedent: ~0.28).
- The "ceiling" is Gemini-flash-lite label noise: expect ~0.90+ to be the
  practical max without re-labeling.

## 8. Environment / commands cheat-sheet

```bash
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
uv run --no-sync python -u experiments/hybrid/...        # after ad-hoc pip installs
uv run --with "transformers==4.56.2" python ...          # only for Trendyol model
uv run pytest tests/ -q                                  # 30 tests, all pass
```

Downloads needed (if the team proceeds with 2B VLMs):
```
uv run hf download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B
uv run hf download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir models/SmolVLM2-2.2B
```

## 9. Risks & open questions

1. **4k rows:** multi-task (4 heads) helps, but the VLM LoRA phase overfits
   easily — keep LoRA small, epochs low, and lean on the frozen-VLM baseline
   as the safety result.
2. **MPS speed:** fine-tuning a 2B VLM ≈ 2-4s/sample → a 3-epoch run ≈
   6-12h. Budget overnight runs; SmolVLM2-2.2B is ~2-3x faster than Qwen-VL-2B;
   Trendyol-Vision-Flash (1B) is the fastest of all three.
3. **Hidden-state caching size:** Trendyol D=1024 (≈6.5GB fp16 for 4k×800);
   SmolVLM2 D=2048 (≈13GB) — chunk the cache or subsample tokens; the query
   block only needs them at train time.
4. **Attention guidance targets are weak** (keyword-matched from eval_reason)
   — if KL guidance hurts, drop w5; the multi-head structure alone is the
   main win.
5. **Category tree:** 645 leaves; hierarchical (hyperbolic) head needs the
   tree structure extracted from CategoryHierarchy strings — a small prep
   task; the flat head is the safe default.
6. **Trendyol-Vision-Flash quirks when used as the hybrid encoder:** custom
   code requires the transformers 4.56.2 overlay (`uv run --with
   "transformers==4.56.2"`); verify `output_hidden_states=True` returns the
   full token-level sequence (its forward signature accepts it; test on one
   product before the batch cache job). Its zero-shot DECISIONS are not
   aligned with the labels (0.376 as a judge) — that is irrelevant for the
   hybrid (we only consume its hidden states; the query block + heads learn
   the alignment).
7. **Data privacy:** the parquet is irreplaceable (HF repo gone). Never push
   it; data/ and models/ are gitignored.

## 10. First actions for the new session

1. Verify state: `ls data/qwen_images | wc -l` → 4000;
   `ls .cache/moderation/smol17/` → caches present; `uv run pytest tests/ -q`.
2. **No download needed to start**: Trendyol-Vision-Flash is on disk
   (`models/Trendyol-Vision-Flash`) — build `experiments/hybrid/vlm_hidden.py`
   around it first (transformers 4.56.2 overlay), and download SmolVLM2-2.2B
   (+ Qwen2-VL-2B as quality tier) only when needed for Phase 2.
3. Build `experiments/hybrid/` per §6 Phase 0-1 (frozen VLM + query block on
   cached hidden states) — this is the decisive cheap experiment.
4. Only then move to LoRA + attention guidance (Phase 2).
