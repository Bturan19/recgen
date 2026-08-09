# recgen — Technical Audit Brief

This document is the handoff for a fresh agentic session. It lists every
concern raised during development, the current status, and exactly how to
verify/fix each one. Work through it top to bottom; do not trust prior
conclusions — re-derive them.

## 0. Context

- Repo: `recgen` (public: https://github.com/Bturan19/recgen). Python 3.12,
  uv. SmolLM2-360M in `models/` (gitignored), SmolLM2-1.7B + Qwen3-0.6B also
  downloaded for comparison.
- Hardware: M1 Pro (32GB), MPS. Env quirk: run everything with
  `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` (libomp segfaults otherwise).
- Claims currently on the README/blog must ALL be re-verified after this
  audit; update docs if anything changes.

## 1. Label leakage (HIGH — was CONFIRMED, partially fixed)

- **Confirmed bug (fixed in the code, numbers pending re-run):** in the
  e-commerce next-item benchmark, users whose held-out (last) purchase also
  appears in their history (repeat purchase) leak the answer into the model
  input — the verbalized history literally names the test item. 1,134/25,000
  sampled users affected (~4.5%).
- Fix applied: `experiments/ecommerce/leakfree.py` filters such users before
  splitting; `sota_benchmark.py` and `strong_baselines.py` were patched to
  use it and to encode a fresh `user_emb_noleak.npy` cache.
- **TODO:**
  1. Re-run `sota_benchmark.py` + `strong_baselines.py` to completion.
  2. Sanity-check every other benchmark for the same class of bug:
     - IMDB: reviews are independent; row-level split — OK by construction,
       but verify no review text duplicates across train/test.
     - Adult / house prices / cardiotocography: row-level, no grouping — OK.
     - SMS spam: same-row, no leak — verify.
  3. Add a `test_leakfree.py` that asserts: no test user's history contains
     their target; the verbalized history text of any test user does not
     contain the test item's title; negative candidates exclude positive +
     history items (already done in code — re-check).
- Expected impact: all published e-commerce numbers will change; update
  README/blog/notes with the leak-free figures and add a "leakage audit"
  line to every results table.

## 2. Evaluation correctness (HIGH)

- `CatalogRankingHead.evaluate` and the sota/strong baseline scripts compute
  ranks in different ways — audit all of them:
  - Full-catalog rank: `np.argsort(-scores, axis=1)` then locate target —
    verify no tie-handling bias, no already-interacted-item masking that
    differs between models.
  - 2-stage candidate sets: MUST be identical for every model (they are —
    re-verify after the leak filter changes the user sample).
  - Bootstrap CI function samples with replacement over ranks; verify it
    cannot emit NaN (division guard exists — test it).
  - SASRec eval: uses scalar indexing now (advanced-indexing bug fixed);
    verify HR@10 in [0.1-ish, 1] and that val HR is NOT 1.0 (that was the
    bug symptom).
- Verify the recgen head's `best_mrr` early stopping uses the same val split
  definition across runs (it changes when users are filtered).

## 3. Baselines fairness (HIGH)

- ALS/ItemKNN/EASE are our own implementations; SASRec is a vanilla
  implementation (128-dim, 2 layers, 3 negatives, 20 epochs, val early-stop).
  They are NOT grid-searched. State this in any public claim.
- ALS: trained on all users' pre-test history (fair factors for test users).
  Re-verify after leak filter (matrix must exclude leaked users).
- Do NOT claim "SOTA". Claim only "beats our tuned-class baselines (ALS-128,
  SASRec) under the standard leave-one-out + 100-negative protocol on Amazon
  Musical Instruments". If the session wants a stronger claim, run official
  RecBole SASRec/BERT4Rec numbers on the same data+protocol, or cite
  published numbers for the same dataset (none known for Musical_Instruments
  — this is also why our claim is not "SOTA").
- Check the negative-sampling RNG seeds: identical seed per user index used
  by all models? (yes for SASRec/recgen eval; verify ALS/popularity use the
  same candidate sets).

## 4. Framework completeness (MEDIUM)

- Heads: ClassificationHead, RegressionHead, MultiLabelHead (new — has a
  unit test only, no real-data benchmark yet), CatalogRankingHead.
- "Any LLM -> classifier/regressor/ranker, multi-output, multi-label" —
  verify MultiLabelHead on a real small multi-label dataset (e.g., a HF
  multi-label text set) to back the claim.
- LoraHeadTrainer: verify `load()` path reproduces `fit()` results; the
  Adult LoRA experiment showed no gain at 4k rows — keep that negative
  documented.
- Pipeline: `RecgenPipeline.transform/predict` on unseen data must not
  re-tokenize per call (cache) and must handle polars+pandas (polars support
  was added — test both).

## 5. Model/encoder correctness (MEDIUM)

- FrozenEncoder: non-finite pooled values are zeroed with a warning (MPS
  fp16 instability, 67/5574 SMS rows). Audit: does zeroing rows bias any
  benchmark? Count zeroed rows per cache and report.
- Verify pooling correctness: masked mean (divide by attention count), last
  token = last non-pad. Add unit tests with tiny hand-computed examples.
- Embedding cache: sha256 of joined texts; verify staleness detection works
  when texts change (test exists — extend to negative case with same-length
  different texts).

## 6. API/serving correctness (MEDIUM)

- `api/app.py`: `/v1/encode`, `/v1/rank`, `/v1/rank_all`, `/health`.
  - `rank_all` caches catalog embeddings under a FIXED key
    ("catalog_default") — verify the cache key includes the model/pooling,
    else switching backends returns stale embeddings. (BUG candidate.)
  - Verify request-size limits, error codes, and that the model is loaded
    lazily (first request latency acceptable).
- Dockerfile: builds? (CPU-only path, `RECGEN_DEVICE=cpu`.)

## 7. Repo hygiene / IP (MEDIUM)

- `docs/product.md` was REMOVED and purged from git history (moved to
  `../recgen-private/product.md` — outside repo). Verify: `git log --all --
  docs/product.md` is empty; remote 404s; no other sensitive files
  (`.cache/`, `models/`, `data/`, `checkpoints/` are gitignored — verify
  `.env`/tokens are not committed anywhere; check `git ls-files`).
- `ROADMAP.md` is the public product-facing doc — keep it non-sensitive.
- CI workflow exists locally but is NOT pushed (OAuth token lacks `workflow`
  scope). TODO: run `gh auth refresh -s workflow --hostname github.com`,
  then commit `.github/workflows/ci.yml` and push.

## 8. Final deliverables of this audit session

1. `scripts/audit_report.md` — one page per section above: status, evidence
   (commands + outputs), fixes applied.
2. Leak-free headline numbers in README/blog/notes, each with a one-line
   "audit: leakage-free by construction (see AUDIT.md)" note.
3. Unit tests added for: leak filter, pooling, cache staleness, CI-safe
   heads, bootstrap CI.
4. Push everything; report the diffs to the previous published numbers.

## Commands

```bash
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
uv run python experiments/ecommerce/sota_benchmark.py      # ~1h (encode + SASRec)
uv run python experiments/ecommerce/strong_baselines.py    # full-catalog table
uv run pytest tests/ -q
uv run python scripts/benchmark.py --model smol360
uv run python scripts/summarize.py                          # results.csv
```
