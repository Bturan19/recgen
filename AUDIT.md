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
- **Audit status (2026-08-09): FULL AUDIT COMPLETE.** All sections below
  were re-derived in `scripts/audit_report.md`. Published numbers in
  README/blog/notes were updated to the final leak-free (title-aware) run.

## 1. Label leakage (HIGH — FIXED and re-verified)

- Item-id leak (repeat purchase): 1,134/25,000 users (~4.5%) — excluded by
  `experiments/ecommerce/leakfree.py`.
- **Title-level leak (found in audit):** a different item_id sharing the
  held-out item's title also leaks the answer into the verbalized history.
  1/23,866 users, in train only (impact nil). `leak_mask(splits, meta=meta)`
  now excludes these too. Final clean set: 23,865 users / 13,136 items /
  2,387 test users.
- All other benchmarks audited: IMDB (9/3,000 duplicate texts, all
  label-consistent), Adult, house prices, SMS, cardiotocography — row-level,
  OK.
- Tests: `tests/test_leakfree.py` asserts item-id, title-in-text, negative
  candidates, split disjointness, and pins the clean count at 23,865.

## 2. Evaluation correctness (HIGH — verified)

- `CatalogRankingHead.evaluate`, `sota_benchmark.hr_ndcg`, and
  `strong_baselines.full_ranks` all rank via `np.argsort(-scores)` and locate
  the target; no model-dependent tie/masking differences.
- 2-stage candidates identical for every ranker (popularity ∪ embedding-kNN,
  300).
- Bootstrap CI has a `where=ranks>0` guard; tested for NaN (incl. rank-0).
- SASRec eval uses scalar indexing; val HR peaked 0.1412 (not 1.0).
- Head `best_mrr` early-stops on the same internal 90/10 val in both scripts.

## 3. Baselines fairness (HIGH — verified; state honestly)

- ALS/ItemKNN/EASE own implementations, SASRec vanilla (128-dim, 2 layers,
  3 negatives, 20 epochs, val early-stop). NOT grid-searched — state this.
- ALS trained on all users' pre-test history; matrix excludes leaked users.
- Do NOT claim "SOTA". Claim only "beats our tuned-class baselines under
  leave-one-out + 100-negative on Amazon Musical Instruments".
- Same-seed candidate sets confirmed for all models.

## 4. Framework completeness (MEDIUM — done)

- `experiments/multilabel.py` (new): go_emotions multi-label benchmark —
  recgen micro-F1 0.400 vs TF-IDF 0.376, LightGBM-on-h 0.258. Backs the
  multi-label claim.
- `RecgenPipeline` polars+pandas tested (cache hit across fit/transform/
  predict) — `tests/test_pipeline_audit.py`.
- `LoraHeadTrainer.load()` fixed: out_dim now derives from `classes.npy`
  (was hardcoded to 2). Adult LoRA negative (0.812 vs 0.831 head-only) stays
  documented in notes.

## 5. Model/encoder correctness (MEDIUM — verified)

- Sanitization is per-element (not whole-row) on encode output; all stored
  caches scanned — 0 non-finite rows anywhere. No benchmark reads raw data.
- Pooling extracted to `recgen.encoder.pool_hidden`/`sanitize`; hand-computed
  unit tests for masked mean and last-non-pad.
- Cache staleness: sha256 of joined texts; tests cover same-length-different
  and reordered texts (`tests/test_encoder_audit.py`).

## 6. API/serving correctness (MEDIUM — two bugs fixed)

- **Fixed:** `rank_catalog` returned ascending (worst-first) rankings
  (`np.argsort(-scores)[::-1]`). Now descending. Regression-tested.
- **Fixed:** `rank_all` cache key was a fixed "catalog_default" with no
  model/pooling — stale embeddings across backend switches. Keys are now
  `{model}_{pooling}_{key}`. Verified via smoke test + unit tests.
- `RECGEN_DEVICE` now wired into the encoder constructor (CPU Docker path).
- Dockerfile: BUILD VERIFIED (image 6.1GB, `/health` OK in container on the
  CPU path; model files must be mounted/baked in for real requests).
- Lazy model load confirmed (`/health` dim=null before first encode).

## 7. Repo hygiene / IP (MEDIUM — done)

- `docs/product.md`: `git log --all` empty (purged); still gitignored.
- No secrets/tokens in `git ls-files`.
- **Fixed:** `.github/` removed from `.gitignore` (it silently blocked the CI
  workflow from ever being committed).
- TODO (needs user): `gh auth refresh -s workflow --hostname github.com`
  (token lacks `workflow` scope), then commit `.github/workflows/ci.yml`
  and push.

## 8. Final deliverables — DONE

1. `scripts/audit_report.md` — per-section status, evidence, fixes, final
   tables with diffs vs previous published numbers.
2. Leak-free headline numbers in README/blog/notes with "leakage-free by
   construction (see AUDIT.md)" notes.
3. Unit tests added: leak filter, pooling, cache staleness, bootstrap CI,
   API ordering/cache key, pipeline polars+pandas. 25 passed.
4. Push pending user's `gh auth refresh -s workflow`.

## Final numbers (leak-free, 2026-08-09)

Standard protocol: recgen **HR@10 0.4214 / NDCG@10 0.2375**; popularity
0.3326/0.2033; ALS-128 0.3037/0.1742; SASRec 0.1282/0.0582. Cold-start:
recgen 0.4174/0.2422. Full-catalog rec@10 0.0218; 2-stage rec@10 0.0289.
Claim wording: "~39% vs ALS, ~3.3x vs SASRec" (was 41%/3.5x). Serving:
30.2ms -> 33.1 req/s, ~$0.63/1M. Multi-label go_emotions: micro-F1 0.400.

## Commands

```bash
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
uv run python experiments/ecommerce/sota_benchmark.py      # ~1h (encode + SASRec)
uv run python experiments/ecommerce/strong_baselines.py    # full-catalog table
uv run pytest tests/ -q
uv run python scripts/benchmark.py --model smol360
uv run python scripts/summarize.py                          # results.csv
```
