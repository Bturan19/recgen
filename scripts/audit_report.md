# recgen audit report — 2026-08-09

Audit session per `AUDIT.md`. Commands run with
`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` on the M1 Pro (32GB, MPS).

## 1. Label leakage — VERIFIED FIXED, filter extended

- Item-id leak (repeat purchase): 1,134/25,000 users (4.54%) — excluded.
- **New finding:** a *different* item_id sharing the test item's **title**
  also leaks the answer into the verbalized history text. Measured:
  **1/23,866 users**, and that user lands in the TRAIN split (index 4624), so
  it cannot affect test metrics. `leakfree.leak_mask` now also excludes
  title-level collisions (23866 -> 23865 users).
- All row-level benchmarks audited for the same bug class:
  - IMDB: independent reviews, row-level split. Exact-duplicate review text
    across train/test: 9/3,000 test rows, ALL label-consistent (benign for
    both recgen and TF-IDF equally).
  - Adult / house prices / cardiotocography / SMS spam: row-level, no
    grouping, no group-by leaks. OK.
- Negative candidates: positive + history items excluded, fixed seed per
  user index (same for all models). Verified by test.
- Tests: `tests/test_leakfree.py` (item-id, title-in-text, negative
  candidates, split disjointness, count pin at 23,865).

## 2. Evaluation correctness — OK

- `CatalogRankingHead.evaluate`, `sota_benchmark.hr_ndcg` and
  `strong_baselines.full_ranks` all rank by `np.argsort(-scores)` and locate
  the target — no tie-handling bias (identical input -> identical order);
  no already-interacted masking that differs between models.
- 2-stage candidates: identical candidate set for every ranker
  (popularity ∪ embedding-kNN, 300) — same loop, same `cands` list.
- Bootstrap CI: division guard `where=ranks>0`; tested to never emit NaN
  even with rank-0 rows (`tests/test_eval_audit.py`).
- SASRec eval uses scalar indexing (`out[0, len-1]`), HR@10=0.1198 in the
  expected band; val HR peaked 0.1345 (not 1.0).
- `best_mrr` early stopping: head internal 90/10 val on the same ordered
  `H[:n_tr]` in both sota_benchmark and strong_baselines — consistent.

## 3. Baselines fairness — OK (state honestly)

- ALS (implicit, 128 factors, 30 iters), ItemKNN (k=100), EASE (λ=200),
  popularity, last-item, SASRec (128-dim, 2 layers, 3 negatives, 20 epochs,
  val early-stop) — none grid-searched. ALS trained on ALL users' pre-test
  history (fair factors), matrix excludes leaked users.
- Claim wording verified in blog/README: "beats our tuned-class baselines
  under the standard leave-one-out + 100-negative protocol", NOT "SOTA".

## 4. Framework completeness — MultiLabelHead now benchmarked

- `experiments/multilabel.py` (new): go_emotions, 6,000 rows, 28 labels,
  1,008 multi-label rows. Frozen SmolLM2-360M + MultiLabelHead:

  | model | micro-F1 |
  |---|---|
  | MultiLabelHead | **0.4004** |
  | TF-IDF + LogReg (binary relevance) | 0.3756 |
  | LightGBM on h (binary relevance) | 0.2581 |

  This backs the "multi-label" claim on real data.
- `RecgenPipeline` polars+pandas: both frame types tested with cached
  encode; transform/predict hit the cache (no re-encode per call)
  (`tests/test_pipeline_audit.py`).
- `LoraHeadTrainer.load()` bug fixed: `out_dim` was hardcoded to 2 for
  classifiers; now reads `classes.npy` first (multi-class safe).
  Adult LoRA negative result (0.812 vs 0.831 head-only @ 4k rows) stays
  documented in notes.

## 5. Encoder correctness — OK

- Non-finite sanitization: zeroing is per-element on the *sanitized output*
  at encode time; all stored caches scanned — **0 non-finite rows in every
  cache** (incl. SMS 5,574-row cache; the 67 bad rows were fixed at encode).
  No benchmark reads unsanitized data.
- Pooling extracted to `recgen.encoder.pool_hidden` + `sanitize`, unit
  tested with hand-computed examples (masked mean, last-non-pad).
- Cache staleness: hash = sha256 of joined texts; tested: same-length
  different texts and reordered texts both invalidate (`test_encoder_audit`).

## 6. API/serving — two bugs fixed

- **BUG (fixed):** `rank_catalog` returned rankings in ASCENDING score
  order (`np.argsort(-scores)[::-1]`) — the "top_k" were the least similar
  items. Now `np.argsort(-scores)`. Regression tests added.
- **BUG (fixed):** `rank_all` catalog cache key was a fixed
  `"catalog_default"` with no model/pooling — switching backends returned
  stale embeddings. Keys now `{model}_{pooling}_{key}`. Verified via API
  smoke test (cache hit on second identical call).
- `RECGEN_DEVICE` env var was parsed but never passed to the encoder; now
  wired through (matters for the CPU Docker image).
- Lazy model load verified (`/health` dim=null before first request).
- Docker build: see item below.

## 7. Repo hygiene / IP — OK

- `docs/product.md`: `git log --all -- docs/product.md` empty (purged);
  `docs/product.md` stays in `.gitignore`.
- `git ls-files` scan: no tokens/keys committed; `.env`/models/data/cache
  not tracked.
- **Fixed:** `.github/` was in `.gitignore`, which is why the CI workflow
  was never pushed. Removed from `.gitignore`.
- TODO (needs user action): `gh auth refresh -s workflow --hostname
  github.com` (token lacks `workflow` scope), then commit
  `.github/workflows/ci.yml` and push.
- Remote: https://github.com/Bturan19/recgen (public).

## 8. Final numbers (leak-free, title-aware; 2026-08-09 re-run)

Setup: 23,865 clean users (1,134 item-id leaks + 1 title collision removed),
13,136 items, 2,387 test users, SmolLM2-360M frozen, mean pooling, cached.

Standard protocol (leave-one-out + 100 random negatives):

| model | HR@10 | NDCG@10 |
|---|---|---|
| **recgen (frozen 360M + catalog head)** | **0.4214** | **0.2375** |
| popularity | 0.3326 | 0.2033 |
| ALS-128 (fair protocol) | 0.3037 | 0.1742 |
| SASRec (vanilla, ours) | 0.1282 | 0.0582 |

Cold-start (history <= 7, 1284 users): recgen 0.4174/0.2422, ALS-128
0.3224/0.1894, popularity 0.3528/0.2189, SASRec 0.1332/0.0613.

Full-catalog (13,136 items): recgen rec@10=0.0218 rec@20=0.0352
mrr@20=0.0114 [0.0085,0.0144]; ALS-128 0.0189/0.0323/0.0083; popularity
0.0239/0.0310/0.0112. 2-stage (300 candidates, same for all):
recgen 0.0289/0.0411/0.0138; ALS-128 0.0168/0.0256/0.0077.

Diff vs previously published (item-id filter, 23,866 users / 13,137 items):

| model | published HR@10 | new HR@10 | delta |
|---|---|---|---|
| recgen | 0.4179 | 0.4214 | +0.84% rel |
| popularity | 0.3266 | 0.3326 | +1.8% rel |
| ALS-128 | 0.2956 | 0.3037 | +2.7% rel |
| SASRec | 0.1198 | 0.1282 | +7.0% rel |

All within rounding of the published 3-decimal numbers (0.418/0.333/0.304/
0.128). **Claim changes:** "41% vs ALS" -> ~39%; "3.5x vs SASRec" -> ~3.3x;
"cold-start advantage (0.424 > 0.418 overall)" no longer holds on the final
run (0.417 vs 0.421 overall) — docs updated to "holds up on cold-start".

Serving (re-audited, idle machine): 30.2 ms/request -> 33.1 req/s,
projected $0.63 / 1M requests (README/blog updated).

Multi-label (new): go_emotions, 28 labels — recgen 0.4004 micro-F1 vs
TF-IDF+LogReg 0.3756, LightGBM-on-h 0.2581.

## Commands used

```
uv run pytest tests/ -q                      # 25 passed
uv run python experiments/ecommerce/sota_benchmark.py    # re-run (cache hits)
uv run python experiments/ecommerce/strong_baselines.py  # re-run (title-aware, re-encode)
uv run python experiments/multilabel.py                  # go_emotions multi-label
uv run uvicorn api.app:app --port 8199       # API smoke test
docker build -f api/Dockerfile .             # build check
```
