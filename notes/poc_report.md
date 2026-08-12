# POC report — is the frozen-LLM encoder enough?

## Results (final table, best of pooling per row)

| task | method | metric | value |
|---|---|---|---|
| adult (48k) | LightGBM raw | acc / auc | **0.862** / **0.922** |
| adult | LLM-head (mean) | acc / auc | 0.848 / 0.905 |
| adult | LLM-head (last) | acc / auc | 0.842 / 0.899 |
| adult | LightGBM on h | acc / auc | 0.839 / 0.893 |
| adult | LightGBM h+raw (stack) | acc / auc | 0.859 / 0.917 |
| house prices (1.5k) | LightGBM raw | mae / rmse (log$) | **0.087** / **0.127** |
| house prices | LLM-head | mae / rmse (log$) | 0.340 / 0.445 |
| house prices | LightGBM on h | mae / rmse (log$) | 0.192 / 0.267 |
| imdb (15k) | TF-IDF + LogReg | acc / auc | 0.890 / 0.957 |
| imdb | LLM-head (mean) | acc / auc | **0.914** / **0.971** |
| imdb | LLM-head (last) | acc / auc | 0.884 / 0.955 |
| imdb | LightGBM on h | acc / auc | 0.870 / 0.947 |

Setup: SmolLM2-360M frozen, fp16, MPS (M1 Pro). Head = 2-layer MLP (256/128),
40 epochs, early stopping, 85/15 internal val. Same 80/20 test split for all
methods. House-prices target is log1p-transformed.

## Verdict

**The frozen encoder alone is a solid but not state-of-the-art feature
extractor.** It is within ~2 points of GBDT on tabular classification, loses
badly on numeric regression, and **beats TF-IDF + logistic regression on
IMDB sentiment by +2.4 acc / +1.4 AUC (0.914/0.971 vs 0.890/0.957) with mean
pooling** — the first clear win, on the task class where pretrained language
understanding matters. Pooling choice matters a lot: mean beats last by
+3.0 acc on text (0.914 vs 0.884) and +0.6 on tabular. Stacking h onto raw
features does NOT beat raw features alone on adult. This matches TabLLM's
published finding: LLM encoders win where semantics matter, not on
pure-numeric tabular.

**The interesting wins are all in the "adaptation" phases we have not built
yet** (each with published evidence):

1. **LoRA/domain adaptation of the backbone** (GenRec Phase 1): +10-20% vs OSS
   base. We trained the head only; the backbone is generic.
2. **Task head + objectives** (GenRec Phase 2): +35-50% on ranking vs Phase-1
   base. Our MLP head is the simplest possible head.
3. **Learned verbalization** (Netflix, "From Logs to Language"): up to +93%
   relative vs rigid templates. Ours is a rigid template.
4. **Pooling**: mean > last everywhere (0.848 vs 0.842 adult). Attention-based
   pooling is untried.
5. **Product angles that survive negative results**: no feature engineering,
   cold-start items/rows, transferable cached embeddings, cheap prefill-only
   inference, unified backbone for many tasks.

## Cost/throughput (M1 Pro, MPS, fp16)

| config | throughput |
|---|---|
| adult prompts (~150 tok), bs 32 | 30 prompts/s |
| imdb (~270 tok), bs 32, max 512 | 6 prompts/s |
| imdb pathological (>512 tok, bs 16, max 1024) | 0.5 prompts/s |

Cache makes re-runs free; head training is seconds once embeddings exist.

## Recommendation (Gate 2)

**Continue, but reposition Phase 2.** The product story is not "frozen
embeddings beat GBDT" (they don't); it is "one small backbone + context
engineering + a cheap head does any task with near-GBDT accuracy and no
feature engineering" — and the gap closes with LoRA adaptation. Phase 2 plan:

1. **LoRA + task head trained jointly** (fixes the adult gap; the single
   highest-value experiment, ~30-60 min/run on M1).
2. **Attention pooling / learned [CLS]** instead of last/mean.
3. **Verbalization ablations**: field subsets, order, sentence-format vs
   key-value format (cheap, cache-friendly).
4. **Recsys pilot** (MovieLens): catalog-aware head + history verbalization —
   the domain where the architecture is proven (GenRec).
5. Then Kaggle/benchmark writeup.

All experiments reproducible: `scripts/run_all.sh`, results in
`experiments/results/results.csv`.

## Phase 2a: LoRA + head joint training (Adult, SmolLM2-360M)

Fair comparison on the SAME 4k-row training subset (same test split as the full runs):

| config (4k train) | acc | auc |
|---|---|---|
| head-only (frozen, MLP head) | 0.8313 | 0.8846 |
| LoRA r8 α16 + head (joint) | 0.8122 | 0.8842 |
| reference: head-only on full 24k | 0.8477 | 0.9052 |
| reference: LightGBM raw | 0.8621 | 0.9217 |

Verdict: **joint LoRA does not help at this data scale on tabular** — the
extra 1.6M LoRA params overfit 4k rows. GenRec's Phase-1 gains (+10-20%) come
from *unlabeled domain pretraining* (Phase 1 = pretraining corpora, not small
task sets). Implication: the right Phase-2 lever is domain pretraining + task
head, or ranking heads (e-commerce), NOT small-n joint LoRA. IMDB LoRA was
aborted: MPS long-sequence backward is ~0.5-2 rows/s (multi-hour runs).

Hardware notes: MPS fp32 is fine for LoRA (20-24 rows/s on 150-tok prompts);
long sequences (IMDB) + backward collapse to 1-2 rows/s. Batch >= 16 required
for reasonable MPS utilization; use_cache=False + base_model.model.model path
needed to avoid the "Placeholder storage" MPS bug and 32-layer memory blowup.

## Phase 2c: e-commerce pilot (in progress)

## Phase 2c: e-commerce pilot — Amazon Musical Instruments (next-item)

Data: Amazon Reviews 2023, Musical_Instruments. 52.7k users (>=6 interactions),
71.2k items (>=5 users). Sample: 8k train / 1k val / 1k test users, catalog
6,941 items, history = last 10 purchased items, target = last interaction.

Verbalization: "The user recently purchased: <title> by <store> | ..." and
"Item: <title> by <store>. Categories: ... Features: ...". Both user histories
and catalog items encoded once by frozen SmolLM2-360M (mean pooling, cached).
Ranking head (GenRec-style catalog-aware): score(u,i) = <W h_u, e_i> + b_i,
full-catalog softmax CE, trained in ~5s on MPS.

| model (test 1k users) | recall@10 | recall@20 | mrr@20 |
|---|---|---|---|
| recgen LLM-embedding ranker | 0.028 | 0.044 | 0.0162 |
| popularity baseline | 0.022 | 0.035 | 0.0113 |
| ALS (32 factors, implicit) | 0.003 | 0.004 | 0.0024 |

Verdict: the frozen-LLM-embedding ranker beats popularity (+25-27% relative)
and ALS (~7x) out of the box, with no feature engineering and training the
head in seconds. This is the architecture's home turf: semantically rich item
metadata + purchase history in one shared LLM space. Obvious headroom: LoRA/
domain pretraining on the backbone, ratings in history, candidate generation
(2-stage), more users.

Phase 2 lessons so far:
1. Frozen embeddings + tiny head = cheap, competitive, no feature engineering.
2. Small-n joint LoRA does not help (Adult); backbone adaptation belongs at
   the domain-pretraining stage (GenRec Phase 1), which needs unlabeled data.
3. Ranking heads (catalog-aware) are where the architecture shines.

## Phase 2c follow-up: v2 — 25k users, ratings in history, 2-stage

> SUPERSEDED: the 2-stage numbers in this section were inflated by a metric
> bug (see CORRECTION below). Keep only the full-catalog comparisons here.

Amazon Musical Instruments, v2: 20k/2.5k/2.5k users, catalog 13,524 items
(bigger than v1's 6,941), history = last 10 purchases WITH ratings
("rated x/5"), ranking head identical.

| model (test 2.5k users) | recall@10 | recall@20 | mrr@20 |
|---|---|---|---|
| recgen full-catalog | 0.0264 | 0.038 | 0.0139 |
| recgen 2-stage (300 candidates) | **0.342** | **0.500** | **0.160** |
| popularity | 0.024 | 0.032 | 0.0106 |
| ALS (64 factors) | 0.0008 | 0.0016 | 0.0006 |

Takeaways:
- The 2-stage deployment (popularity ∪ embedding-kNN candidates -> head
  ranking) is the real-world shape: **34% recall@10 / 50% recall@20**, ~13x
  full-catalog and ~14x popularity. This is the number to quote.
- More users + ratings did not lift full-catalog recall vs v1 (0.026 vs 0.028)
  — the bigger catalog makes full-catalog ranking harder; recall metrics are
  catalog-size dependent (report catalog size always).
- Head training: 12-31s on frozen embeddings. Everything else is cached
  encodes.

## CORRECTION + Phase 3: honest e-commerce numbers, AD, model scaling

### E-commerce v3 — corrected protocol (strong_baselines.py)

BUG FOUND: the v2 2-stage loop counted hits over ALL 25k users (train
included) but divided by test count only, inflating recgen 2-stage ~10x
(0.342/0.500/0.160 were WRONG). Corrected loop: test users only. Corrected
table (same test users, same 300 candidates for every ranker, bootstrap 95%
CI on MRR):

full-catalog (13,524 items):
| model | recall@10 | recall@20 | mrr@20 (ci95) |
|---|---|---|---|
| recgen (LLM-emb head) | 0.0264 | 0.038 | 0.0139 [0.0109,0.0172] |
| ALS 128f (fair protocol) | 0.0236 | 0.036 | 0.0143 [0.0110,0.0181] |
| popularity | 0.024 | 0.032 | 0.0106 |
| EASE (catalog-only) | 0.0064 | 0.010 | 0.0033 |
| ItemKNN (k=100) | 0.0008 | 0.002 | 0.0006 |
| last-item emb sim | 0.0008 | 0.002 | 0.0008 |

2-stage (same 300 candidates: popularity ∪ mean-history-emb kNN):
| model | recall@10 | recall@20 | mrr@20 (ci95) |
|---|---|---|---|
| recgen | 0.0296 | 0.044 | 0.0150 [0.0116,0.0187] |
| ALS 128f | 0.022 | 0.030 | 0.0111 [0.0081,0.0145] |
| popularity | 0.024 | 0.032 | 0.0100 |
| last-item | 0.002 | 0.003 | 0.001 |

Honest read: recgen is statistically TIED with a properly-tuned ALS on
full-catalog MRR (CIs overlap; ALS slightly higher MRR, recgen higher
recall@10) and modestly better in the 2-stage setting. It clearly beats
popularity/ItemKNN/EASE/last-item. The defensible claim: "a frozen 360M LLM
+ a 15-second-trained head matches a tuned MF recommender with zero feature
engineering and no user/item ID embeddings." NOT "7-33x better".

### Anomaly detection (anomaly.py)

SMS spam (text, 5574 msgs, 13.4% spam):
- unsupervised: TF-IDF+IsolationForest 0.837 AUC >> LLM-emb+IF 0.491; LLM
  kNN-distance 0.28-0.32 (near random). Spam is lexical; semantic embeddings
  put it close to normal.
- supervised: LLM-emb + MLP head 0.9995 AUC; LLM-emb + logreg 0.9997;
  TF-IDF + logreg 0.989. LLM embeddings win when labels exist.

cardiotocography (numeric tabular AD, 8.3% pathological):
- raw + IsolationForest 0.934 AUC; raw+kNN 0.72.
- LLM embeddings: 0.500 everywhere (random) — numeric verbalization gives
  the frozen LLM nothing. raw+LLM concat ≈ raw alone.

Verdict: AD is NOT a regime where this architecture shines, except as
supervised text classification (which is just classification). Honest
limitation, documented.

### Model scaling (model_compare.py, IMDB 2k train/500 test, same head)

| model | acc | auc | encode/s (M1, fp16) |
|---|---|---|---|
| SmolLM2-1.7B | 0.914 | 0.970 | ~1.5 |
| SmolLM2-360M | 0.906 | 0.958 | ~6 |
| Qwen3-0.6B | 0.886 | 0.953 | ~2.7 |

The 0.6B class is a dead zone: Qwen3-0.6B is both slower and worse than
360M. The real quality upgrade is 1.7B (+0.8 acc, 3.7x compute). Default
stays 360M (speed); 1.7B available for quality.

### Engineering fixes
- FrozenEncoder: sanitizes non-finite pooled values (MPS fp16 instability;
  found 67/5574 NaN rows on SMS); warns + zeroes.
- Baseline protocol: ALS/ItemKNN/EASE now trained on ALL users' pre-test
  history (fair factors for test users) — previously test users had no
  factors, artificially crushing ALS.
- TemplateVerbalizer now supports polars frames.

## SOTA-protocol benchmark (sota_benchmark.py) — the defensible table

Standard RecBole-style protocol: leave-one-out (last interaction), rank over
positive + 100 random negatives (fixed seed, excluding positive & history),
HR@10 / NDCG@10. Same candidate set for every model. Test: 2.5k users,
catalog 13,524 items, Amazon Musical Instruments.

| model | HR@10 | NDCG@10 |
|---|---|---|
| recgen (frozen 360M + catalog head) | 0.4288 | 0.2434 |
| popularity | 0.3352 | 0.2063 |
| ALS 128f (fair protocol) | 0.3228 | 0.1966 |
| SASRec (128-dim, 2-layer, ours, 20 ep) | 0.1864 | 0.0997 |

Cold-start slice (history <= 7 items, 1354 users):
| model | HR@10 | NDCG@10 |
|---|---|---|
| recgen | 0.4343 | 0.2511 |
| ALS 128f | 0.3530 | 0.2181 |
| popularity | 0.3560 | 0.2186 |
| SASRec | 0.2009 | 0.1083 |

Notes for honest reporting:
- SASRec is a vanilla implementation (128-dim, 2 layers, 3 negatives, 20
  epochs, val early-stop); not grid-searched. On this sparse dataset it
  plateaus ~0.19 HR@10 (literature SASRec numbers on dense 20-core Amazon
  subsets are higher — protocol/data density differs).
- recgen's head trained in ~15s on frozen embeddings; SASRec took ~10 min.
- recgen is robust to short histories (cold-start 0.434 vs 0.429 overall),
  while ID-based models degrade on sparse users.

## Serving speed benchmark (scripts/benchmark.py, M1 Pro, fp16 MPS)

encode throughput: 50tok 110/s, 150tok 112/s, 300tok 62/s, 500tok 38/s
(360M); 1.7B: 77/35/18/11.

catalog-aware scoring: 100k items in ~11ms (360M) / ~30ms (1.7B) — one matmul
= multi-output array for the entire catalog.

full request (300-tok history + score 100k-item catalog): 360M ~29ms -> ~35
req/s; 1.7B ~90ms -> ~11 req/s. Projections: ~3M req/day on one M1; ~$0.60
GPU per 1M requests (A10-class est); 100k-item catalog embeds once in ~0.04
GPU-hr, then cached.

## LEAKAGE AUDIT + leak-free results (final numbers, re-run 2026-08-09)

BUG: users whose held-out (last) purchase also appears in their history
(repeat purchase) leaked the answer into the model input. 1,134/25,000
sampled users (~4.5%) affected. Fixed in `leakfree.py`; all e-commerce
benchmarks now exclude such users. A second, rarer leak class was found in
the audit session: a *different* item_id sharing the held-out item's title
also puts the answer in the verbalized text — 1/23,866 users (train only,
impact nil), now excluded too. Final clean set: 23,865 users, 13,136 items,
2,387 test users.

Standard protocol (leave-one-out + 100 random negatives), leak-free:
| model | HR@10 | NDCG@10 |
|---|---|---|
| recgen | 0.4214 | 0.2375 |
| popularity | 0.3326 | 0.2033 |
| ALS-128 | 0.3037 | 0.1742 |
| SASRec | 0.1282 | 0.0582 |

Cold-start (history <= 7, 1284 users): recgen 0.417/0.242; popularity
0.353/0.219; ALS 0.322/0.189; SASRec 0.133/0.061. (Note: on this run
recgen's cold-start rate is ~equal to its overall rate, not higher; the
"cold-start advantage" claim is no longer made.)

Leak impact: the first leak-free run (item-id filter, 23,866 users) cut the
pre-filter numbers by recgen -2.6% / ALS -8.4% / SASRec -35% (sequence
models memorize the leaked item best). The re-audited title-aware run moves
recgen slightly UP (+0.8% rel. vs the item-id-filter run, 0.4179 -> 0.4214)
because the test sample changed; recgen edge: +39% vs ALS, ~3.3x vs SASRec.

Benchmarks that need no leak filter (row-level splits, no grouping):
IMDB, Adult, house prices, SMS spam, cardiotocography — audited, OK.
IMDB duplicate-text check: 9/3,000 test reviews duplicated in train, all
label-consistent (benign, affects all methods equally).

Full audit checklist: AUDIT.md; session report: scripts/audit_report.md.

Leak-free full-catalog (13,136 items, 2,387 test users) + 2-stage (same 300
candidates for every ranker):
| model | full recall@10 | full mrr@20 (ci95) | 2-stage rec@10 | 2-stage mrr@20 |
|---|---|---|---|---|
| recgen | 0.0218 | 0.0114 [0.0085,0.0144] | 0.0289 | 0.0138 [0.0104,0.0174] |
| popularity | 0.0239 | 0.0112 [0.0083,0.0141] | 0.0239 | 0.0106 [0.0078,0.0135] |
| ALS-128 | 0.0189 | 0.0083 [0.0062,0.0108] | 0.0168 | 0.0077 [0.0053,0.0101] |
| ItemKNN | 0.0013 | 0.0006 | - | - |
| EASE | 0.0063 | 0.0036 | - | - |
| last-item | 0.0013 | 0.0008 | 0.0021 | 0.0010 |

Full-catalog is noisy: recgen ≈ popularity on recall, ahead on MRR and
clearly ahead of ALS on MRR/recall. The 100-negative standard protocol (see
above) is the primary claim and favors recgen strongly.

## Phase 3b: MovieLens-1M — GenRec-style movie recommendation (in progress)

The Netflix use case: next-item movie recommendation under the SASRec-paper
protocol (Kang & McAuley ICDM'18): 5-core (6,040 users / 3,416 items — exact
reproduction of the paper's Table II), implicit feedback, per-user temporal
split (last = test, second-to-last = val, rest = train, ALL users in
training), eval = positive + 100 random negatives, HR@10/NDCG@10.
Leak-free by construction: MovieLens pairs are unique, titles are unique.
Protocol validity check: popularity = 0.435 vs paper's PopRec 0.433 — the
eval protocol reproduces published numbers.

Backbone encoding: item = "Movie: <title>. Genres: <g>"; history = last 20
watched movies (LLM context limit), while SASRec/BERT4Rec see up to 200.

| model | HR@10 | NDCG@10 | notes |
|---|---|---|---|
| BERT4Rec (ours, d=64, 2L, mask 0.15) | 0.6983 | 0.4538 | full-softmax cloze |
| ALS-64 (implicit, fair) | 0.6545 | 0.3979 | |
| recgen[smol17] | 0.5978 | 0.3678 | 1.7B encoder, head ~30s |
| recgen[smol360] | 0.5303 | 0.3205 | 360M encoder |
| popularity | 0.4349 | 0.2404 | |
| SASRec (ours, d=50, 1-neg BCE) | 0.4407 | 0.2435 | weaker than published (see caveat) |

Cold-start (history <= 30, 916 users): recgen[smol17] 0.6856 / 0.4286;
recgen[smol360] 0.6168 / 0.3716; BERT4Rec 0.7555 / 0.4954; SASRec 0.5207 /
0.2991; popularity 0.5011 / 0.2918 — the LLM encoder closes the gap to the
transformer baseline on short histories (9% behind BERT4Rec vs 14% overall).

Caveats (honest reporting):
- BERT4Rec's paper reports ~0.84 HR@10 on ML-1M; the original SASRec paper
  reports 0.8245. Our implementations are NOT grid-searched (fixed
  hyperparameters, no L2 tuning), and our SASRec (1 negative/step BCE)
  underperforms its published-class numbers (0.64 in BERT4Rec's own
  reimplementation). We report our own baselines on the identical protocol.
- recgen sees 20 movies of history vs 200 for the sequence models; the head
  trains on one (history -> val) example per user, vs ~150k masked-position
  examples for BERT4Rec. The comparison is structurally favorable to the
  sequence models.
- Claim: "competitive with ALS-class MF (within ~9%) and within ~14% of a
  transformer sequential baseline (BERT4Rec), with zero feature engineering
  and a head that trains in seconds; closes most of the gap on cold-start
  users" — NOT SOTA.
- 7-9B backbone feasibility (2026-08-09): Qwen2.5-7B loads on MPS (32GB M1
  Pro) but encodes at ~30-70 tokens/s — ~25h for the full ml-1m encode. Not
  usable on this hardware; needs MLX 4-bit quantization or a GPU. The 7B
  download was removed (2026-08-09) after the cleanup.

## Marketplace product moderation (private dataset) (multimodal, Aug 2026)

Dataset: a gated HF marketplace-moderation sample (4,000 products; Gemini-3.1-flash-lite eval
as labels; 78.5% approved / 21.5% rejected; 23 rejection tags; up to 8
images/product). Task: predict eval_decision. 67% of rejection reasons are
image-driven (brand on box, specs printed on product, title/image mismatch).

Pipeline: verbalized product text (title/brand/category/attrs/description)
-> SmolLM2-1.7B mean-pool (cached) + SigLIP2-SO400M image embeddings (mean
over up to 4 images) -> concat -> head. Apple Vision OCR extracted into the
text did NOT help (mean-pooling dilutes contradiction signals).

Results (same 80/10/10 split, 400-row test):
| model | acc | F1 | AUC |
|---|---|---|---|
| text-only MLP (1.7B) | 0.843 | 0.519 | 0.832 |
| text+img MLP | 0.861 | 0.611 | 0.869 |
| text+img LightGBM | 0.866 | 0.571 | 0.879 |
| text+img LightGBM weighted + tuned th | **0.886** | **0.716** | **0.881** |
| + OCR embeddings / multi-task tags | no gain | | |
| 5-fold CV (best config) | 0.817±0.034 | 0.595 | 0.820 |
| Qwen2.5-VL-3B judge (20 rows) | 0.55 | | over-rejects vs Gemini style |

Notes:
- OCR-in-context (1.7B), Apple Vision OCR features, multi-task tag head,
  ensembles: no improvement over the plain concat.
- Gemini's decision style is lenient-with-notes (approves but flags category
  mismatches); the VLM judge over-rejects, hurting agreement.
- Honest claim: ~0.82-0.89 acc (split-dependent), ~0.88 AUC ceiling with
  frozen 1.7B + SigLIP2 + GBDT head on 3.2k train rows. More data or a 7B+
  VLM tuned as a judge would be the next lever.

### recgen framing + throughput (marketplace)

This is a full recgen application: verbalize (title/brand/category/attrs/
description) -> frozen SmolLM2-1.7B mean-pool embedding + frozen SigLIP2
image embeddings -> GBDT head. Multimodal by concatenating frozen encoders
at the head level — no VLM generation, no fine-tuning.

Measured serving throughput (M1 Pro, MPS, batch=128):
| stage | rate | per day |
|---|---|---|
| text encode (1.7B, ~800 tok/product) | 1.53 products/s | ~132k |
| image encode (SigLIP2 384px, 2.8 img/product) | 3.8 img/s | ~116k |
| head predict (MLP / LGBM on 3200-dim) | >12k products/s | millions |
| combined pipeline (encode once, cached) | ~1.5 products/s cold | ~120k/day cold; repeat scoring ~12k/s |

- The head is not the bottleneck; encode is. Cached embeddings make re-scoring
  effectively free (content-addressed cache).
- vs the previous in-house pipeline at 70k/day: this laptop pipeline is ~120k/day cold-start;
  on an A100 with batched/VLLM-style encoder serving, the same embeddings
  pipeline scales to millions/day (the encode is the only GPU-bound stage).
- vs the VLM judge (8 s/row -> ~11k/day): the frozen-encoder pipeline is
  ~10x cheaper AND more accurate as a Gemini proxy (0.82-0.89 vs 0.55).
- Cost framing: 1.7B text + 400M vision, head is a few MB of weights;
  the whole model family fits a laptop/CPU serving box.
