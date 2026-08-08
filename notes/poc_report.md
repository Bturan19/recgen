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
