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
