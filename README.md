# recgen

A generic "LLM-as-encoder" framework: verbalize any instance as text, encode it
with a small frozen/adapted LLM, and train a lightweight head for
classification, regression, or ranking. Inspired by Netflix's GenRec
(prefill-only LLM recommendation via a pooled hidden state + task head).

## Idea

```
raw instance (row / history / text)
   -> verbalizer (template config)          # "context engineering"
   -> small LLM (frozen or LoRA-adapted)     # the "preference encoder"
   -> pooled hidden state h (last / mean)
   -> lightweight head                       # ClassifierHead / RegressionHead
```

The LLM is never decoded at inference: one forward pass, one pooling op, one
MLP. Cost scales with context length, not generation.

## Setup

```bash
uv sync                      # install deps (torch, transformers, lightgbm, ...)
uv run hf download HuggingFaceTB/SmolLM2-360M --local-dir models/SmolLM2-360M
```

macOS note: set `OMP_NUM_THREADS=1` (and `KMP_DUPLICATE_LIB_OK=TRUE`) before
running anything that mixes pyarrow/LightGBM/torch — libomp worker threads can
segfault model loading otherwise.

## Usage

```python
from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer

verbalizer = TemplateVerbalizer(
    fields=["age", "workclass", "occupation", "hours-per-week"],
    instruction="Predict whether annual income exceeds $50K.",
)
encoder = FrozenEncoder("models/SmolLM2-360M", pooling="mean", batch_size=32)
pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=".cache/my_task")
pipe.fit(X, y)          # X: DataFrame or list[str]
pipe.predict(X_test)
pipe.transform(X_test)  # -> (n, 960) embeddings, for stacking/GBDT pipelines
```

Embeddings are cached (sha256-keyed) so re-runs are instant; the head trains in
seconds once the cache is warm.

## Experiments (POC)

| task | model | metric | value |
|---|---|---|---|
| adult | LightGBM raw | acc / auc | 0.862 / 0.922 |
| adult | LLM-head (last) | acc / auc | 0.842 / 0.899 |
| adult | LLM-head (mean) | acc / auc | 0.848 / 0.905 |
| adult | LightGBM on h | acc / auc | 0.832 / 0.887 |
| adult | LightGBM h+raw | acc / auc | 0.859 / 0.917 |
| house prices | LightGBM raw | mae / rmse (log) | 0.087 / 0.126 |
| house prices | LLM-head | mae / rmse (log) | 0.340 / 0.445 |
| imdb (15k) | TF-IDF + LogReg | acc / auc | 0.890 / 0.957 |
| imdb | LLM-head (mean) | acc / auc | **0.914** / **0.971** |
| imdb | LLM-head (last) | acc / auc | 0.884 / 0.955 |
| imdb | LightGBM on h (mean) | acc / auc | 0.895 / 0.965 |
| adult (4k) | head-only MLP | acc / auc | 0.831 / 0.885 |
| adult (4k) | LoRA r8 + head (joint) | acc / auc | 0.812 / 0.884 |
| ecom next-item | recgen ranker | rec@10 / mrr@20 | **0.028** / **0.016** |
| ecom next-item | popularity | rec@10 / mrr@20 | 0.022 / 0.011 |
| ecom next-item | ALS (32f) | rec@10 / mrr@20 | 0.003 / 0.002 |

E-commerce pilot (Amazon Musical Instruments, 6.9k-item catalog, next-item
prediction): histories + item metadata verbalized and encoded by the frozen
360M model; a GenRec-style catalog-aware head `score = <W h_u, e_i> + b_i`
trained in ~5s beats popularity and ALS with zero feature engineering.

Preliminary read: the frozen LLM encoder is a few points behind GBDT on
pure-numeric tabular (expected — see TabLLM) but **beats TF-IDF + logistic
regression on text classification with mean pooling (0.914 vs 0.890 acc)**.
The interesting regime: text-heavy data, cold-start, semantically meaningful
features — plus the workflow wins (no feature engineering, cacheable
embeddings, prefill-only inference).

```bash
scripts/run_all.sh          # runs all three experiments
uv run python scripts/summarize.py
```

## Structure

```
recgen/
  verbalizers/template.py   # TemplateVerbalizer
  encoder.py                # FrozenEncoder (MPS/CPU, pooling, batching)
  heads.py                  # ClassificationHead / RegressionHead (torch MLP)
  pipeline.py               # RecgenPipeline (sklearn-style fit/predict/transform)
  cache.py                  # sha256-keyed embedding cache
experiments/                # adult.py, house_prices.py, imdb.py + results/
notes/research.md           # GenRec/GenZ/TabLLM notes and lessons
```
