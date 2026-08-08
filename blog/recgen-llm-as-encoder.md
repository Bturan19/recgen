# LLMs are cheap when you stop generating: a generic "LLM-as-encoder" framework

*The GenRec insight, generalized to any classification, regression, or ranking task — with a 360M model, a laptop, and honest benchmarks.*

---

In July 2026, Netflix's tech blog described **GenRec**: an LLM-based ranker that
verbalizes user histories into natural language, runs the prompt through a
foundation LLM *once*, and attaches a catalog-aware scoring head on the pooled
hidden state. No autoregressive decoding. Just one forward pass and a small
head — in GenRec's words, the shift from *feature engineering* to *context
engineering*.

That architecture is not specific to movies. **Any** task where you can
verbalize an instance as text can be reduced to:

```
instance -> text (verbalizer) -> small LLM (encoder) -> pooled vector h -> tiny head
```

We built **recgen** — an open-source framework that does exactly this for
classification, regression, and ranking. This post is the honest story of what
we found, including where it fails.

## The core idea

The expensive part of an LLM is *generation*. An encoder-only forward pass on a
360M model costs fractions of a cent and runs on a laptop. The bet: the hidden
states of a pretrained LLM, pooled over a verbalized instance, already encode
semantic structure that a cheap MLP head can exploit for your task.

Three ingredients:

1. **Verbalizer** — your feature engineering, now as text templates
   ("Age: 39 | Occupation: Adm-clerical | Hours-per-week: 40").
2. **Frozen encoder** — a small LLM (we use SmolLM2-360M, 690MB, fp16 on an
   M1 Pro) producing one 960-dim vector per instance via masked mean-pooling.
3. **Lightweight head** — MLP for classification/regression; a GenRec-style
   catalog-aware head `score(u,i) = <W·h_u, e_i> + b_i` for ranking.

Embeddings are content-addressed and cached (sha256-keyed). Once cached, head
training takes seconds.

## Results

### Text classification — wins

| model | acc | auc |
|---|---|---|
| **recgen (frozen 360M + MLP head)** | **0.914** | **0.971** |
| TF-IDF + logistic regression | 0.890 | 0.957 |

IMDB sentiment, 12k training reviews. Mean pooling matters: last-token pooling
gets 0.884 — a 3-point swing from a pooling choice.

### Tabular classification — close

Adult income: recgen 0.848 acc / 0.905 auc vs LightGBM 0.862 / 0.922.
Within ~1.5 points, with zero feature engineering and no tuning.

### Tabular regression — loses, honestly

House prices: recgen 0.34 MAE (log$) vs LightGBM 0.087. On pure numeric
tabular, GBDT remains king. This matches the TabLLM result and we're not going
to pretend otherwise.

### E-commerce next-item recommendation — the home turf

Amazon Musical Instruments, next-item prediction over a 6,941-item catalog:

| model | recall@10 | recall@20 | mrr@20 |
|---|---|---|---|
| **recgen ranker** | **0.028** | **0.044** | **0.0162** |
| popularity | 0.022 | 0.035 | 0.0113 |
| ALS (32 factors) | 0.003 | 0.004 | 0.0024 |

The ranking head — trained in **5 seconds** on frozen embeddings — beats
popularity by ~25% relative MRR and ALS by ~7x. Purchase histories and item
metadata (title, store, categories, features) live in one shared semantic
space. No collaborative feature engineering. No user/item ID embeddings.

## What we learned the hard way

- **Small-n joint LoRA does not help.** Training LoRA + head jointly on 4k
  rows gave 0.812 acc vs 0.831 for head-only on the same data — the extra
  parameters overfit. GenRec's Phase-1 gains come from *unlabeled domain
  pretraining*, not small task sets.
- **Pooling is a free accuracy lever.** mean > last consistently (text: +3pts).
- **M1/MPS quirks:** `use_cache=False` + calling the base model directly
  avoids the notorious "Placeholder storage" crash; `OMP_NUM_THREADS=1` avoids
  libomp segfaults; batch ≥ 16 for reasonable GPU utilization.

## The economics (why this is a product, not just a demo)

- One forward pass per instance — no token-by-token decoding. Cost scales with
  context length, not generation.
- Embeddings are cacheable — repeated requests cost ~0.
- A 360M model serves comfortably on a laptop; heads are tiny.
- A trained ranker scores thousands of catalog items in milliseconds
  (`predict_scores` is one matmul).

We wrapped this in a FastAPI service (`api/`) — `POST /v1/encode`,
`POST /v1/rank`, content-addressed caching, Docker-ready. The trained
embeddings of your users/items are the asset; the heads are the cheap
per-task layer.

## Honest limitations

- Pure-numeric tabular: use GBDT (or stack — embeddings as extra features).
- The absolute numbers on next-item prediction are low because the task is
  genuinely hard (6.9k items, one held-out purchase); the *relative* wins and
  the zero-feature-engineering workflow are the point.
- 360M is not 70B — for zero-shot reasoning you need bigger models; recgen is
  for the *learned head* regime.

## Try it

```bash
uv sync
uv run hf download HuggingFaceTB/SmolLM2-360M --local-dir models/SmolLM2-360M
uv run python experiments/imdb.py --pooling mean   # +2.4 acc over TF-IDF
```

- Repo: https://github.com/Bturan19/recgen
- Results log: `experiments/results/results.csv`
- Notes on GenRec, GenZ, TabLLM, and Netflix's "From Logs to Language":
  `notes/research.md`

*recgen is a weekend project grown into a framework. If you build something on
it, or want embeddings-as-a-service for your catalog, we'd love to hear from
you.*
