<p align="center">
  <h1 align="center">recgen</h1>
  <p align="center"><b>LLMs are cheap when you stop generating.</b></p>
  <p align="center">Verbalize any instance as text -> encode with a small LLM -> train a lightweight head for classification, regression, or ranking.</p>
</p>

Inspired by Netflix's [GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)
(prefill-only LLM recommendation via a pooled hidden state + catalog-aware
head), `recgen` generalizes the "LLM-as-encoder" idea to any prediction task.

```
instance (row / history / text)
  -> verbalizer (template config)          # context engineering, not feature engineering
  -> small LLM (frozen)                    # SmolLM2-360M / Qwen / any HF model
  -> pooled hidden state h (mean/last)     # 960-dim semantic embedding
  -> lightweight head                      # ClassifierHead / RegressionHead / CatalogRankingHead
```

No autoregressive decoding at inference. One forward pass, one pooling op, one
small MLP. Cost scales with context length, not generation. Embeddings are
content-addressed and cached, so re-training heads costs seconds.

## Highlights

- **IMDB sentiment** (12k reviews): recgen 0.914 acc / 0.971 auc vs
  TF-IDF+LogReg 0.890 / 0.957 — beats a classic text baseline with a frozen
  360M model.
- **E-commerce next-item rec** (Amazon Musical Instruments, 25k users /
  13.5k-item catalog): ranking head trained in ~15s beats popularity (13x) and
  ALS (33x) with zero feature engineering; the 2-stage deployment
  (popularity + embedding-kNN candidates -> head) hits **34% recall@10 /
  50% recall@20 / MRR@20 0.160**.
- **Adult income**: 0.848/0.905 vs LightGBM 0.862/0.922 — within 1.5pts,
  no feature engineering.
- Honest limits: pure-numeric regression (house prices) still belongs to GBDT.

Full numbers and analysis: [blog post](blog/recgen-llm-as-encoder.md) and
[notes/poc_report.md](notes/poc_report.md).

## Install

```bash
uv sync                          # python 3.12 env with all deps
uv run hf download HuggingFaceTB/SmolLM2-360M --local-dir models/SmolLM2-360M
uv run pytest tests/ -q          # sanity checks (CPU, no model needed)
```

macOS note: export `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` before
running anything that mixes pyarrow/LightGBM/torch.

## Quickstart

```python
from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer

verbalizer = TemplateVerbalizer(
    fields=["age", "workclass", "occupation", "hours-per-week"],
    instruction="Predict whether annual income exceeds $50K.",
)
encoder = FrozenEncoder("models/SmolLM2-360M", pooling="mean", batch_size=32)
pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=".cache/my_task")

pipe.fit(X, y)                    # X: DataFrame or list[str]
pipe.predict(X_test)
pipe.transform(X_test)            # -> (n, 960) embeddings, for stacking/GBDT pipelines
```

Ranking (GenRec-style catalog-aware head):

```python
from recgen import CatalogRankingHead

head = CatalogRankingHead(dim=960)
head.fit(H_users, y_next_item, E_items)      # frozen LLM embeddings in, dot-product out
head.evaluate(H_test, y_test)                # recall@k, mrr@20 over the full catalog
```

## Experiments

```bash
scripts/run_all.sh                # adult + house prices + imdb
uv run python experiments/ecommerce/train_rank.py --n-users 25000
uv run python scripts/summarize.py
```

Results land in `experiments/results/results.csv`.

## Serving product (`api/`)

FastAPI service exposing `POST /v1/encode` and `POST /v1/rank` with
content-addressed caching — the "embeddings as a service" shape:

```bash
uv add fastapi uvicorn
uv run uvicorn api.app:app --port 8000
```

See [api/README.md](api/README.md).

## Repo layout

```
recgen/        encoder, heads, ranking head, verbalizers, pipeline, cache, LoRA trainer
experiments/   adult, house_prices, imdb, adult_lora, e-commerce next-item rec
api/           FastAPI service + Dockerfile
blog/          write-ups
notes/         research notes (GenRec, GenZ, TabLLM, From Logs to Language)
```

## License

MIT. The core is free and open; the hosted service is the business.
