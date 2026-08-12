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

- **E-commerce benchmark, leakage-audited** (Amazon Musical Instruments,
  23.9k clean users / 13.1k items, leave-one-out + 100 random negatives,
  repeat-purchase and title-collision users excluded): recgen
  **HR@10 0.421 / NDCG@10 0.238** vs ALS-128 0.304/0.174, popularity
  0.333/0.203, SASRec 0.128/0.058 — and it holds up on cold-start users
  (HR@10 0.417, ~equal to its overall rate; ID-based models degrade on
  sparse users). Head trains in ~15s; no feature engineering, no user/item
  ID embeddings. Audit: leakage-free by construction (see AUDIT.md).
- **Multi-label** (go_emotions, 28 labels): recgen micro-F1 0.400 vs
  TF-IDF+LogReg 0.376 — first multi-label real-data benchmark.
- **IMDB sentiment** (12k reviews): recgen 0.914 acc / 0.971 auc vs
  TF-IDF+LogReg 0.890 / 0.957 — beats a classic text baseline with a frozen
  360M model.
- **Adult income**: 0.848/0.905 vs LightGBM 0.862/0.922 — within 1.5pts,
  no feature engineering.
- **Serving speed on an M1 Pro** (re-audited 2026-08-09): one full
  recommendation request (300-token history + scoring a 100k-item catalog
  as a single multi-output matmul) takes ~30 ms — ~33 requests/s; projected
  GPU cost ~$0.63 per 1M requests. See `scripts/benchmark.py`.
- Honest limits: pure-numeric regression (house prices) and unsupervised
  anomaly detection belong to classical methods; the LLM encoder wins where
  semantics + labels meet.

Full numbers, corrected baselines, and analysis:
[notes/poc_report.md](notes/poc_report.md) and
[blog post](blog/recgen-llm-as-encoder.md).

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

## Quickstart example

```bash
uv run python examples/quickstart_ranking.py
```

Takes a 26-item catalog CSV, encodes it once (cached), and ranks it against a
purchase history — the whole onboarding flow in one file.

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
