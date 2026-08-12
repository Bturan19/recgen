# recgen-api — LLM embeddings & ranking as a service

Prefill-only inference (GenRec-style): a small LLM encodes prompts once into
semantic embeddings; lightweight heads predict/rank. No autoregressive
decoding, so cost scales with context length, not generation.

## Run locally

```bash
uv add fastapi uvicorn
RECGEN_MODEL_DIR=models/SmolLM2-360M uv run uvicorn api.app:app --port 8000
```

## Endpoints

- `GET /health` — model, pooling, cache stats
- `POST /v1/encode` — text or structured rows → embeddings (cacheable)

```json
{
  "texts": ["The user recently purchased: Zoom A2 Acoustic Guitar Pedal by Zoom"]
}
// or structured rows:
{
  "rows": [{"age": 39, "occupation": "Adm-clerical", "hours-per-week": 40}],
  "fields": ["age", "occupation", "hours-per-week"],
  "instruction": "Predict whether annual income exceeds $50K.",
  "cache_key": "my_dataset_v1"
}
```

- `POST /v1/rank` — score candidate items against a context

```json
{
  "context": "The user recently purchased: Zoom A2 Acoustic Guitar Pedal by Zoom",
  "items": ["Item: Fender cable", "Item: Boss distortion pedal"],
  "top_k": 2
}
```

- `POST /v1/rank_all` — score a full catalog in one multi-output matmul
  (GenRec-style prefill-only serving); catalog embeddings are cached under a
  key that includes the model + pooling, so backend switches never reuse stale
  embeddings

```json
{
  "context": "The user recently purchased: Zoom A2 Acoustic Guitar Pedal by Zoom",
  "catalog": ["Item: Fender cable", "Item: Boss distortion pedal"],
  "top_k": 2
}
```

## Economics (why this is cheap)

1. One forward pass per request — no token-by-token generation.
2. Embeddings are content-addressed and cached (`cache_key`) — repeated
   requests cost ~0.
3. The serving backbone is a 360M-1B model — runs on a laptop or cheap CPU/GPU.
4. Heads are tiny MLPs or catalog dot-products — a trained recommender scores
   thousands of items in milliseconds.

## Product roadmap

- [x] `/v1/encode`, `/v1/rank` (cosine/trained-head scoring)
- [x] `/v1/rank_all` (one-encode + one-matmul full-catalog scoring)
- [ ] Trained `CatalogRankingHead` checkpoints served via `RECGEN_HEAD_DIR`
- [ ] Authentication + usage metering (the monetizable layer)
- [ ] Quantized backends (bitsandbytes / MLX) for CPU serving
