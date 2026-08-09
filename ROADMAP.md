# Roadmap

Public product direction (strategy details are kept private).

## Where recgen is now (Aug 2026)

- Verbalizer -> frozen LLM encoder -> lightweight head, for classification,
  regression, multi-label, and catalog ranking.
- Benchmarked: IMDB (text clf), Adult (tabular clf), house prices (reg),
  SMS spam + cardiotocography (AD), Amazon Musical Instruments
  (next-item rec with SASRec/ALS baselines).
- Serving: FastAPI (`/v1/encode`, `/v1/rank`, `/v1/rank_all`),
  content-addressed caching, Docker. Measured ~35 req/s (300-token history +
  100k-item catalog scored in one matmul) on an M1 Pro.
- Honest negatives on file: numeric regression, unsupervised anomaly
  detection, and 0.6B-class backbones.

## Next

1. **Leakage-free reproducibility suite** — `AUDIT.md` + pinned seeds/caches;
   re-verify all headline numbers after the repeat-purchase filter.
2. **Denser benchmarks** — Amazon Beauty/Office (20-core) with published
   baseline numbers (SASRec/BERT4Rec from papers) for a defensible
   "competitive with SOTA-class" claim.
3. **Backbone options** — SmolLM2-1.7B quality tier, quantization (MLX /
   bitsandbytes) for CPU serving.
4. **Domain adaptation** — LoRA on unlabeled domain corpora (GenRec
   Phase-1 analog) once we have GPU access.
5. **Learned verbalization** — RL-tuned templates (Netflix: +93% relative).
6. **Platform** — hosted embeddings/rank API with auth + metering,
   candidate-generation service, dashboard.
7. **Distribution** — Kaggle notebooks, HF demo (needs Pro), blog series.

## How to contribute

Issues/PRs welcome. Benchmarks live in `experiments/`; every claim must be
reproducible via `scripts/run_all.sh` + the experiment scripts, and any new
result needs a leakage audit (see `AUDIT.md`).
