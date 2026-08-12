# Marketplace recgen system design — embeddings as the product backbone

**Scope:** how a marketplace team can deploy the recgen (LLM-as-encoder)
architecture on their data and infra, covering moderation, search, and
recommendation from one embedding backbone.

**Date:** 2026-08 — based on the audited the moderation benchmark (4k products,
Gemini-3.1-flash-lite labels) and measured M1-Pro throughput; numbers are
directional and must be re-measured on the target GPUs.

---

## 0. Executive summary

- **One embedding backbone, three products:** the same product embeddings
  power (a) moderation heads, (b) semantic search, (c) recommendation
  candidates. Embeddings are computed **once** and stored in Qdrant; every
  downstream task is a cheap head or a vector query.
- **Yes, vLLM stays in the stack** — but as an **encoder server**
  (pooling/embeddings task, no generation). vLLM is what makes backfill and
  query encoding fast at scale. SGLang is the alternative with better
  prefix caching for the shared rules text.
- **Two embedding spaces** (deliberate): a *retrieval space* (SigLIP,
  image-text aligned) for search/recommendation, and a *task space*
  (1.7B LLM + SigLIP concat) for moderation heads. Both live in Qdrant.
- **Rule/policy changes never touch the encoder:** re-embed the new rules
  text once, retrain the head (minutes), ship it. Products are not
  re-encoded. This is the property that makes the system cheap to operate.
- **Cost shape:** on one A100, ~2.5-5M product encodes/day (encoder mode);
  heads are CPU-servable at >10k predictions/s; Qdrant is ~25GB for 5M
  products at 2048 dims. A 5M-product backfill is ~1-2 GPU-days.

---

## 1. Why this architecture (the numbers behind it)

Measured on the the moderation benchmark (M1 Pro; A100 figures projected):

| approach | accuracy vs Gemini eval | throughput (1 A100) | re-score after rule change |
|---|---|---|---|
| frozen 1.7B + SigLIP + GBDT head | 0.82-0.89 acc / 0.88 AUC | ~2.5-5M products/day | **free** (embeddings cached) |
| fine-tuned 9B VLM (20-tok JSON) | 0.90+ (if distilled on labels) | ~0.4-0.8M products/day | full re-prefill |
| fine-tuned 9B VLM (1-token decision) | 0.90+ | ~0.8-1.5M products/day | full re-prefill |

The accuracy gap of the head path is the cross-modal reading (brand on box,
specs printed on the product). The cascade in §6 captures most of that gap
while keeping the cost shape of the head path.

---

## 2. Embedding strategy — two spaces, one pipeline

### 2.1 Retrieval space (search + recommendation)
- **Encoder:** SigLIP2 (or the current best CLIP-class model). It is
  *contrastively trained*: image and text map into the same space, so a
  search query (text) can be compared directly to product images and
  product titles/descriptions.
- **Product vector:** `mean(SigLIP(image_i))` over up to N images, fused
  with `SigLIP(text: title + category + top attributes)` — or keep two
  vectors per product (image-vector, text-vector) for flexible querying.
- **Query vector:** `SigLIP(query text)`.
- **Use:** cosine similarity in Qdrant; themes/collections are just
  embeddings of their descriptions ("yılbaşı hediye fikirleri").

### 2.2 Task space (moderation + prediction heads)
- **Encoder:** SmolLM2-1.7B (or larger) mean-pooled over the verbalized
  product listing + SigLIP image embedding, concatenated (3200 dims in the
  benchmark).
- **Heads:** MLP (framework-native) or GBDT on embeddings. Multi-task:
  decision + rejection tags + category. Trained in seconds-to-minutes.
- **Note:** the moderation instruction is **not** baked into the product
  prompt (see §7 — policy separation). Product vectors are policy-agnostic.

### 2.3 Why not one space?
Retrieval wants cross-modal alignment (SigLIP); moderation wants deep
semantic understanding (LLM). Fusing both into one vector dilutes both.
Two spaces cost ~nothing extra — both are computed in the same pipeline
pass.

---

## 3. Serving topology

```
                    ┌──────────────────────────────────────────────┐
  catalog / CMS ──► │ ingestion worker (Kafka / cron)              │
                    │  verbalize + images ──► encode ──► Qdrant    │
                    └───────────────┬──────────────────────────────┘
                                    │ (embeddings, immutable)
         ┌──────────────┬───────────┴────────────┬─────────────────┐
         ▼              ▼                        ▼                 ▼
  moderation head  search API               recommendation    monitoring/
  (CPU service)   (Qdrant query)            candidates         eval jobs
         │              │                        │
         ▼              ▼                        ▼
  decision+tags    ranked products          ranked items     drift reports
  + VLM cascade
```

### 3.1 vLLM (or SGLang) — encoder server
- **Mode:** embeddings/pooling task — prefill-only, no generation. For
  causal LLMs, expose the pooled last-layer hidden state (mean pooling).
- **vLLM:** `--task pooling --pooling-method mean` (verify flags for the
  installed version). Alternative: SGLang with `return_hidden_states` +
  RadixAttention (free prefix caching when the verbalizer template is
  shared across products).
- **Why keep vLLM:** batch scheduling + high GPU utilization during
  backfill, and low-latency query encoding at serving time. A naive batch
  loop works up to ~10-20k encodes/day; above that, use vLLM/SGLang.
- **Vision encoder (SigLIP):** can run in the same process (torch, batched)
  — it is small (~400M-1B) and does not need vLLM. On A100 it encodes
  ~100-300 images/s.

### 3.2 Head service (stateless, CPU)
- Loads head weights (MLP or GBDT artifacts), serves `POST /predict`
  (vector in → decision + tags + calibrated confidence). <1ms/request,
  horizontal scaling by just adding pods — it is pure math, no GPU.
- Head artifacts are versioned (see §7); routing header `policy_version`.

### 3.3 Qdrant
- **Collections:**
  - `product_retrieval` — SigLIP vectors, payload: product_id, category,
    brand, seller, price, active flag, moderation status.
  - `product_task` — task-space vectors (or same collection with a
    different vector name — Qdrant supports multiple vectors per point).
  - `query_theme` — search queries and theme/collection embeddings
    (for analytics and re-identification).
- **Indexing:** HNSW, cosine; payload filters (category, seller, price
  range) applied at query time. Filtered search: `prefer`/`oversampling`.
- **Sizing:** 2048-dim fp16 ≈ 4KB/vector + ~1KB payload. 5M products ≈
  25-30GB + replicas — a small cluster. Qdrant is CPU/cheap-GPU; no model
  needed on search nodes.
- **Multimodal search:** since SigLIP aligns text and images, a query can
  match product images even when titles are poor (e.g., "kırmızı deri
  çanta" matching a photo whose title says "moda çanta").

### 3.4 Orchestration
- **Backfill:** batch job (Spark/Argo/Batch) reading the catalog, encoding
  with vLLM + SigLIP, upserting into Qdrant (idempotent by product_id).
- **Incremental:** new/updated products → same pipeline, event-driven
  (Kafka) or scheduled (e.g., every 15 min).
- **Ordering:** encode → Qdrant → heads run on the stored vectors (no
  re-encode in the serving path).

---

## 4. Sizing math (directional)

Assume: 5M products, 3 images/product, ~1.5k tokens verbalized text, A100
encoder at ~50-100k tokens/s prefill (batched):

| item | value |
|---|---|
| backfill text encode | ~5M × 1.5k tok = 7.5G tok ≈ **1-2 GPU-days** (1 A100) |
| backfill image encode | ~15M images ≈ 0.5-1 GPU-day |
| storage (Qdrant) | ~25-35GB + 1-2 replicas |
| moderation head | CPU, >10k pred/s per replica |
| semantic search p99 | 20-60ms (Qdrant HNSW, filtered) |
| incremental (e.g., 10k new products/day) | <15 min GPU/day |

Note: if the catalog is smaller (e.g., 500k products), backfill is a
few hours on one GPU.

---

## 5. Moderation workflow (production)

1. **Score:** head on task-space vector → `P(reject)` + predicted tags.
2. **Route:** if confidence high (e.g., `P < 0.2` or `P > 0.85`, thresholds
   chosen on the val set) → auto decision.
3. **Cascade:** mid-confidence → fine-tuned 1-token VLM judge (see §8)
   which reads the images/text directly → decision.
4. **Audit:** sample auto-decisions for human review; feed corrections
   back as new head-training data (active learning loop).
5. **Calibration:** isotonic/Platt on the head outputs per policy version.

This keeps ~85-90% of products on the cheap path while the VLM catches
the cross-modal contradictions (brand/spec mismatch) that the head cannot
see.

---

## 6. Search and recommendation use cases (the Qdrant payoff)

### 6.1 Semantic search
```
query text ──► SigLIP(query) ──► Qdrant.search(product_retrieval, top_k=50,
              filter={category, price, active:true}) ──► ranked products
```
Works with images via the same space (multimodal retrieval).

### 6.2 "Themes" (collections, campaigns)
Embed the theme text once, reuse forever:
- "yılbaşı hediye fikirleri" → top-k products via Qdrant.
- Re-ranking is trivial: head or rules-embedding conditioning.

### 6.3 Recommendation
Two compatible paths (both validated in this repo on MovieLens-1M):
- **Candidate generation:** user history → `mean(product_task vectors)` or
  `SigLIP(verbalized history)` → Qdrant kNN → candidate set (e.g., 300).
- **Ranking:** either (a) cosine over the same space, or (b) a trained
  catalog-aware head (`score(u,i) = <W·h_u, e_i> + b_i` — the GenRec head,
  HR@10 0.598 on MovieLens with a 1.7B encoder) on task-space vectors.
- **Cold start:** new products enter the embedding space immediately —
  no interaction history needed (the LLM/SigLIP semantics carry the item).

### 6.4 Multi-task upside
The task-space vectors can host other heads cheaply: category
classification, attribute extraction, duplicate/near-duplicate detection
(same vector neighborhood), seller quality scoring. Each is a head
training on frozen embeddings — the benchmark shows minutes, not days.

---

## 7. Policy change lifecycle (the robustness property)

1. Rules text changes → **one encode** of the new rules text (or a new
   `rules_emb` vector), keep product embeddings untouched.
2. Re-label a sample (Gemini eval or human review) → **retrain heads**
   (minutes; embeddings cached) → CI/CD artifact per `policy_version`.
3. Deploy head; Qdrant payload gains a `policy_version` field; dual-run
   old vs new heads on a shadow slice for a day.
4. Rollback = point the router at the previous head artifact.

No GPU re-encode of the catalog. This is the operational property that
distinguishes the architecture from re-running a VLM over the catalog on
every policy change.

---

## 8. Optional accuracy layer: fine-tuned VLM judge (1-token)

- **Model:** current 9B-class VLM (Qwen VLM family), QLoRA 4-bit on A100
  40GB (fits: ~5-6GB weights + LoRA + activations).
- **Task format:** input = system rules + product text + up to 4 images;
  output = exactly one decision token (`Onaylandı`/`Reddedildi`) — train
  with cross-entropy on the first token only, or short JSON if the team
  prefers reasons.
- **Critical:** train it to **predict the labeler** (distill Gemini/human
  decisions), not to moderate from first principles — zero-shot judges
  disagree with the existing label style (measured 0.55 agreement).
- **Serving:** vLLM, take the first token's logits; ~0.8-1.5M decisions/day
  per A100; used only in the mid-confidence band of the cascade (§5).
- **Re-training on policy change:** required (it bakes the policy into
  weights), which is exactly why it is the *exception* path, not the
  backbone.

---

## 9. Monitoring and evaluation

- **Embedding drift:** track mean-vector shift per category over time
  (new sellers/brands moving the distribution); alert on sudden changes.
- **Head calibration:** reliability diagram per policy version; precision
  at fixed recall operating points chosen with the business.
- **Label quality:** periodic re-eval of a fixed 500-product sample with a
  stronger model/human to measure labeler noise (ceiling estimate).
- **Search quality:** recall@k on logged queries with product clicks as
  implicit labels; A/B themes.
- **Recommendation:** the MovieLens protocol in this repo (leave-one-out +
  100 negatives, HR@10/NDCG@10, leak-free by construction) as the offline
  gate before every ranking-head release.

---

## 10. Team roadmap (suggested)

| phase | duration | deliverables |
|---|---|---|
| 1. Embedding backbone | 1-2 weeks | verbalizer templates; vLLM pooling server; SigLIP server; backfill script; Qdrant collections |
| 2. Moderation heads | 1 week | multi-task head (decision+tags), threshold tuning, cascade stub, eval harness (CV + leak checks) |
| 3. Search | 1 week | query API on Qdrant; theme embeddings; filtered search |
| 4. Recommendations | 1-2 weeks | history embedding → kNN candidates; catalog-aware ranking head (GenRec head) |
| 5. Policy-change CI | 1 week | head versioning, shadow deploy, rollback |
| 6. VLM cascade | 1-2 weeks (optional) | QLoRA distill on labels; 1-token serving; mid-confidence routing |

Total: ~6-9 weeks with 2 engineers, one A100, one Qdrant cluster.

---

## 11. Open questions the team must answer

- **Data size:** actual catalog size and new-product rate (sizing above is
  parametric).
- **Label target:** is Gemini-3.1-flash-lite the target, or a human-reviewed
  set? (Recommend: keep Gemini for scale, sample human reviews for the
  ceiling estimate.)
- **Latency budget** for moderation at ingestion (batch vs real-time).
- **Qdrant vs alternatives:** Qdrant assumed here (self-hosted, filters,
  multi-vector); Milvus/Weaviate are equivalent choices.
- **vLLM pooling-mode specifics** must be verified against the exact
  installed vLLM version at build time.

---

*Implementation notes from this repo: `experiments/moderation/` (moderation
benchmark), `experiments/movielens/` (GenRec ranking head + protocol),
`api/` (FastAPI reference serving with content-addressed caching),
`recgen/` (FrozenEncoder, heads, verbalizers, cache).*
