# Research notes: LLM-as-encoder for generic prediction tasks

## GenRec (Netflix Tech Blog, 2026-07-30) — the originating idea
netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3

- Two-phase framework:
  - **Phase 1**: adapt an OSS LLM into a domain foundation backbone (Netflix corpora:
    content understanding, member behavior). Updated rarely. Shared across apps.
  - **Phase 2**: post-train for the task (ranking) with multi-objective loss:
    catalog-aware cross-entropy ranking objective + LM objective + reward-weighted
    alignment loss. Refreshed frequently.
- **Architecture**: decoder-only LLM + catalog-aware scoring head. Pipeline:
  verbalizer V(history, context, item metadata) -> prompt x -> LLM forward ->
  pooled hidden state h -> head φ(h, e_item) (dot-product / small MLP) ->
  softmax over catalog -> ranking.
- **Serving**: prefill-only (no autoregressive decoding); context compaction
  (retain high-signal events, omit low-signal, compress binges, elaborate cold-start);
  shared prefixes for KV-cache reuse; smaller/distilled models.
- **Key results**:
  - vs production ranker: +1.6% MRR with 40x fewer Phase-2 labels; online A/B win.
  - Phase-1 vs OSS base: +10-20%. Phase-2 vs Phase-1: +35-50% (up to ~80% when stale).
  - Context can be cut to ~1/3 with negligible degradation (elbow point analysis).
  - Bigger backbones + more data scale consistently (scaling-law-like behavior).
- **Conceptual shift**: feature engineering -> *context engineering*.
  Prompt = new feature vector. LLM = shared preference/instance encoder.

## Why it generalizes (our thesis)
The same recipe works for ANY task where an instance can be verbalized as text:
verbalize -> frozen/adapted small LLM -> pooled hidden state -> lightweight head.
Regression and classification are special cases of the ranking head (logits/scalar).

## GenZ (arXiv 2512.24834) — strongest tabular precedent
"Foundational models as latent variable generators within traditional statistical models"

- Frozen LLM + iterative feature discovery (EM): discover semantic feature
  descriptors contrasting error groups, LLM classifies items into discovered
  features, learned statistical model maps features -> target.
- House prices: 12% median relative error vs 38% for GPT-5 knowledge-only baseline.
- Cold-start movie rec: predicts CF embeddings at 0.59 cosine similarity from
  semantic descriptions alone (~= 4000 user ratings of CF).
- Lesson: dataset-specific *semantic features* discovered from data beat raw LLM
  world-knowledge. (GenZ goes further than our frozen-encoder POC: it learns the
  feature *descriptions*, we learn the *head*.)

## From Logs to Language (arXiv 2602.20558, Netflix)
- Learned verbalization via RL beats rigid templates by up to 93% relative on
  discovery-item recommendation accuracy.
- Emergent strategies: interest summarization, noise removal, syntax normalization.
- Lesson: verbalizer design is the #1 lever after model choice. Our
  TemplateVerbalizer is the baseline; learned verbalization is a future phase.

## TabLLM (KDD 2023) — the negative result to know
- LLM (T0) prompts + serialization for tabular classification.
- Often beats strong baselines in the "small dataset / many-shot" regime but
  LOSES to GBDT on larger, numeric-heavy tabular data.
- Lesson: do not sell LLM encoders on pure-numeric tabular. Target:
  (a) text-heavy tasks, (b) small data, (c) high-cardinality categoricals,
  (d) features with semantic meaning a pretrained LM understands,
  (e) stacking with GBDT (embeddings as extra features).
  (f) tasks needing transfer across datasets (frozen embeddings reusable).

## Practical takeaways for recgen POC
1. Frozen 360M encoder on M1 Pro: ~30 prompts/s with batch=32, fp16, 512 tok cap.
2. Encode-once + cache makes head training cheap -> fast iteration.
3. MPS + libomp: LightGBM needs num_threads=1; pyarrow+libomp can segfault torch
   model loads -> set OMP_NUM_THREADS=1 process-wide (macOS quirk).
4. Benchmarks: LLM-head vs LightGBM vs TF-IDF (text) vs stacking variants.
5. Next phases: LoRA domain adaptation (GenRec Phase-1 analog), learned verbalizers,
   catalog-aware ranking head (recsys), prefill-only serving via vLLM.
