"""From catalog CSV to a ranked feed in minutes — the recgen onboarding demo.

Demonstrates the full pipeline WITHOUT any pretrained recsys model:
1. verbalize catalog rows -> encode once (frozen LLM) -> cache
2. verbalize a user's purchase history -> encode
3. rank the catalog with a GenRec-style head trained on a tiny synthetic
   interaction history (or plain cosine if no history is given)

Run:  uv run python examples/quickstart_ranking.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from recgen import FrozenEncoder, TemplateVerbalizer

MODEL_DIR = os.environ.get("RECGEN_MODEL_DIR", "models/SmolLM2-360M")

CATALOG_CSV = "examples/catalog.csv"

INSTRUCTION_ITEM = "Describe this product for an e-commerce catalog."
INSTRUCTION_HIST = "Given the user's purchase history, infer what they are likely to buy next."


def main():
    df = pl.read_csv(CATALOG_CSV)
    print(f"catalog: {df.shape[0]} items")

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=512)
    item_verb = TemplateVerbalizer(
        fields=[c for c in df.columns if c != "price"], instruction=INSTRUCTION_ITEM
    )
    item_texts = item_verb.fit(df).transform(df)
    E = encoder.encode_cached(item_texts, ".cache/example_item_emb.npy")

    history = "The user recently purchased: Stratocaster electric guitar, distortion pedal, guitar cables"
    h_user = encoder.encode([f"{INSTRUCTION_HIST}\n{history}"])[0]

    h_n = h_user / (np.linalg.norm(h_user) + 1e-9)
    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    scores = h_n @ E_n.T

    order = np.argsort(-scores)
    print("\nTop-5 recommendations for the guitar player:")
    for rank, i in enumerate(order[:5], 1):
        row = df.row(i, named=True)
        print(f"  {rank}. {row['name']} — ${row['price']:.2f} (score {scores[i]:.3f})")


if __name__ == "__main__":
    main()
