import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer

from common import record

MODELS = {
    "smol360": "models/SmolLM2-360M",
    "qwen06": "models/Qwen3-0.6B",
    "smol17": "models/SmolLM2-1.7B",
}

INSTRUCTION = "Given a movie review, predict whether the sentiment is positive or negative."

DEMO_QUERIES = [
    "warm overdrive tone for blues",
    "studio recording microphone",
    "cheap guitar strings for beginners",
    "portable practice amp",
]

DEMO_ITEMS = [
    "Item: Fender Player Stratocaster. Category: electric guitar. Single-coil pickups, classic twang.",
    "Item: Boss DS-1 Distortion Pedal. Category: effects pedal. Legendary hard rock distortion.",
    "Item: Ibanez TS9 Tube Screamer. Category: effects pedal. Vintage overdrive, warm mid boost.",
    "Item: Shure SM57 Microphone. Category: microphone. Cardioid, instrument miking standard.",
    "Item: Ernie Ball Slinky Strings. Category: strings. Nickel wound 10-46, bright tone.",
    "Item: Marshall MG30GFX Combo Amp. Category: amplifier. 30 watts, British crunch.",
]


def main(n_train: int = 2000, n_test: int = 500):
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas().sample(n_train + n_test, random_state=0)
    y = (df["label"] > 0.5).astype(int).to_numpy()
    texts = df["text"].tolist()

    Xtr, Xva, ytr, yva = train_test_split(texts, y, test_size=n_test / (n_train + n_test), random_state=0, stratify=y)

    for name, model_dir in MODELS.items():
        encoder = FrozenEncoder(model_dir, pooling="mean", batch_size=32, max_length=512)
        verbalizer = TemplateVerbalizer(instruction=INSTRUCTION)
        pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=f".cache/compare/{name}")
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        train_s = time.time() - t0
        proba = pipe.predict_proba(Xva)
        acc = accuracy_score(yva, pipe.predict(Xva))
        auc = roc_auc_score(yva, proba[:, 1])
        record(f"compare_{name}", acc=acc, auc=auc, train_s=train_s)
        print(f"{name}: acc={acc:.4f} auc={auc:.4f} train_s={train_s:.0f}")

        demos = encoder.encode(DEMO_ITEMS, progress=False)
        for q in DEMO_QUERIES:
            qe = encoder.encode([q], progress=False)[0]
            scores = np.dot(qe, demos.T)
            norms = np.linalg.norm(qe) * np.linalg.norm(demos, axis=1)
            scores = scores / norms
            tops = [DEMO_ITEMS[i].split(".")[0] for i in np.argsort(-scores)[:2]]
            print(f"   [{name}] '{q}' -> {tops}")


if __name__ == "__main__":
    main()
