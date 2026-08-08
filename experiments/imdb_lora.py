import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from recgen import FrozenEncoder, TemplateVerbalizer
from recgen.trainer import LoraHeadTrainer

from common import record
from imdb import INSTRUCTION, load_imdb

MODEL_DIR = "models/SmolLM2-360M"


def main(n_train: int = 4000):
    texts, y = load_imdb()
    Xtr, Xva, ytr, yva = train_test_split(texts, y, test_size=0.2, random_state=0, stratify=y)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xtr), size=n_train, replace=False)
    Xsub = [Xtr[i] for i in sub]
    ysub = ytr[sub]

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=16, max_length=512)
    trainer = LoraHeadTrainer(
        encoder,
        task="classifier",
        lora_rank=16,
        lora_alpha=32,
        epochs=5,
        batch_size=16,
        grad_accum=1,
        patience=2,
        max_length=512,
    )
    ckpt = f"checkpoints/imdb_lora_sub{n_train}"
    if os.path.isdir(f"{ckpt}/adapter"):
        print(f"[imdb_lora] loading existing checkpoint {ckpt}")
        trainer.load(ckpt, Xtr, ytr)
    else:
        trainer.fit(Xsub, ysub, out_dir=ckpt)

    proba = trainer.predict_proba(Xva)
    acc = accuracy_score(yva, trainer.predict(Xva))
    auc = roc_auc_score(yva, proba[:, 1])
    record(f"imdb_lora_sub{n_train}", acc=acc, auc=auc)
    print(f"IMDB LoRA ({n_train}): acc={acc:.4f} auc={auc:.4f}")
    print("  head-only mean (12k train): acc=0.9137 auc=0.9712 | tfidf: acc=0.8897 auc=0.9574")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    args = ap.parse_args()
    main(args.n_train)
