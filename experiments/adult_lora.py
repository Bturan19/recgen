import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer
from recgen.trainer import LoraHeadTrainer

from adult import CAT_COLS, FEATURE_COLS, INSTRUCTION, load_adult
from common import record

MODEL_DIR = "models/SmolLM2-360M"


def main(n_train: int = 4000):
    df = load_adult()
    print(f"Adult: {df.shape}, LoRA train subset: {n_train}")

    verbalizer = TemplateVerbalizer(fields=FEATURE_COLS, instruction=INSTRUCTION)
    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=512)

    X = df[FEATURE_COLS]
    y = df["target"].to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xtr), size=n_train, replace=False)
    Xsub, ysub = Xtr.iloc[sub], ytr[sub]

    pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=".cache/adult_lora_sub")
    from common import run_pipeline
    res = run_pipeline(pipe, Xsub, ysub, f"adult_headonly_sub{n_train}")
    print(f"head-only ({n_train}): acc={res['acc']:.4f} auc={res['auc']:.4f}")

    trainer = LoraHeadTrainer(
        encoder,
        task="classifier",
        lora_rank=8,
        lora_alpha=16,
        epochs=4,
        batch_size=32,
        grad_accum=2,
        patience=2,
        max_length=512,
    )
    ckpt = f"checkpoints/adult_lora_sub{n_train}"
    if os.path.isdir(f"{ckpt}/adapter"):
        print(f"[adult_lora] loading existing checkpoint {ckpt}")
        trainer.load(ckpt, Xtr, ytr, verbalizer=verbalizer)
    else:
        trainer.fit(Xsub, ysub, verbalizer=verbalizer, out_dir=ckpt)

    proba = trainer.predict_proba(Xva)
    acc = accuracy_score(yva, trainer.predict(Xva))
    auc = roc_auc_score(yva, proba[:, 1])
    record(f"adult_lora_sub{n_train}", acc=acc, auc=auc)
    print(f"LoRA ({n_train}): acc={acc:.4f} auc={auc:.4f}")
    print("  head-only full: acc=0.8477 auc=0.9052 | lgbm raw: acc=0.8621 auc=0.9217")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    args = ap.parse_args()
    main(args.n_train)
