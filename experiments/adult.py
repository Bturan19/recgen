import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer
from recgen.heads import ClassificationHead

from common import lgbm_baseline, lgbm_on_embeddings, record, run_pipeline

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/adult"

CAT_COLS = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
NUM_COLS = ["age", "capital.gain", "capital.loss", "hours.per.week"]
FEATURE_COLS = NUM_COLS + CAT_COLS

INSTRUCTION = "Given the profile of a person, predict whether their annual income exceeds $50K."


def load_adult():
    try:
        from datasets import load_dataset
        ds = load_dataset("scikit-learn/adult-census-income", split="all")
        df = ds.to_pandas()
        df.rename(columns={"income": "target"}, inplace=True)
        df = df.replace("?", pd.NA).dropna()
        df["target"] = (df["target"].astype(str).str.strip().str.startswith(">")).astype(int)
    except Exception as e:
        print(f"datasets hub failed ({e}); falling back to OpenML")
        from sklearn.datasets import fetch_openml
        df = fetch_openml("adult", version=2, as_frame=True).frame
        df = df.replace("?", pd.NA).dropna()
        df["target"] = (df["target"].astype(str).str.strip().str.startswith(">")).astype(int)
    return df


def lgbm_on_h_plus_raw(H, X_raw, y, cat_cols, random_state=0):
    from common import encode_categoricals
    from sklearn.preprocessing import StandardScaler

    Xc = encode_categoricals(X_raw, cat_cols).reset_index(drop=True)
    Xc = Xc.astype(float)
    Xs = StandardScaler().fit_transform(Xc)
    Xc = np.concatenate([Xs, H], axis=1)
    Xtr, Xva, ytr, yva = train_test_split(Xc, y, test_size=0.2, random_state=random_state, stratify=y)
    model = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=63, num_threads=1, random_state=random_state)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    return {
        "acc": accuracy_score(yva, model.predict(Xva)),
        "auc": roc_auc_score(yva, model.predict_proba(Xva)[:, 1]),
    }


def main(pooling: str = "last"):
    df = load_adult()
    print(f"Adult: {df.shape}, target mean = {df['target'].mean():.3f}")

    verbalizer = TemplateVerbalizer(fields=FEATURE_COLS, instruction=INSTRUCTION)
    encoder = FrozenEncoder(MODEL_DIR, pooling=pooling, batch_size=32, max_length=512)
    pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=CACHE_DIR)
    res = run_pipeline(pipe, df[FEATURE_COLS], df["target"], f"adult_llmhead_{pooling}")
    print(f"LLM-head ({pooling}): acc={res['acc']:.4f} auc={res['auc']:.4f}")

    texts = verbalizer.fit(df[FEATURE_COLS]).transform(df[FEATURE_COLS])
    H = encoder.encode_cached(texts, f"{CACHE_DIR}/emb_{pooling}.npy")
    y = df["target"].to_numpy()

    baseline = lgbm_baseline(df[FEATURE_COLS], df["target"], CAT_COLS)
    record("adult_lgbm_raw", acc=baseline["acc"], auc=baseline["auc"])
    print(f"LightGBM raw: acc={baseline['acc']:.4f} auc={baseline['auc']:.4f}")

    stack = lgbm_on_embeddings(H, y)
    record(f"adult_lgbm_on_h_{pooling}", acc=stack["acc"], auc=stack["auc"])
    print(f"LightGBM on h ({pooling}): acc={stack['acc']:.4f} auc={stack['auc']:.4f}")

    if pooling == "last":
        stack_raw = lgbm_on_h_plus_raw(H, df[FEATURE_COLS], y, CAT_COLS)
        record("adult_lgbm_h_plus_raw", acc=stack_raw["acc"], auc=stack_raw["auc"])
        print(f"LightGBM h+raw: acc={stack_raw['acc']:.4f} auc={stack_raw['auc']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = ap.parse_args()
    main(args.pooling)
