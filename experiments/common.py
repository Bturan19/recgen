import csv
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RESULTS_DIR = Path(__file__).parent / "results"


def record(name, **metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    row = {"name": name, **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}}
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  recorded {name}: {row}")


def encode_categoricals(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def lgbm_baseline(X, y, cat_cols, random_state=0, metric="accuracy"):
    X = encode_categoricals(X, cat_cols).reset_index(drop=True)
    y = np.asarray(y).ravel()
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    cats = [X.columns.get_loc(c) for c in cat_cols]
    params = dict(objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=63, num_threads=1, random_state=random_state)
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    p = model.predict(Xva)
    proba = model.predict_proba(Xva)[:, 1]
    acc = accuracy_score(yva, p)
    auc = roc_auc_score(yva, proba)
    return {"acc": acc, "auc": auc, "model": model}


def lgbm_on_embeddings(H, y, random_state=0):
    Xtr, Xva, ytr, yva = train_test_split(H, y, test_size=0.2, random_state=random_state, stratify=y)
    model = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=63, num_threads=1, random_state=random_state)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    return {
        "acc": accuracy_score(yva, model.predict(Xva)),
        "auc": roc_auc_score(yva, model.predict_proba(Xva)[:, 1]),
    }


def run_pipeline(pipeline, X, y, tag, random_state=0):
    y = np.asarray(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    t0 = time.time()
    pipeline.fit(Xtr, ytr)
    train_s = time.time() - t0
    p = pipeline.predict(Xva)
    proba = pipeline.predict_proba(Xva)
    acc = accuracy_score(yva, p)
    auc = roc_auc_score(yva, proba[:, 1])
    record(tag, acc=acc, auc=auc, train_s=train_s)
    return {"acc": acc, "auc": auc}


def run_regression_pipeline(pipeline, X, y, tag, random_state=0):
    y = np.asarray(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=random_state)
    t0 = time.time()
    pipeline.fit(Xtr, ytr)
    train_s = time.time() - t0
    p = pipeline.predict(Xva)
    mae = mean_absolute_error(yva, p)
    rmse = root_mean_squared_error(yva, p)
    record(tag, mae=mae, rmse=rmse, train_s=train_s)
    return {"mae": mae, "rmse": rmse}


def lgbm_reg_baseline(X, y, cat_cols, random_state=0):
    X = encode_categoricals(X, cat_cols).reset_index(drop=True)
    y = np.asarray(y).ravel()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=random_state)
    cats = [X.columns.get_loc(c) for c in cat_cols]
    model = lgb.LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=63, num_threads=1, random_state=random_state)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = model.predict(Xva)
    return {
        "mae": mean_absolute_error(yva, p),
        "rmse": root_mean_squared_error(yva, p),
        "model": model,
    }


def lgbm_reg_on_embeddings(H, y, random_state=0):
    Xtr, Xva, ytr, yva = train_test_split(H, y, test_size=0.2, random_state=random_state)
    model = lgb.LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=63, num_threads=1, random_state=random_state)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return {
        "mae": mean_absolute_error(yva, model.predict(Xva)),
        "rmse": root_mean_squared_error(yva, model.predict(Xva)),
    }
