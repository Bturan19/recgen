import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer

from common import lgbm_reg_baseline, lgbm_reg_on_embeddings, record, run_regression_pipeline

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/houseprices"

NUM_COLS = [
    "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal", "MoSold", "YrSold", "OverallQual", "OverallCond",
]
CAT_COLS = [
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType",
    "SaleCondition",
]
FEATURE_COLS = NUM_COLS + CAT_COLS

INSTRUCTION = "Given the features of a house, predict its sale price in US dollars."


def load_house_prices():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("house_prices", as_frame=True, parser="pandas", return_X_y=True)
    df = X
    df = df.dropna(axis=1, thresh=len(df) * 0.6)
    df = df[[c for c in FEATURE_COLS if c in df.columns]]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].astype("object")
            df[c] = df[c].fillna("None")
    df["SalePrice"] = y
    df = df[[c for c in FEATURE_COLS if c in df.columns] + ["SalePrice"]]
    return df


def main(pooling: str = "last"):
    df = load_house_prices()
    print(f"HousePrices: {df.shape}")
    feature_cols = [c for c in df.columns if c != "SalePrice"]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    y = df["SalePrice"].to_numpy()
    y_log = np.log1p(y)

    baseline = lgbm_reg_baseline(df[feature_cols], y_log, cat_cols)
    record("hp_lgbm_raw", mae=baseline["mae"], rmse=baseline["rmse"])
    print(f"LightGBM raw: mae={baseline['mae']:.4f} rmse={baseline['rmse']:.4f}")

    verbalizer = TemplateVerbalizer(fields=feature_cols, instruction=INSTRUCTION)
    encoder = FrozenEncoder(MODEL_DIR, pooling=pooling, batch_size=32, max_length=512)
    pipe = RecgenPipeline(encoder, verbalizer, head="regression", cache_dir=CACHE_DIR)
    res = run_regression_pipeline(pipe, df[feature_cols], y_log, f"hp_llmhead_{pooling}")
    print(f"LLM-head ({pooling}): mae={res['mae']:.4f} rmse={res['rmse']:.4f}")

    texts = verbalizer.fit(df[feature_cols]).transform(df[feature_cols])
    H = encoder.encode_cached(texts, f"{CACHE_DIR}/emb_{pooling}.npy")
    stack = lgbm_reg_on_embeddings(H, y_log)
    record(f"hp_lgbm_on_h_{pooling}", mae=stack["mae"], rmse=stack["rmse"])
    print(f"LightGBM on h ({pooling}): mae={stack['mae']:.4f} rmse={stack['rmse']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = ap.parse_args()
    main(args.pooling)
