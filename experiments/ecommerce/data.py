import polars as pl

DATA_DIR = "data"
REVIEWS = f"{DATA_DIR}/raw/review_categories/Musical_Instruments.jsonl"
META_GLOBS = [f"{DATA_DIR}/raw_meta_Musical_Instruments/full-*.parquet"]


def load_meta() -> pl.DataFrame:
    df = pl.scan_parquet(META_GLOBS).collect()
    df = df.rename({"parent_asin": "item_id"})
    df = df.select(["item_id", "title", "store", "price", "categories", "features"]).unique(subset="item_id")
    df = df.filter(pl.col("title").is_not_null() & (pl.col("title").str.len_chars() > 3))
    return df


def load_reviews() -> pl.DataFrame:
    df = pl.scan_ndjson(REVIEWS).select(
        ["user_id", "parent_asin", "timestamp", "rating"]
    ).rename({"parent_asin": "item_id"}).collect()
    return df


def prep_data(min_user_items: int = 6, min_item_users: int = 5, max_history: int = 15):
    meta = load_meta()
    reviews = load_reviews()
    items = set(meta["item_id"].to_list())
    reviews = reviews.filter(pl.col("item_id").is_in(items))
    print(f"reviews: {len(reviews):,}, items with meta: {meta.shape[0]:,}")

    item_counts = reviews.group_by("item_id").len().rename({"len": "n"})
    keep_items = item_counts.filter(pl.col("n") >= min_item_users)["item_id"].to_list()
    reviews = reviews.filter(pl.col("item_id").is_in(keep_items))
    print(f"after item filter (>= {min_item_users} users): {len(keep_items):,} items, {len(reviews):,} reviews")

    user_counts = reviews.group_by("user_id").len().rename({"len": "n"})
    keep_users = user_counts.filter(pl.col("n") >= min_user_items)["user_id"].to_list()
    reviews = reviews.filter(pl.col("user_id").is_in(keep_users))
    print(f"after user filter (>= {min_user_items} interactions): {len(keep_users):,} users, {len(reviews):,} reviews")

    reviews = reviews.sort(["user_id", "timestamp"])
    return reviews, meta


def build_splits(reviews: pl.DataFrame, max_history: int = 15):
    train, test = [], []
    for (uid,), g in reviews.group_by("user_id"):
        rows = g.to_dicts()
        test_item = rows[-1]["item_id"]
        history = rows[:-1][-max_history:]
        if len(history) >= 5:
            train.append((uid, [h["item_id"] for h in history], test_item))
    return train


def verbalize_history(history_items: list[str], meta: pl.DataFrame) -> str:
    m = meta.filter(pl.col("item_id").is_in(history_items))
    parts = []
    for r in m.sort("item_id").to_dicts():
        brand = f" by {r['store']}" if r.get("store") else ""
        parts.append(f"{r['title']}{brand}")
    return "The user recently purchased: " + " | ".join(parts)


def verbalize_item(item_id: str, meta: pl.DataFrame) -> str:
    m = meta.filter(pl.col("item_id") == item_id)
    if m.is_empty():
        return f"Item: {item_id}"
    r = m.to_dicts()[0]
    brand = f" by {r['store']}" if r.get("store") else ""
    cats = ""
    c = r.get("categories")
    if c:
        cats = " Categories: " + ", ".join(c)
    feats = ""
    f = r.get("features") or []
    if f:
        feats = " Features: " + "; ".join(str(x) for x in f[:3])
    return f"Item: {r['title']}{brand}.{cats}{feats}"
