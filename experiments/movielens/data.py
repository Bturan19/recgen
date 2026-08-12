"""MovieLens-1M data prep for the GenRec-style next-item benchmark.

Protocol mirrors the SASRec paper (Kang & McAuley, ICDM'18):
- iterative 5-core filtering (users and items with >= 5 interactions)
- implicit feedback (presence of a rating)
- per-user split by timestamp: last action = test, second-to-last = val,
  all earlier actions = train
- eval: positive + 100 random negatives (excluding positive + history)

Leakage: MovieLens has at most one rating per (user, movie), so the held-out
movie is NEVER in the user's history — leak-free by construction. Movie
titles are unique, so no title-level collision either.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

DATA_DIR = "data/ml-1m"
RATINGS = f"{DATA_DIR}/ratings.dat"
MOVIES = f"{DATA_DIR}/movies.dat"

INSTRUCTION_HIST = "Given the user's movie-watching history, infer what they are likely to watch next."


def load_raw():
    import pandas as pd

    ratings = pl.from_pandas(
        pd.read_csv(RATINGS, sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    ).select(["user_id", "movie_id", "timestamp"])
    movies = pl.from_pandas(
        pd.read_csv(MOVIES, sep="::", header=None, names=["movie_id", "title", "genres"], engine="python", encoding="latin-1")
    ).with_columns(pl.col("genres").str.replace("(no genres listed)", "Unknown"))
    return ratings, movies


def prep_data(min_actions: int = 5):
    """Iterative k-core on users and items, sorted by timestamp."""
    ratings, movies = load_raw()
    for _ in range(20):
        user_counts = ratings.group_by("user_id").len()
        item_counts = ratings.group_by("movie_id").len()
        keep_users = set(user_counts.filter(pl.col("len") >= min_actions)["user_id"].to_list())
        keep_items = set(item_counts.filter(pl.col("len") >= min_actions)["movie_id"].to_list())
        n_before = len(ratings)
        ratings = ratings.filter(pl.col("user_id").is_in(keep_users) & pl.col("movie_id").is_in(keep_items))
        if len(ratings) == n_before:
            break
    ratings = ratings.sort(["user_id", "timestamp"])
    movies = movies.filter(pl.col("movie_id").is_in(keep_items))
    print(f"ml-1m {min_actions}-core: {len(ratings):,} interactions, "
          f"{ratings['user_id'].n_unique():,} users, {ratings['movie_id'].n_unique():,} items")
    return ratings, movies


def build_splits(ratings: pl.DataFrame, min_history: int = 3):
    """Per user: (history, val_item, test_item). History = all but last two."""
    splits = []
    for (uid,), g in ratings.group_by("user_id"):
        rows = [r["movie_id"] for r in g.sort("timestamp").to_dicts()]
        if len(rows) < min_history + 2:
            continue
        history = rows[:-2]
        val_item = rows[-2]
        test_item = rows[-1]
        splits.append((uid, history, val_item, test_item))
    return splits


def verbalize_item(movie_id: int, movies: pl.DataFrame) -> str:
    m = movies.filter(pl.col("movie_id") == movie_id)
    if m.is_empty():
        return f"Movie: {movie_id}"
    r = m.to_dicts()[0]
    return f"Movie: {r['title']}. Genres: {r['genres']}"


def verbalize_history(history: list[int], movies: pl.DataFrame, max_items: int = 20) -> str:
    m = movies.filter(pl.col("movie_id").is_in(history[-max_items:])).to_dicts()
    by_id = {r["movie_id"]: r for r in m}
    parts = []
    for mid in history[-max_items:]:
        r = by_id.get(mid)
        if r is None:
            parts.append(str(mid))
        else:
            parts.append(f"{r['title']} ({r['genres']})")
    return "The user recently watched: " + " | ".join(parts)


def make_negatives(splits, n_neg: int = 100, seed: int = 0):
    all_items = sorted({i for s in splits for i in s[1]} | {s[2] for s in splits} | {s[3] for s in splits})
    negs = {}
    for j, (uid, hist, val_item, test_item) in enumerate(splits):
        forbidden = set(hist) | {val_item, test_item}
        pool = np.array([i for i in all_items if i not in forbidden], dtype=np.int64)
        negs[j] = np.random.default_rng(seed + j).choice(pool, n_neg, replace=False)
    return negs, all_items
