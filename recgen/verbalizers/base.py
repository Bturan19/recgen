from abc import ABC, abstractmethod

import numpy as np


class Verbalizer(ABC):
    @abstractmethod
    def transform(self, X) -> list[str]:
        pass

    @abstractmethod
    def fit(self, X):
        pass


def to_strings(X) -> list[str]:
    if isinstance(X, list):
        return X
    if isinstance(X, np.ndarray):
        return [str(v) for v in X.tolist()]
    import polars as pl

    if isinstance(X, pl.DataFrame):
        return [str(v) for v in X.rows()]
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X.apply(lambda r: str(r.to_dict()), axis=1).tolist()
    raise TypeError(f"Unsupported input type: {type(X)}")
