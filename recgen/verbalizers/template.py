from dataclasses import dataclass, field
from typing import Optional

import polars as pl
import pandas as pd

from .base import Verbalizer


@dataclass
class TemplateVerbalizer(Verbalizer):
    fields: Optional[list[str]] = None
    instruction: str = ""
    separator: str = " | "
    missing: str = "unknown"
    _fit_fields: list[str] = field(default_factory=list, repr=False)

    def fit(self, X):
        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            cols = list(X.columns)
            if self.fields is None:
                self._fit_fields = cols
            else:
                missing = [f for f in self.fields if f not in cols]
                if missing:
                    raise ValueError(f"Fields not in frame: {missing}")
                self._fit_fields = self.fields
        elif self.fields is not None:
            self._fit_fields = self.fields
        return self

    def transform(self, X) -> list[str]:
        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            return [self._row_text(row) for _, row in X.iterrows()]
        return list(X)

    def _row_text(self, row) -> str:
        parts = []
        for f in self._fit_fields:
            v = row[f]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                v = self.missing
            elif isinstance(v, float) and v.is_integer():
                v = int(v)
            parts.append(f"{f}: {v}")
        body = self.separator.join(parts)
        if self.instruction:
            return f"{self.instruction}\n{body}"
        return body

    def transform_rows(self, rows: list[dict]) -> list[str]:
        return [self._row_text(r) for r in rows]
