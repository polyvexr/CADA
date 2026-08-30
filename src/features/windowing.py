"""
Windowing feature utilities for backwards compatibility.
"""

from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import RAW_SENSOR_COLS


class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """Computes rolling window statistics over target sensor columns."""

    def __init__(
        self,
        window_size: int = 5,
        target_cols: Optional[List[str]] = None,
        include_std: bool = True,
        include_range: bool = True,
        include_energy: bool = True
    ):
        self.window_size = window_size
        self.target_cols = target_cols or RAW_SENSOR_COLS
        self.include_std = include_std
        self.include_range = include_range
        self.include_energy = include_energy

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.target_cols:
            if col in df.columns:
                roll = df[col].rolling(window=self.window_size, min_periods=1)
                df[f"{col}_roll_mean"] = roll.mean()
                if self.include_std:
                    df[f"{col}_roll_std"] = roll.std().fillna(0.0)
                if self.include_range:
                    df[f"{col}_roll_range"] = roll.max() - roll.min()
                if self.include_energy:
                    df[f"{col}_roll_energy"] = (df[col]**2).rolling(window=self.window_size, min_periods=1).mean()
        return df

