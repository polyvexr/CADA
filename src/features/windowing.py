"""
Rolling window feature extraction for temporal dynamics in motion telemetry.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_SENSOR_COLS


class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Computes rolling window statistical features (mean, std, min, max, energy)
    over sequential sensor measurements.
    """

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
        """
        Calculates windowed aggregations over target sensor columns.
        """
        df = X.copy()
        
        for col in self.target_cols:
            if col in df.columns:
                rolling = df[col].rolling(window=self.window_size, min_periods=1)
                
                # Rolling Mean
                df[f"{col}_roll_mean"] = rolling.mean()
                
                # Rolling Standard Deviation
                if self.include_std:
                    df[f"{col}_roll_std"] = rolling.std().fillna(0.0)
                
                # Rolling Range (Peak to Peak)
                if self.include_range:
                    df[f"{col}_roll_range"] = rolling.max() - rolling.min()
                
                # Rolling Energy
                if self.include_energy:
                    df[f"{col}_roll_energy"] = (df[col]**2).rolling(window=self.window_size, min_periods=1).mean()

        return df

