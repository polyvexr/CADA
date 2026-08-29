"""
Motion data preprocessor for cleaning, sorting, and imputing telemetry data.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_SENSOR_COLS, TIMESTAMP_COL
from src.data.loader import validate_motion_dataframe


class MotionDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for cleaning raw motion sensor telemetry data.
    """

    def __init__(
        self,
        sensor_cols: Optional[List[str]] = None,
        timestamp_col: Optional[str] = None,
        sort_by_timestamp: bool = True,
        handle_duplicates: bool = True,
        clip_quantiles: Optional[tuple] = None
    ):
        self.sensor_cols = sensor_cols or RAW_SENSOR_COLS
        self.timestamp_col = timestamp_col or TIMESTAMP_COL
        self.sort_by_timestamp = sort_by_timestamp
        self.handle_duplicates = handle_duplicates
        self.clip_quantiles = clip_quantiles
        self.clip_bounds_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits preprocessor parameters (such as quantile clipping bounds if configured).
        """
        validate_motion_dataframe(X, require_target=False, sensor_cols=self.sensor_cols)
        
        if self.clip_quantiles:
            low_q, high_q = self.clip_quantiles
            for col in self.sensor_cols:
                low_val = X[col].quantile(low_q)
                high_val = X[col].quantile(high_q)
                self.clip_bounds_[col] = (low_val, high_val)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning, duplicate removal, sorting, and clipping to the motion DataFrame.
        """
        validate_motion_dataframe(X, require_target=False, sensor_cols=self.sensor_cols)
        df = X.copy()

        # Deduplicate
        if self.handle_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        # Sort by timestamp if available
        if self.sort_by_timestamp and self.timestamp_col in df.columns:
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        # Handle missing values if any
        if df[self.sensor_cols].isnull().any().any():
            df[self.sensor_cols] = df[self.sensor_cols].ffill().bfill()

        # Apply clipping bounds if fitted
        if self.clip_bounds_:
            for col, (low, high) in self.clip_bounds_.items():
                df[col] = df[col].clip(lower=low, upper=high)

        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

