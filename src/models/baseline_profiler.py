"""
Normal driving behavior baseline profiler for statistical deviation estimation.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


class NormalBaselineProfiler:
    """
    Learns statistical baseline parameters from NORMAL driving telemetry:
    - Multi-feature Z-score deviation distributions
    - Empirically calibrated percentile limits (e.g. 95th percentile)
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        percentile_cutoff: float = 95.0,
        eps: float = 1e-8
    ):
        self.feature_cols = feature_cols
        self.percentile_cutoff = percentile_cutoff
        self.eps = eps
        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None
        self.normal_limit_: Optional[float] = None
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "NormalBaselineProfiler":
        if self.feature_cols is None:
            self.feature_cols = [c for c in X.columns if c not in ['Class', 'Timestamp'] and pd.api.types.is_numeric_dtype(X[c])]

        data = X[self.feature_cols]
        self.means_ = data.mean()
        self.stds_ = data.std().replace(0, self.eps)

        z_scores = np.abs((data - self.means_) / self.stds_)
        mean_z_per_sample = z_scores.mean(axis=1)
        self.normal_limit_ = float(np.percentile(mean_z_per_sample, self.percentile_cutoff))
        if self.normal_limit_ <= self.eps:
            self.normal_limit_ = 1.0
        self.fitted_ = True
        return self

    def score(self, X: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        if not self.fitted_:
            raise RuntimeError("NormalBaselineProfiler must be fitted before scoring.")

        if isinstance(X, dict):
            features_vec = np.array([float(X.get(col, 0.0)) for col in self.feature_cols])
            z = np.abs((features_vec - self.means_.values) / self.stds_.values)
            dev_score = np.mean(z)
            return float(np.clip((dev_score / self.normal_limit_) * 50.0, 0.0, 100.0))

        data = X[self.feature_cols]
        z_scores = np.abs((data - self.means_) / self.stds_)
        deviation_score = z_scores.mean(axis=1)
        return np.clip((deviation_score / self.normal_limit_) * 50.0, 0.0, 100.0)

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NormalBaselineProfiler":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Baseline profiler file not found at: {path}")
        return joblib.load(path)

