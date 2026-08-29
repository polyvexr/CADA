"""
Normal driving behavior baseline profiler for statistical deviation estimation.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import FEATURE_COLS


class NormalBaselineProfiler:
    """
    Learns statistical baseline parameters from known NORMAL driving telemetry:
    - Feature means and standard deviations
    - Multi-feature Z-score deviation distributions
    - Empirically calibrated percentile limits (e.g. 95th percentile)
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        percentile_cutoff: float = 95.0,
        eps: float = 1e-8
    ):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.percentile_cutoff = percentile_cutoff
        self.eps = eps

        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None
        self.normal_limit_: Optional[float] = None
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "NormalBaselineProfiler":
        """
        Fits baseline profile using only NORMAL driving samples.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing baseline normal driving features.
        """
        missing = [col for col in self.feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in baseline training data: {missing}")

        data = X[self.feature_cols]

        self.means_ = data.mean()
        self.stds_ = data.std().replace(0, self.eps)

        # Compute multi-feature absolute Z-scores for normal baseline
        z_scores = np.abs((data - self.means_) / self.stds_)
        mean_z_per_sample = z_scores.mean(axis=1)

        # Calculate empirical threshold (e.g., 95th percentile of normal driving variation)
        self.normal_limit_ = float(np.percentile(mean_z_per_sample, self.percentile_cutoff))
        self.fitted_ = True
        return self

    def score(self, X: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        """
        Computes normalized Statistical Risk Score (0 - 100).
        A score of 50 corresponds to the 95th percentile limit of normal driving.
        """
        if not self.fitted_:
            raise RuntimeError("NormalBaselineProfiler must be fitted before scoring.")

        if isinstance(X, dict):
            # Single sample scoring
            features_vec = np.array([X.get(col, 0.0) for col in self.feature_cols])
            z = np.abs((features_vec - self.means_.values) / self.stds_.values)
            dev_score = np.mean(z)
            stat_risk = float(np.clip((dev_score / self.normal_limit_) * 50.0, 0.0, 100.0))
            return stat_risk

        # Batch DataFrame scoring
        data = X[self.feature_cols]
        z_scores = np.abs((data - self.means_) / self.stds_)
        deviation_score = z_scores.mean(axis=1)

        stat_risk = np.clip((deviation_score / self.normal_limit_) * 50.0, 0.0, 100.0)
        return stat_risk

    def save(self, filepath: Union[str, Path]):
        """Persists fitted profiler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NormalBaselineProfiler":
        """Loads fitted profiler from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Baseline profiler file not found at: {path}")
        return joblib.load(path)

