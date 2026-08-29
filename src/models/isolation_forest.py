"""
Isolation Forest detector for out-of-distribution driving maneuver detection.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src.config import FEATURE_COLS, IsolationForestConfig


class IsolationForestModel:
    """
    Wraps scikit-learn IsolationForest to fit on NORMAL driving baselines
    and produce calibrated, normalized continuous risk scores in [0, 100].
    """

    def __init__(
        self,
        config: Optional[IsolationForestConfig] = None,
        feature_cols: Optional[List[str]] = None
    ):
        self.config = config or IsolationForestConfig()
        self.feature_cols = feature_cols or FEATURE_COLS

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        self.score_min_: float = -0.5
        self.score_max_: float = 0.5
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "IsolationForestModel":
        """
        Fits Isolation Forest on baseline NORMAL driving features and computes calibration bounds.
        """
        missing = [col for col in self.feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        data = X[self.feature_cols]
        self.model.fit(data)

        # Calibrate decision function scaling bounds
        scores = self.model.decision_function(data)
        self.score_min_ = float(scores.min())
        self.score_max_ = float(scores.max())
        self.fitted_ = True
        return self

    def score(self, X: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        """
        Calculates normalized Isolation Forest Risk Score (0 - 100).
        Higher score indicates higher anomaly / out-of-distribution risk.
        """
        if not self.fitted_:
            raise RuntimeError("IsolationForestModel must be fitted before scoring.")

        if isinstance(X, dict):
            features_df = pd.DataFrame([[X.get(col, 0.0) for col in self.feature_cols]], columns=self.feature_cols)
            raw_score = float(self.model.decision_function(features_df)[0])
            norm = (raw_score - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-8)
            iso_risk = float(np.clip((1.0 - norm) * 100.0, 0.0, 100.0))
            return iso_risk

        data = X[self.feature_cols]
        raw_scores = self.model.decision_function(data)
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-8)
        iso_risk = np.clip((1.0 - norm_scores) * 100.0, 0.0, 100.0)
        return pd.Series(iso_risk, index=data.index)

    def save(self, filepath: Union[str, Path]):
        """Persists fitted model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestModel":
        """Loads fitted model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Isolation Forest model file not found at: {path}")
        return joblib.load(path)
