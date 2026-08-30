"""
Isolation Forest detector for out-of-distribution driving maneuver detection.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src.config import IsolationForestConfig, DEFAULT_IFOREST_CONFIG


class IsolationForestModel:
    """
    Wraps IsolationForest on NORMAL driving baselines
    to produce calibrated, normalized continuous risk scores in [0, 100].
    """

    def __init__(
        self,
        config: Optional[IsolationForestConfig] = None,
        feature_cols: Optional[List[str]] = None
    ):
        self.config = config or DEFAULT_IFOREST_CONFIG
        self.feature_cols = feature_cols
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
        if self.feature_cols is None:
            self.feature_cols = [c for c in X.columns if c not in ['Class', 'Timestamp'] and pd.api.types.is_numeric_dtype(X[c])]

        data = X[self.feature_cols]
        self.model.fit(data)
        scores = self.model.decision_function(data)
        self.score_min_ = float(scores.min())
        self.score_max_ = float(scores.max())
        self.fitted_ = True
        return self

    def score(self, X: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        if not self.fitted_:
            raise RuntimeError("IsolationForestModel must be fitted before scoring.")

        if isinstance(X, dict):
            features_df = pd.DataFrame([[float(X.get(col, 0.0)) for col in self.feature_cols]], columns=self.feature_cols)
            raw_score = float(self.model.decision_function(features_df)[0])
            norm = (raw_score - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-8)
            return float(np.clip((1.0 - norm) * 100.0, 0.0, 100.0))

        data = X[self.feature_cols]
        raw_scores = self.model.decision_function(data)
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-8)
        return pd.Series(np.clip((1.0 - norm_scores) * 100.0, 0.0, 100.0), index=data.index)

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestModel":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Isolation Forest model file not found at: {path}")
        return joblib.load(path)
