"""
Master CADA Composite Continuous Risk Scoring Engine.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import DEFAULT_SCORER_CONFIG, CADAScorerConfig, FEATURE_COLS
from src.features.kinematics import KinematicFeatureExtractor
from src.models.baseline_profiler import NormalBaselineProfiler
from src.models.isolation_forest import IsolationForestModel
from src.scoring.risk_tiers import assign_risk_tier, assign_risk_tiers_batch, RiskTier


class CADACompositeScorer:
    """
    Master CADA continuous driving anomaly risk scoring system.
    Combines:
    1. Isolation Forest Anomaly Risk (geometric / distributional novelty)
    2. Statistical Z-Score Deviation Risk (calibrated against 95th percentile normal baseline)
    3. Temporal Rate-of-Change Jerk Risk (acceleration and angular velocity changes)
    """

    def __init__(
        self,
        baseline_profiler: Optional[NormalBaselineProfiler] = None,
        iso_model: Optional[IsolationForestModel] = None,
        config: Optional[CADAScorerConfig] = None
    ):
        self.baseline_profiler = baseline_profiler or NormalBaselineProfiler()
        self.iso_model = iso_model or IsolationForestModel()
        self.config = config or DEFAULT_SCORER_CONFIG
        self.kinematics = KinematicFeatureExtractor()

        # Calibration parameters for temporal variation
        self.temporal_min_: float = 0.0
        self.temporal_max_: float = 5.0
        self.fitted_: bool = False

        # Internal state for exponential moving average (EMA) streaming
        self._last_ema_score: Optional[float] = None

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "CADACompositeScorer":
        """
        Fits the entire CADA scoring engine using training data.
        Baseline parameters and Isolation Forest are calibrated strictly on NORMAL driving data.
        """
        # Ensure kinematic features are computed
        if 'AccMag' not in X_train.columns or 'AccMag_Change' not in X_train.columns:
            df_feat = self.kinematics.fit_transform(X_train)
        else:
            df_feat = X_train.copy()

        # Filter normal driving baseline
        if y_train is not None:
            normal_mask = (y_train == 'NORMAL')
            normal_feat = df_feat[normal_mask].reset_index(drop=True)
        else:
            normal_feat = df_feat

        # Fit Sub-models on NORMAL driving
        self.baseline_profiler.fit(normal_feat)
        self.iso_model.fit(normal_feat)

        # Calibrate temporal variation normalization bounds
        temporal_dev = np.abs(df_feat['AccMag_Change']) + np.abs(df_feat['GyroMag_Change'])
        self.temporal_min_ = float(temporal_dev.min())
        self.temporal_max_ = float(np.percentile(temporal_dev, 99.0) if len(temporal_dev) > 0 else 5.0)
        if self.temporal_max_ <= self.temporal_min_:
            self.temporal_max_ = self.temporal_min_ + 1.0

        self.fitted_ = True
        return self

    def compute_temporal_risk(self, df_or_sample: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        """Calculates normalized temporal jerk risk score in [0, 100]."""
        if isinstance(df_or_sample, dict):
            acc_chg = abs(float(df_or_sample.get('AccMag_Change', 0.0)))
            gyro_chg = abs(float(df_or_sample.get('GyroMag_Change', 0.0)))
            dev = acc_chg + gyro_chg
            norm = (dev - self.temporal_min_) / (self.temporal_max_ - self.temporal_min_ + 1e-8)
            return float(np.clip(norm * 100.0, 0.0, 100.0))

        df = df_or_sample
        dev = np.abs(df['AccMag_Change']) + np.abs(df['GyroMag_Change'])
        norm = (dev - self.temporal_min_) / (self.temporal_max_ - self.temporal_min_ + 1e-8)
        return np.clip(norm * 100.0, 0.0, 100.0)

    def score_sample(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """
        Scores a single streaming telemetry observation statefully.

        Parameters
        ----------
        sample : dict
            Raw sensor readings dict containing AccX, AccY, AccZ, GyroX, GyroY, GyroZ.

        Returns
        -------
        dict
            Enriched result with CADA_Score, Risk_Tier, and risk components.
        """
        if not self.fitted_:
            raise RuntimeError("CADACompositeScorer must be fitted before scoring.")

        # Compute kinematic features statefully
        enriched_sample = self.kinematics.transform_sample(sample)

        # Compute sub-scores
        iso_r = self.iso_model.score(enriched_sample)
        stat_r = self.baseline_profiler.score(enriched_sample)
        temp_r = self.compute_temporal_risk(enriched_sample)

        # Composite weighted score
        w1, w2, w3 = self.config.weight_iso, self.config.weight_stat, self.config.weight_temporal
        total_w = w1 + w2 + w3
        raw_cada_score = (w1 * iso_r + w2 * stat_r + w3 * temp_r) / total_w

        # Optional EMA smoothing for streaming
        if self.config.enable_ema:
            if self._last_ema_score is None:
                final_score = raw_cada_score
            else:
                alpha = self.config.ema_alpha
                final_score = alpha * raw_cada_score + (1 - alpha) * self._last_ema_score
            self._last_ema_score = final_score
        else:
            final_score = raw_cada_score

        tier = assign_risk_tier(final_score, self.config)

        return {
            "CADA_Score": round(float(final_score), 2),
            "Risk_Tier": tier.value,
            "Iso_Risk": round(float(iso_r), 2),
            "Stat_Risk": round(float(stat_r), 2),
            "Temporal_Risk": round(float(temp_r), 2),
            "AccMag": round(float(enriched_sample['AccMag']), 4),
            "GyroMag": round(float(enriched_sample['GyroMag']), 4),
            "AccMag_Change": round(float(enriched_sample['AccMag_Change']), 4),
            "GyroMag_Change": round(float(enriched_sample['GyroMag_Change']), 4)
        }

    def score_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a batch DataFrame of telemetry data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing raw sensors and/or kinematic features.

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with Iso_Risk, Stat_Risk, Temporal_Risk, CADA_Score, and Risk_Tier.
        """
        if not self.fitted_:
            raise RuntimeError("CADACompositeScorer must be fitted before scoring.")

        if 'AccMag' not in X.columns or 'AccMag_Change' not in X.columns:
            df = self.kinematics.transform(X)
        else:
            df = X.copy()

        iso_r = self.iso_model.score(df)
        stat_r = self.baseline_profiler.score(df)
        temp_r = self.compute_temporal_risk(df)

        w1, w2, w3 = self.config.weight_iso, self.config.weight_stat, self.config.weight_temporal
        total_w = w1 + w2 + w3
        cada_score = (w1 * iso_r + w2 * stat_r + w3 * temp_r) / total_w

        df['Iso_Risk'] = iso_r
        df['Stat_Risk'] = stat_r
        df['Temporal_Risk'] = temp_r
        df['CADA_Score'] = cada_score
        df['Risk_Tier'] = assign_risk_tiers_batch(cada_score, self.config)

        return df

    def reset_stream_state(self):
        """Resets streaming history state."""
        self.kinematics.reset_state()
        self._last_ema_score = None

    def save(self, filepath: Union[str, Path]):
        """Persists the complete CADA scorer bundle to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CADACompositeScorer":
        """Loads fitted CADA scorer from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CADA scorer file not found at: {path}")
        return joblib.load(path)

