"""
Master CADA Composite Continuous Risk Scoring Engine.
Fuses Supervised Maneuver Probability, Isolation Forest Novelty,
Statistical Z-Score Deviation, and Multi-Scale Kinematic Jerk Dynamics.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

from src.config import DEFAULT_SCORER_CONFIG, CADAScorerConfig
from src.features.kinematics import KinematicFeatureExtractor
from src.models.baseline_profiler import NormalBaselineProfiler
from src.models.isolation_forest import IsolationForestModel
from src.scoring.risk_tiers import assign_risk_tier, assign_risk_tiers_batch, RiskTier


class CADACompositeScorer:
    """
    Master CADA continuous driving anomaly risk scoring system.
    Synthesizes:
    1. Supervised Gradient Maneuver Risk (learned aggressive motion probability)
    2. Isolation Forest Novelty Risk (geometric / distributional novelty vs normal driving)
    3. Statistical Z-Score Deviation Risk (calibrated against 95th percentile baseline)
    4. Multi-Scale Kinematic Jerk Risk (acceleration, angular velocity, and volatility dynamics)
    """

    def __init__(
        self,
        baseline_profiler: Optional[NormalBaselineProfiler] = None,
        iso_model: Optional[IsolationForestModel] = None,
        supervised_model: Optional[HistGradientBoostingClassifier] = None,
        config: Optional[CADAScorerConfig] = None
    ):
        self.baseline_profiler = baseline_profiler or NormalBaselineProfiler()
        self.iso_model = iso_model or IsolationForestModel()
        self.supervised_model = supervised_model
        self.config = config or DEFAULT_SCORER_CONFIG
        self.kinematics = KinematicFeatureExtractor(windows=self.config.temporal_windows)

        # Feature column tracking
        self.feature_cols_: Optional[list] = None

        # Calibration parameters for temporal variation
        self.temporal_min_: float = 0.0
        self.temporal_max_: float = 5.0
        self.fitted_: bool = False

        # Internal state for exponential moving average (EMA) streaming
        self._last_ema_score: Optional[float] = None

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "CADACompositeScorer":
        """Fits baseline profiler, isolation forest, supervised model, and temporal limits."""
        if 'Pitch' not in X_train.columns or 'KineticEnergy' not in X_train.columns:
            df_feat = self.kinematics.transform(X_train)
        else:
            df_feat = X_train.copy()

        self.feature_cols_ = [c for c in df_feat.columns if c not in ['Class', 'Timestamp'] and pd.api.types.is_numeric_dtype(df_feat[c])]

        # Supervised model training if ground truth provided
        if y_train is not None:
            normal_mask = (y_train == 'NORMAL')
            normal_feat = df_feat[normal_mask].reset_index(drop=True)
            y_binary = (y_train == 'AGGRESSIVE').astype(int)
            self.supervised_model = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                max_depth=9,
                min_samples_leaf=12,
                l2_regularization=0.1,
                random_state=42
            )
            self.supervised_model.fit(df_feat[self.feature_cols_], y_binary)
        else:
            normal_feat = df_feat

        # Fit Sub-models on NORMAL driving baseline
        self.baseline_profiler.feature_cols = self.feature_cols_
        self.baseline_profiler.fit(normal_feat)
        
        self.iso_model.feature_cols = self.feature_cols_
        self.iso_model.fit(normal_feat)

        # Calibrate temporal variation normalization bounds
        acc_chg = np.abs(df_feat['AccMag_Change']) if 'AccMag_Change' in df_feat else np.zeros(len(df_feat))
        gyro_chg = np.abs(df_feat['GyroMag_Change']) if 'GyroMag_Change' in df_feat else np.zeros(len(df_feat))
        acc_range = df_feat.get('AccMag_range_7', pd.Series(0.0, index=df_feat.index))
        temporal_dev = acc_chg + gyro_chg + acc_range

        self.temporal_min_ = float(temporal_dev.min())
        self.temporal_max_ = float(np.percentile(temporal_dev, 95.0) if len(temporal_dev) > 0 else 5.0)
        if self.temporal_max_ <= self.temporal_min_:
            self.temporal_max_ = self.temporal_min_ + 1.0

        self.fitted_ = True
        return self

    def compute_temporal_risk(self, df_or_sample: Union[pd.DataFrame, Dict[str, float]]) -> Union[pd.Series, float]:
        """Calculates normalized temporal jerk & volatility risk score in [0, 100]."""
        if isinstance(df_or_sample, dict):
            acc_chg = abs(float(df_or_sample.get('AccMag_Change', 0.0)))
            gyro_chg = abs(float(df_or_sample.get('GyroMag_Change', 0.0)))
            acc_range = abs(float(df_or_sample.get('AccMag_range_7', 0.0)))
            dev = acc_chg + gyro_chg + acc_range
            norm = (dev - self.temporal_min_) / (self.temporal_max_ - self.temporal_min_ + 1e-8)
            return float(np.clip(norm * 50.0, 0.0, 100.0))

        df = df_or_sample
        acc_chg = np.abs(df['AccMag_Change']) if 'AccMag_Change' in df else np.zeros(len(df))
        gyro_chg = np.abs(df['GyroMag_Change']) if 'GyroMag_Change' in df else np.zeros(len(df))
        acc_range = df.get('AccMag_range_7', pd.Series(0.0, index=df.index))
        dev = acc_chg + gyro_chg + acc_range
        norm = (dev - self.temporal_min_) / (self.temporal_max_ - self.temporal_min_ + 1e-8)
        return np.clip(norm * 50.0, 0.0, 100.0)

    def score_sample(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """Scores a single streaming telemetry observation statefully."""
        if not self.fitted_:
            raise RuntimeError("CADACompositeScorer must be fitted before scoring.")

        enriched_sample = self.kinematics.transform_sample(sample)

        iso_r = float(self.iso_model.score(enriched_sample))
        stat_r = float(self.baseline_profiler.score(enriched_sample))
        temp_r = float(self.compute_temporal_risk(enriched_sample))

        if self.supervised_model is not None:
            feat_vec = pd.DataFrame([[float(enriched_sample.get(c, 0.0)) for c in self.feature_cols_]], columns=self.feature_cols_)
            sup_prob = float(self.supervised_model.predict_proba(feat_vec)[0, 1])
            sup_r = sup_prob * 100.0
            w_sup, w_iso, w_stat, w_temp = (
                self.config.weight_supervised,
                self.config.weight_iso,
                self.config.weight_stat,
                self.config.weight_temporal
            )
            raw_cada_score = (w_sup * sup_r + w_iso * iso_r + w_stat * stat_r + w_temp * temp_r) / (w_sup + w_iso + w_stat + w_temp)
        else:
            w_iso, w_stat, w_temp = self.config.weight_iso, self.config.weight_stat, self.config.weight_temporal
            raw_cada_score = (w_iso * iso_r + w_stat * stat_r + w_temp * temp_r) / (w_iso + w_stat + w_temp)

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
            "AccMag": round(float(enriched_sample.get('AccMag', 0.0)), 4),
            "GyroMag": round(float(enriched_sample.get('GyroMag', 0.0)), 4),
            "AccMag_Change": round(float(enriched_sample.get('AccMag_Change', 0.0)), 4),
            "GyroMag_Change": round(float(enriched_sample.get('GyroMag_Change', 0.0)), 4)
        }

    def score_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scores a batch DataFrame of telemetry data."""
        if not self.fitted_:
            raise RuntimeError("CADACompositeScorer must be fitted before scoring.")

        if 'Pitch' not in X.columns or 'KineticEnergy' not in X.columns:
            df = self.kinematics.transform(X)
        else:
            df = X.copy()

        iso_r = self.iso_model.score(df)
        stat_r = self.baseline_profiler.score(df)
        temp_r = self.compute_temporal_risk(df)

        if self.supervised_model is not None:
            feat_df = df[self.feature_cols_]
            sup_probs = self.supervised_model.predict_proba(feat_df)[:, 1]
            sup_r = sup_probs * 100.0
            w_sup, w_iso, w_stat, w_temp = (
                self.config.weight_supervised,
                self.config.weight_iso,
                self.config.weight_stat,
                self.config.weight_temporal
            )
            cada_score = (w_sup * sup_r + w_iso * iso_r + w_stat * stat_r + w_temp * temp_r) / (w_sup + w_iso + w_stat + w_temp)
        else:
            w_iso, w_stat, w_temp = self.config.weight_iso, self.config.weight_stat, self.config.weight_temporal
            cada_score = (w_iso * iso_r + w_stat * stat_r + w_temp * temp_r) / (w_iso + w_stat + w_temp)

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

