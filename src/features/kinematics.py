"""
Unified Kinematic & Temporal Feature Extraction Pipeline for CADA.
Vectorized multi-scale dynamics, physical energetics, and streaming state buffer.
"""

from typing import Optional, List, Dict, Any, Tuple
import collections
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_SENSOR_COLS


class KinematicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    High-performance feature extractor for driving motion telemetry.
    Extracts 3D/2D spatial kinematics, orientation angles, cross-axis energy,
    jerk derivatives, and multi-scale temporal rolling dynamics.
    """

    def __init__(
        self,
        windows: Tuple[int, ...] = (3, 7, 15, 25, 40),
        drop_initial_na: bool = False,
        fill_initial_value: float = 0.0
    ):
        self.windows = tuple(windows)
        self.max_window = max(self.windows) if self.windows else 40
        self.drop_initial_na = drop_initial_na
        self.fill_initial_value = fill_initial_value
        self._history: collections.deque = collections.deque(maxlen=self.max_window)
        self._last_acc_mag: Optional[float] = None
        self._last_gyro_mag: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KinematicFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorized batch DataFrame transformation without memory fragmentation."""
        ax = X['AccX'].to_numpy(dtype=float)
        ay = X['AccY'].to_numpy(dtype=float)
        az = X['AccZ'].to_numpy(dtype=float)
        gx = X['GyroX'].to_numpy(dtype=float)
        gy = X['GyroY'].to_numpy(dtype=float)
        gz = X['GyroZ'].to_numpy(dtype=float)

        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        acc_xy = np.sqrt(ax**2 + ay**2)
        gyro_xy = np.sqrt(gx**2 + gy**2)

        feat_dict: Dict[str, Any] = {
            'AccX': ax, 'AccY': ay, 'AccZ': az,
            'GyroX': gx, 'GyroY': gy, 'GyroZ': gz,
            'AccMag': acc_mag, 'GyroMag': gyro_mag,
            'AccXY': acc_xy, 'GyroXY': gyro_xy,
            'Pitch': np.arctan2(ay, np.sqrt(ax**2 + az**2)),
            'Roll': np.arctan2(-ax, az),
            'KineticEnergy': 0.5 * (acc_mag**2),
            'RotationalEnergy': 0.5 * (gyro_mag**2),
            'TotalPower': acc_mag * gyro_mag,
            'AccX_AccY': ax * ay,
            'AccY_AccZ': ay * az,
            'AccMag_GyroZ': acc_mag * gz,
        }

        # 1st & 2nd Order Differences (Jerk and Angular Acceleration)
        for name, arr in [('AccMag', acc_mag), ('GyroMag', gyro_mag), ('AccX', ax), ('AccY', ay), ('AccZ', az), ('GyroZ', gz)]:
            d1 = np.insert(np.diff(arr), 0, self.fill_initial_value)
            d2 = np.insert(np.diff(d1), 0, self.fill_initial_value)
            feat_dict[f'{name}_diff1'] = d1
            feat_dict[f'{name}_diff2'] = d2
            if name == 'AccMag':
                feat_dict['AccMag_Change'] = d1
            elif name == 'GyroMag':
                feat_dict['GyroMag_Change'] = d1

        # Multi-scale rolling window dynamics
        signals = [
            ('AccMag', acc_mag), ('GyroMag', gyro_mag),
            ('AccX', ax), ('AccY', ay), ('AccZ', az),
            ('GyroZ', gz), ('AccXY', acc_xy)
        ]

        for name, arr in signals:
            s = pd.Series(arr)
            for w in self.windows:
                roll = s.rolling(window=w, min_periods=1)
                r_mean = roll.mean().to_numpy()
                r_std = roll.std().fillna(0.0).to_numpy()
                r_max = roll.max().to_numpy()
                r_min = roll.min().to_numpy()
                feat_dict[f'{name}_mean_{w}'] = r_mean
                feat_dict[f'{name}_std_{w}'] = r_std
                feat_dict[f'{name}_range_{w}'] = r_max - r_min
                feat_dict[f'{name}_energy_{w}'] = (s**2).rolling(window=w, min_periods=1).mean().to_numpy()
                feat_dict[f'{name}_ewm_{w}'] = s.ewm(span=w).mean().to_numpy()
                feat_dict[f'{name}_dev_{w}'] = arr - r_mean

        out_df = pd.DataFrame(feat_dict, index=X.index)
        for col in ['Class', 'Timestamp']:
            if col in X.columns:
                out_df[col] = X[col]
        return out_df

    def transform_sample(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Stateful single-sample extraction with rolling temporal memory for streaming."""
        ax = float(sample.get('AccX', 0.0))
        ay = float(sample.get('AccY', 0.0))
        az = float(sample.get('AccZ', 0.0))
        gx = float(sample.get('GyroX', 0.0))
        gy = float(sample.get('GyroY', 0.0))
        gz = float(sample.get('GyroZ', 0.0))

        # Direct instantaneous calculations
        acc_mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))
        acc_change = (acc_mag - self._last_acc_mag) if self._last_acc_mag is not None else self.fill_initial_value
        gyro_change = (gyro_mag - self._last_gyro_mag) if self._last_gyro_mag is not None else self.fill_initial_value
        self._last_acc_mag = acc_mag
        self._last_gyro_mag = gyro_mag

        self._history.append({
            'AccX': ax, 'AccY': ay, 'AccZ': az,
            'GyroX': gx, 'GyroY': gy, 'GyroZ': gz
        })

        hist_df = pd.DataFrame(list(self._history))
        feat_df = self.transform(hist_df)
        last_row = feat_df.iloc[-1].to_dict()

        result = dict(sample)
        result.update(last_row)
        result['AccMag'] = acc_mag
        result['GyroMag'] = gyro_mag
        result['AccMag_Change'] = acc_change
        result['GyroMag_Change'] = gyro_change
        return result

    def reset_state(self):
        """Resets streaming rolling history."""
        self._history.clear()
        self._last_acc_mag = None
        self._last_gyro_mag = None


def create_features(df: pd.DataFrame, drop_initial_na: bool = False) -> pd.DataFrame:
    """Convenience function to extract all kinematic & temporal features."""
    extractor = KinematicFeatureExtractor(drop_initial_na=drop_initial_na)
    return extractor.transform(df)

