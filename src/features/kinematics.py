"""
Kinematic feature extraction from multi-axis accelerometer and gyroscope sensor telemetry.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_SENSOR_COLS, KINEMATIC_FEATURE_COLS, FEATURE_COLS


class KinematicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts magnitude and rate-of-change kinematic features:
    - AccMag: Euclidean magnitude of 3-axis acceleration
    - GyroMag: Euclidean magnitude of 3-axis angular velocity
    - AccMag_Change: Instantaneous first-order difference of AccMag (jerk indicator)
    - GyroMag_Change: Instantaneous first-order difference of GyroMag (angular acceleration indicator)
    """

    def __init__(
        self,
        drop_initial_na: bool = False,
        fill_initial_value: float = 0.0
    ):
        self.drop_initial_na = drop_initial_na
        self.fill_initial_value = fill_initial_value
        
        # State tracking for streaming single-sample inference
        self._last_acc_mag: Optional[float] = None
        self._last_gyro_mag: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a batch DataFrame of motion data by adding kinematic features.
        """
        df = X.copy()

        # Compute 3D Magnitudes
        df['AccMag'] = np.sqrt(
            df['AccX']**2 + df['AccY']**2 + df['AccZ']**2
        )
        df['GyroMag'] = np.sqrt(
            df['GyroX']**2 + df['GyroY']**2 + df['GyroZ']**2
        )

        # Compute First-Order Differences (Rate of Change / Jerk)
        df['AccMag_Change'] = df['AccMag'].diff()
        df['GyroMag_Change'] = df['GyroMag'].diff()

        if self.drop_initial_na:
            df = df.dropna(subset=['AccMag_Change', 'GyroMag_Change']).reset_index(drop=True)
        else:
            df['AccMag_Change'] = df['AccMag_Change'].fillna(self.fill_initial_value)
            df['GyroMag_Change'] = df['GyroMag_Change'].fillna(self.fill_initial_value)

        return df

    def transform_sample(self, sample: Dict[str, float]) -> Dict[str, float]:
        """
        Extracts kinematic features for a single streaming telemetry sample statefully.

        Parameters
        ----------
        sample : dict
            Dictionary with keys 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'.

        Returns
        -------
        dict
            Dictionary enriched with 'AccMag', 'GyroMag', 'AccMag_Change', 'GyroMag_Change'.
        """
        acc_x = float(sample.get('AccX', 0.0))
        acc_y = float(sample.get('AccY', 0.0))
        acc_z = float(sample.get('AccZ', 0.0))
        gyro_x = float(sample.get('GyroX', 0.0))
        gyro_y = float(sample.get('GyroY', 0.0))
        gyro_z = float(sample.get('GyroZ', 0.0))

        acc_mag = float(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
        gyro_mag = float(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

        if self._last_acc_mag is None:
            acc_change = self.fill_initial_value
            gyro_change = self.fill_initial_value
        else:
            acc_change = acc_mag - self._last_acc_mag
            gyro_change = gyro_mag - self._last_gyro_mag

        self._last_acc_mag = acc_mag
        self._last_gyro_mag = gyro_mag

        result = dict(sample)
        result.update({
            'AccMag': acc_mag,
            'GyroMag': gyro_mag,
            'AccMag_Change': acc_change,
            'GyroMag_Change': gyro_change
        })
        return result

    def reset_state(self):
        """Resets streaming internal state."""
        self._last_acc_mag = None
        self._last_gyro_mag = None


def create_features(df: pd.DataFrame, drop_initial_na: bool = False) -> pd.DataFrame:
    """Convenience function to extract kinematic features from DataFrame."""
    extractor = KinematicFeatureExtractor(drop_initial_na=drop_initial_na)
    return extractor.transform(df)

