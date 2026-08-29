"""
Feature engineering modules for CADA motion telemetry.
"""

from src.features.kinematics import KinematicFeatureExtractor, create_features
from src.features.windowing import WindowFeatureExtractor
from src.config import RAW_SENSOR_COLS as SENSOR_COLS, FEATURE_COLS

__all__ = [
    "KinematicFeatureExtractor",
    "WindowFeatureExtractor",
    "create_features",
    "SENSOR_COLS",
    "FEATURE_COLS"
]

