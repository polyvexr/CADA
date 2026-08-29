"""
Unit tests for kinematic and window feature extractors.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.kinematics import KinematicFeatureExtractor
from src.features.windowing import WindowFeatureExtractor


@pytest.fixture
def sample_motion_df():
    return pd.DataFrame({
        "AccX": [0.0, 1.0, 2.0, 0.0],
        "AccY": [0.0, 0.0, 0.0, 0.0],
        "AccZ": [0.0, 0.0, 0.0, 0.0],
        "GyroX": [0.0, 0.5, 1.0, 0.0],
        "GyroY": [0.0, 0.0, 0.0, 0.0],
        "GyroZ": [0.0, 0.0, 0.0, 0.0],
        "Timestamp": [100, 101, 102, 103]
    })


def test_kinematic_batch_transform(sample_motion_df):
    extractor = KinematicFeatureExtractor()
    transformed = extractor.transform(sample_motion_df)

    assert "AccMag" in transformed.columns
    assert "GyroMag" in transformed.columns
    assert "AccMag_Change" in transformed.columns
    assert "GyroMag_Change" in transformed.columns

    # Test math
    assert transformed.loc[1, "AccMag"] == pytest.approx(1.0)
    assert transformed.loc[2, "AccMag"] == pytest.approx(2.0)
    assert transformed.loc[2, "AccMag_Change"] == pytest.approx(1.0)


def test_kinematic_streaming_transform():
    extractor = KinematicFeatureExtractor()
    extractor.reset_state()

    sample1 = {"AccX": 1.0, "AccY": 0.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0}
    res1 = extractor.transform_sample(sample1)
    assert res1["AccMag"] == pytest.approx(1.0)
    assert res1["AccMag_Change"] == pytest.approx(0.0)

    sample2 = {"AccX": 3.0, "AccY": 4.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0}
    res2 = extractor.transform_sample(sample2)
    assert res2["AccMag"] == pytest.approx(5.0)
    assert res2["AccMag_Change"] == pytest.approx(4.0)


def test_window_feature_extractor(sample_motion_df):
    extractor = WindowFeatureExtractor(window_size=2, target_cols=["AccX"])
    transformed = extractor.transform(sample_motion_df)

    assert "AccX_roll_mean" in transformed.columns
    assert "AccX_roll_std" in transformed.columns
    assert "AccX_roll_range" in transformed.columns
    assert "AccX_roll_energy" in transformed.columns

