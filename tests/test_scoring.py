"""
Unit tests for CADA models, scoring formulas, and risk tiers.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.kinematics import KinematicFeatureExtractor
from src.models.baseline_profiler import NormalBaselineProfiler
from src.models.isolation_forest import IsolationForestModel
from src.scoring.cada_scorer import CADACompositeScorer
from src.scoring.risk_tiers import assign_risk_tier, RiskTier


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "AccX": np.random.normal(0, 0.5, n),
        "AccY": np.random.normal(0, 0.5, n),
        "AccZ": np.random.normal(0, 0.5, n),
        "GyroX": np.random.normal(0, 0.05, n),
        "GyroY": np.random.normal(0, 0.05, n),
        "GyroZ": np.random.normal(0, 0.05, n),
        "Class": ["NORMAL"] * 80 + ["AGGRESSIVE"] * 20
    })
    extractor = KinematicFeatureExtractor()
    return extractor.fit_transform(df)


def test_baseline_profiler(dummy_train_data):
    profiler = NormalBaselineProfiler()
    normal_df = dummy_train_data[dummy_train_data["Class"] == "NORMAL"]
    profiler.fit(normal_df)

    assert profiler.fitted_
    assert profiler.normal_limit_ > 0

    scores = profiler.score(dummy_train_data)
    assert len(scores) == len(dummy_train_data)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()


def test_isolation_forest_model(dummy_train_data):
    model = IsolationForestModel()
    normal_df = dummy_train_data[dummy_train_data["Class"] == "NORMAL"]
    model.fit(normal_df)

    assert model.fitted_
    scores = model.score(dummy_train_data)
    assert len(scores) == len(dummy_train_data)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()


def test_cada_composite_scorer(dummy_train_data):
    scorer = CADACompositeScorer()
    scorer.fit(dummy_train_data, y_train=dummy_train_data["Class"])

    assert scorer.fitted_

    # Batch test
    scored_df = scorer.score_batch(dummy_train_data)
    assert "CADA_Score" in scored_df.columns
    assert "Risk_Tier" in scored_df.columns
    assert (scored_df["CADA_Score"] >= 0.0).all() and (scored_df["CADA_Score"] <= 100.0).all()

    # Single-sample test
    sample = {"AccX": 0.1, "AccY": -0.1, "AccZ": 0.0, "GyroX": 0.01, "GyroY": 0.0, "GyroZ": 0.0}
    single_res = scorer.score_sample(sample)
    assert "CADA_Score" in single_res
    assert "Risk_Tier" in single_res
    assert single_res["Risk_Tier"] in ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]


def test_risk_tier_thresholds():
    assert assign_risk_tier(10.0) == RiskTier.NORMAL
    assert assign_risk_tier(30.0) == RiskTier.MODERATE
    assert assign_risk_tier(60.0) == RiskTier.HIGH
    assert assign_risk_tier(85.0) == RiskTier.CRITICAL

