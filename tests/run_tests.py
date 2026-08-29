"""
Test runner for CADA using Python standard library unittest.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.kinematics import KinematicFeatureExtractor, create_features
from src.features.windowing import WindowFeatureExtractor
from src.models.baseline_profiler import NormalBaselineProfiler
from src.models.isolation_forest import IsolationForestModel
from src.scoring.cada_scorer import CADACompositeScorer
from src.scoring.risk_tiers import assign_risk_tier, RiskTier


class TestKinematicFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "AccX": [0.0, 1.0, 2.0, 0.0],
            "AccY": [0.0, 0.0, 0.0, 0.0],
            "AccZ": [0.0, 0.0, 0.0, 0.0],
            "GyroX": [0.0, 0.5, 1.0, 0.0],
            "GyroY": [0.0, 0.0, 0.0, 0.0],
            "GyroZ": [0.0, 0.0, 0.0, 0.0],
            "Timestamp": [100, 101, 102, 103]
        })

    def test_batch_transform(self):
        extractor = KinematicFeatureExtractor()
        res = extractor.transform(self.df)
        self.assertIn("AccMag", res.columns)
        self.assertIn("GyroMag", res.columns)
        self.assertIn("AccMag_Change", res.columns)
        self.assertIn("GyroMag_Change", res.columns)
        self.assertAlmostEqual(res.loc[1, "AccMag"], 1.0)
        self.assertAlmostEqual(res.loc[2, "AccMag"], 2.0)
        self.assertAlmostEqual(res.loc[2, "AccMag_Change"], 1.0)

    def test_streaming_transform(self):
        extractor = KinematicFeatureExtractor()
        extractor.reset_state()
        res1 = extractor.transform_sample({"AccX": 1.0, "AccY": 0.0, "AccZ": 0.0})
        self.assertAlmostEqual(res1["AccMag"], 1.0)
        self.assertAlmostEqual(res1["AccMag_Change"], 0.0)

        res2 = extractor.transform_sample({"AccX": 3.0, "AccY": 4.0, "AccZ": 0.0})
        self.assertAlmostEqual(res2["AccMag"], 5.0)
        self.assertAlmostEqual(res2["AccMag_Change"], 4.0)

    def test_window_features(self):
        window_ext = WindowFeatureExtractor(window_size=2, target_cols=["AccX"])
        res = window_ext.transform(self.df)
        self.assertIn("AccX_roll_mean", res.columns)
        self.assertIn("AccX_roll_std", res.columns)
        self.assertIn("AccX_roll_range", res.columns)


class TestScoringAndModels(unittest.TestCase):
    def setUp(self):
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
        self.train_data = create_features(df)

    def test_baseline_profiler(self):
        profiler = NormalBaselineProfiler()
        normal_data = self.train_data[self.train_data["Class"] == "NORMAL"]
        profiler.fit(normal_data)
        self.assertTrue(profiler.fitted_)
        self.assertGreater(profiler.normal_limit_, 0)

        scores = profiler.score(self.train_data)
        self.assertEqual(len(scores), len(self.train_data))
        self.assertTrue((scores >= 0.0).all() and (scores <= 100.0).all())

    def test_isolation_forest(self):
        iso = IsolationForestModel()
        normal_data = self.train_data[self.train_data["Class"] == "NORMAL"]
        iso.fit(normal_data)
        self.assertTrue(iso.fitted_)
        scores = iso.score(self.train_data)
        self.assertEqual(len(scores), len(self.train_data))
        self.assertTrue((scores >= 0.0).all() and (scores <= 100.0).all())

    def test_cada_composite_scorer(self):
        scorer = CADACompositeScorer()
        scorer.fit(self.train_data, y_train=self.train_data["Class"])
        self.assertTrue(scorer.fitted_)

        scored_df = scorer.score_batch(self.train_data)
        self.assertIn("CADA_Score", scored_df.columns)
        self.assertIn("Risk_Tier", scored_df.columns)
        self.assertTrue((scored_df["CADA_Score"] >= 0.0).all() and (scored_df["CADA_Score"] <= 100.0).all())

        sample = {"AccX": 0.1, "AccY": -0.1, "AccZ": 0.0, "GyroX": 0.01, "GyroY": 0.0, "GyroZ": 0.0}
        single = scorer.score_sample(sample)
        self.assertIn("CADA_Score", single)
        self.assertIn("Risk_Tier", single)
        self.assertIn(single["Risk_Tier"], ["NORMAL", "MODERATE", "HIGH", "CRITICAL"])

    def test_risk_tiers(self):
        self.assertEqual(assign_risk_tier(10.0), RiskTier.NORMAL)
        self.assertEqual(assign_risk_tier(30.0), RiskTier.MODERATE)
        self.assertEqual(assign_risk_tier(60.0), RiskTier.HIGH)
        self.assertEqual(assign_risk_tier(85.0), RiskTier.CRITICAL)


if __name__ == "__main__":
    unittest.main()

