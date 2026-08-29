"""
API route tests using TestClient and unittest.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from src.features.kinematics import KinematicFeatureExtractor
from src.scoring.cada_scorer import CADACompositeScorer
from api.routes import set_scorer


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        df_feat = extractor.fit_transform(df)
        scorer = CADACompositeScorer()
        scorer.fit(df_feat, y_train=df_feat["Class"])
        set_scorer(scorer)
        cls.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_score_single(self):
        payload = {
            "AccX": 0.5,
            "AccY": -0.2,
            "AccZ": 0.1,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.05,
            "Timestamp": 12345
        }
        res = self.client.post("/api/v1/score", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cada_score", data)
        self.assertIn("risk_tier", data)
        self.assertIn("components", data)
        self.assertIn("kinematics", data)
        self.assertEqual(data["timestamp"], 12345)

    def test_batch_score(self):
        payload = {
            "readings": [
                {"AccX": 0.0, "AccY": 0.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0},
                {"AccX": 1.0, "AccY": 1.0, "AccZ": 1.0, "GyroX": 0.1, "GyroY": 0.1, "GyroZ": 0.1}
            ]
        }
        res = self.client.post("/api/v1/batch_score", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_samples"], 2)
        self.assertEqual(len(data["results"]), 2)


if __name__ == "__main__":
    unittest.main()

