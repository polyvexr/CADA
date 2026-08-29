"""
Integration tests for CADA FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from api.main import app
from src.features.kinematics import KinematicFeatureExtractor
from src.scoring.cada_scorer import CADACompositeScorer
from api.routes import set_scorer


@pytest.fixture(autouse=True)
def init_test_scorer():
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


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_score_single_reading(client):
    payload = {
        "AccX": 0.5,
        "AccY": -0.2,
        "AccZ": 0.1,
        "GyroX": 0.02,
        "GyroY": -0.01,
        "GyroZ": 0.05,
        "Timestamp": 1000
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cada_score" in data
    assert "risk_tier" in data
    assert "components" in data
    assert "kinematics" in data
    assert data["timestamp"] == 1000


def test_stream_and_reset(client):
    reading1 = {"AccX": 0.0, "AccY": 0.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0}
    reading2 = {"AccX": 3.0, "AccY": 4.0, "AccZ": 0.0, "GyroX": 0.1, "GyroY": 0.1, "GyroZ": 0.1}

    res1 = client.post("/api/v1/stream", json=reading1)
    assert res1.status_code == 200

    res2 = client.post("/api/v1/stream", json=reading2)
    assert res2.status_code == 200
    assert res2.json()["kinematics"]["acc_mag_change"] == pytest.approx(5.0, rel=1e-2)

    reset_res = client.post("/api/v1/reset_stream")
    assert reset_res.status_code == 200


def test_batch_score(client):
    payload = {
        "readings": [
            {"AccX": 0.0, "AccY": 0.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0, "Timestamp": 1},
            {"AccX": 1.0, "AccY": 1.0, "AccZ": 1.0, "GyroX": 0.1, "GyroY": 0.1, "GyroZ": 0.1, "Timestamp": 2}
        ]
    }
    response = client.post("/api/v1/batch_score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert len(data["results"]) == 2
    assert "tier_counts" in data

