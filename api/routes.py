"""
FastAPI route handlers for CADA risk scoring and telemetry streaming.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd

from api.schemas import (
    MotionReading,
    BatchMotionRequest,
    CADAScoreResponse,
    BatchCADAScoreResponse,
    RiskComponentBreakdown,
    KinematicMetrics,
    HealthResponse
)
from src.scoring.cada_scorer import CADACompositeScorer
from src.scoring.risk_tiers import RiskTier

router = APIRouter()

# Global reference to scorer
_scorer_instance: Optional[CADACompositeScorer] = None


def load_or_init_scorer() -> CADACompositeScorer:
    """Safely loads bundle from disk or falls back to in-memory model training."""
    global _scorer_instance
    from src.config import MODELS_DIR, RAW_DATA_DIR
    
    model_bundle_path = MODELS_DIR / "cada_model_bundle.joblib"
    if model_bundle_path.exists():
        try:
            _scorer_instance = CADACompositeScorer.load(model_bundle_path)
            return _scorer_instance
        except Exception as e:
            print(f"Warning: Failed to load pre-trained bundle ({e}). Initializing in-memory...")
    
    # In-memory training fallback from raw data if available
    try:
        from src.data.loader import load_motion_data
        from src.data.preprocessor import MotionDataPreprocessor
        from src.features.kinematics import KinematicFeatureExtractor
        
        tr_path = RAW_DATA_DIR / "train_motion_data.csv"
        if tr_path.exists():
            df_train = load_motion_data(tr_path, require_target=True)
            preprocessor = MotionDataPreprocessor()
            df_clean = preprocessor.fit_transform(df_train)
            kinematics = KinematicFeatureExtractor()
            df_feat = kinematics.transform(df_clean)
            scorer = CADACompositeScorer()
            scorer.fit(df_feat, y_train=df_feat['Class'])
            _scorer_instance = scorer
            return _scorer_instance
    except Exception as e:
        print(f"In-memory training note: {e}. Falling back to default heuristics...")

    # Zero-dependency heuristic fallback for ultra-lightweight serverless execution
    scorer = CADACompositeScorer()
    scorer.init_default_heuristics()
    _scorer_instance = scorer
    return _scorer_instance



def get_scorer() -> CADACompositeScorer:
    """Dependency provider for CADA scorer with automatic lazy-loading."""
    global _scorer_instance
    if _scorer_instance is None or not _scorer_instance.fitted_:
        _scorer_instance = load_or_init_scorer()
    return _scorer_instance


def set_scorer(scorer: CADACompositeScorer):
    """Setter for global scorer instance."""
    global _scorer_instance
    _scorer_instance = scorer


@router.get("/", tags=["General"])
def root():
    """Root endpoint providing service metadata and documentation links."""
    return {
        "title": "CADA - Continuous Driving Anomaly Detection API",
        "status": "operational",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "health_url": "/health",
        "version": "1.0.0"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns service operational health and model readiness status."""
    try:
        scorer = get_scorer()
        is_ready = scorer is not None and scorer.fitted_
    except Exception:
        is_ready = False

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        version="1.0.0"
    )


@router.post("/api/v1/score", response_model=CADAScoreResponse, tags=["Scoring"])
def score_single_reading(
    reading: MotionReading,
    scorer: CADACompositeScorer = Depends(get_scorer)
):
    """
    Scores a single instantaneous IMU telemetry reading.
    """
    sample_dict = reading.model_dump()
    result = scorer.score_sample(sample_dict)

    return CADAScoreResponse(
        cada_score=result["CADA_Score"],
        risk_tier=RiskTier(result["Risk_Tier"]),
        components=RiskComponentBreakdown(
            iso_risk=result["Iso_Risk"],
            stat_risk=result["Stat_Risk"],
            temporal_risk=result["Temporal_Risk"]
        ),
        kinematics=KinematicMetrics(
            acc_mag=result["AccMag"],
            gyro_mag=result["GyroMag"],
            acc_mag_change=result["AccMag_Change"],
            gyro_mag_change=result["GyroMag_Change"]
        ),
        timestamp=reading.Timestamp
    )


@router.post("/api/v1/stream", response_model=CADAScoreResponse, tags=["Streaming"])
def stream_reading(
    reading: MotionReading,
    scorer: CADACompositeScorer = Depends(get_scorer)
):
    """
    Stateful streaming endpoint that computes jerk/differences sequentially from consecutive readings.
    """
    return score_single_reading(reading, scorer)


@router.post("/api/v1/reset_stream", tags=["Streaming"])
def reset_stream(scorer: CADACompositeScorer = Depends(get_scorer)):
    """Resets the internal streaming kinematic difference state."""
    scorer.reset_stream_state()
    return {"message": "Streaming state reset successfully."}


@router.post("/api/v1/batch_score", response_model=BatchCADAScoreResponse, tags=["Scoring"])
def score_batch_readings(
    payload: BatchMotionRequest,
    scorer: CADACompositeScorer = Depends(get_scorer)
):
    """
    Scores a sequential batch of IMU telemetry readings and returns aggregated statistics.
    """
    if not payload.readings:
        raise HTTPException(status_code=400, detail="Empty readings list provided.")

    data_dicts = [r.model_dump() for r in payload.readings]
    df = pd.DataFrame(data_dicts)

    df_scored = scorer.score_batch(df)

    results: List[CADAScoreResponse] = []
    for _, row in df_scored.iterrows():
        results.append(CADAScoreResponse(
            cada_score=round(float(row["CADA_Score"]), 2),
            risk_tier=RiskTier(row["Risk_Tier"]),
            components=RiskComponentBreakdown(
                iso_risk=round(float(row["Iso_Risk"]), 2),
                stat_risk=round(float(row["Stat_Risk"]), 2),
                temporal_risk=round(float(row["Temporal_Risk"]), 2)
            ),
            kinematics=KinematicMetrics(
                acc_mag=round(float(row["AccMag"]), 4),
                gyro_mag=round(float(row["GyroMag"]), 4),
                acc_mag_change=round(float(row["AccMag_Change"]), 4),
                gyro_mag_change=round(float(row["GyroMag_Change"]), 4)
            ),
            timestamp=int(row["Timestamp"]) if "Timestamp" in row and pd.notnull(row["Timestamp"]) else None
        ))

    tier_counts = df_scored["Risk_Tier"].value_counts().to_dict()

    return BatchCADAScoreResponse(
        total_samples=len(df_scored),
        mean_cada_score=round(float(df_scored["CADA_Score"].mean()), 2),
        max_cada_score=round(float(df_scored["CADA_Score"].max()), 2),
        tier_counts=tier_counts,
        results=results
    )
