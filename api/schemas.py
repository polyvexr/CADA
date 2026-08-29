"""
Pydantic schemas for CADA REST and streaming API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.scoring.risk_tiers import RiskTier


class MotionReading(BaseModel):
    """Single-timestamp inertial measurement unit (IMU) telemetry sample."""
    AccX: float = Field(..., description="Acceleration along X-axis (m/s^2 or g)")
    AccY: float = Field(..., description="Acceleration along Y-axis (m/s^2 or g)")
    AccZ: float = Field(..., description="Acceleration along Z-axis (m/s^2 or g)")
    GyroX: float = Field(..., description="Angular velocity around X-axis (rad/s or deg/s)")
    GyroY: float = Field(..., description="Angular velocity around Y-axis (rad/s or deg/s)")
    GyroZ: float = Field(..., description="Angular velocity around Z-axis (rad/s or deg/s)")
    Timestamp: Optional[int] = Field(None, description="Optional millisecond timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "AccX": -0.5946,
                "AccY": -0.1224,
                "AccZ": 0.2205,
                "GyroX": -0.0193,
                "GyroY": -0.0293,
                "GyroZ": 0.0878,
                "Timestamp": 3581630
            }
        }
    }


class BatchMotionRequest(BaseModel):
    """Batch of sequential motion telemetry readings."""
    readings: List[MotionReading]


class RiskComponentBreakdown(BaseModel):
    """Breakdown of sub-risk scores (0 - 100)."""
    iso_risk: float = Field(..., description="Isolation Forest anomaly risk score")
    stat_risk: float = Field(..., description="Statistical Z-score baseline deviation score")
    temporal_risk: float = Field(..., description="Kinematic temporal jerk & variation score")


class KinematicMetrics(BaseModel):
    """Derived kinematic feature values."""
    acc_mag: float = Field(..., description="3D Euclidean acceleration magnitude")
    gyro_mag: float = Field(..., description="3D Euclidean angular velocity magnitude")
    acc_mag_change: float = Field(..., description="Instantaneous change in acceleration magnitude")
    gyro_mag_change: float = Field(..., description="Instantaneous change in angular velocity magnitude")


class CADAScoreResponse(BaseModel):
    """Instantaneous CADA driving risk assessment response."""
    cada_score: float = Field(..., description="Composite CADA driving risk score (0 - 100)")
    risk_tier: RiskTier = Field(..., description="Operational risk tier (NORMAL, MODERATE, HIGH, CRITICAL)")
    components: RiskComponentBreakdown
    kinematics: KinematicMetrics
    timestamp: Optional[int] = None


class BatchCADAScoreResponse(BaseModel):
    """Batch assessment response."""
    total_samples: int
    mean_cada_score: float
    max_cada_score: float
    tier_counts: Dict[str, int]
    results: List[CADAScoreResponse]


class HealthResponse(BaseModel):
    """Service health and model status schema."""
    status: str
    model_loaded: bool
    version: str

