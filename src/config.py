"""
Configuration module for CADA (Continuous Anomaly Detection Architecture).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

# Root Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Sensor and Feature Definitions
RAW_SENSOR_COLS: List[str] = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ"
]

KINEMATIC_FEATURE_COLS: List[str] = [
    "AccMag", "GyroMag",
    "AccMag_Change", "GyroMag_Change"
]

FEATURE_COLS: List[str] = RAW_SENSOR_COLS + KINEMATIC_FEATURE_COLS

TIMESTAMP_COL: str = "Timestamp"
TARGET_COL: str = "Class"


@dataclass
class CADAScorerConfig:
    """Configuration for CADA Risk Scoring Engine."""
    # Component weights in composite score (must sum to 1.0 ideally)
    weight_iso: float = 1.0 / 3.0
    weight_stat: float = 1.0 / 3.0
    weight_temporal: float = 1.0 / 3.0
    
    # Statistical baseline percentile cutoff (e.g. 95th percentile of normal driving z-scores)
    normal_percentile_cutoff: float = 95.0
    
    # Risk Tier Thresholds
    tier_moderate_threshold: float = 25.0
    tier_high_threshold: float = 50.0
    tier_critical_threshold: float = 75.0
    
    # Smoothing / EMA parameters for continuous streaming
    enable_ema: bool = False
    ema_alpha: float = 0.3


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest Model."""
    n_estimators: int = 100
    contamination: str = "auto"
    random_state: int = 42
    n_jobs: int = -1


# Default Global Configuration Instance
DEFAULT_SCORER_CONFIG = CADAScorerConfig()
DEFAULT_IFOREST_CONFIG = IsolationForestConfig()

