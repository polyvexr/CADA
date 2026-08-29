"""
Risk Tier discretization and alerting policies for CADA continuous scores.
"""

from enum import Enum
from typing import Union, List
import pandas as pd
import numpy as np

from src.config import DEFAULT_SCORER_CONFIG, CADAScorerConfig


class RiskTier(str, Enum):
    NORMAL = "NORMAL"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def assign_risk_tier(
    score: float,
    config: CADAScorerConfig = DEFAULT_SCORER_CONFIG
) -> RiskTier:
    """
    Maps a continuous risk score (0 - 100) into an operational Risk Tier.

    Parameters
    ----------
    score : float
        Continuous CADA risk score in range [0, 100].
    config : CADAScorerConfig
        Scorer configuration containing threshold cutoffs.

    Returns
    -------
    RiskTier
        Assigned risk tier.
    """
    if score < config.tier_moderate_threshold:
        return RiskTier.NORMAL
    elif score < config.tier_high_threshold:
        return RiskTier.MODERATE
    elif score < config.tier_critical_threshold:
        return RiskTier.HIGH
    else:
        return RiskTier.CRITICAL


def assign_risk_tiers_batch(
    scores: Union[pd.Series, np.ndarray],
    config: CADAScorerConfig = DEFAULT_SCORER_CONFIG
) -> pd.Series:
    """
    Vectorized risk tier assignment for batch scores.
    """
    conditions = [
        scores < config.tier_moderate_threshold,
        (scores >= config.tier_moderate_threshold) & (scores < config.tier_high_threshold),
        (scores >= config.tier_high_threshold) & (scores < config.tier_critical_threshold),
        scores >= config.tier_critical_threshold
    ]
    choices = [
        RiskTier.NORMAL.value,
        RiskTier.MODERATE.value,
        RiskTier.HIGH.value,
        RiskTier.CRITICAL.value
    ]
    assigned = np.select(conditions, choices, default=RiskTier.NORMAL.value)
    return pd.Series(assigned, index=getattr(scores, 'index', None))

