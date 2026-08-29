"""
Evaluation metrics for continuous driving anomaly detection.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


def evaluate_anomaly_detection(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_scores: Optional[Union[pd.Series, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Computes precision, recall, F1, accuracy, and optional ROC-AUC & PR-AUC.
    """
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_scores))
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    return metrics


def compute_risk_correlation(
    true_classes: pd.Series,
    continuous_scores: pd.Series
) -> Dict[str, float]:
    """
    Computes Spearman rank correlation between ordinal risk severity
    (SLOW=0, NORMAL=1, AGGRESSIVE=2) and model continuous scores.
    """
    severity_map = {"SLOW": 0, "NORMAL": 1, "AGGRESSIVE": 2}
    y_ordinal = true_classes.map(severity_map).fillna(1)

    corr, p_value = spearmanr(y_ordinal, continuous_scores)
    return {
        "spearman_correlation": float(corr),
        "p_value": float(p_value)
    }


def compare_models(
    model_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Formats model results dictionary into a clean comparison DataFrame."""
    return pd.DataFrame.from_dict(model_results, orient="index")
