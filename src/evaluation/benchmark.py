"""
Comprehensive benchmark comparing Supervised, Unsupervised, and CADA architectures.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

from src.config import RAW_DATA_DIR, RESULTS_DIR, FEATURE_COLS
from src.data.loader import load_motion_data
from src.data.preprocessor import MotionDataPreprocessor
from src.features.kinematics import KinematicFeatureExtractor
from src.scoring.cada_scorer import CADACompositeScorer
from src.evaluation.metrics import evaluate_anomaly_detection, compute_risk_correlation, compare_models


def run_full_benchmark(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes benchmark comparison across multiple modeling paradigms.
    """
    tr_path = train_path or (RAW_DATA_DIR / "train_motion_data.csv")
    te_path = test_path or (RAW_DATA_DIR / "test_motion_data.csv")
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and process datasets
    df_train = load_motion_data(tr_path, require_target=True)
    df_test = load_motion_data(te_path, require_target=True)

    preproc = MotionDataPreprocessor()
    df_tr_clean = preproc.fit_transform(df_train)
    df_te_clean = preproc.transform(df_test)

    kinematics = KinematicFeatureExtractor()
    df_tr_feat = kinematics.fit_transform(df_tr_clean)
    df_te_feat = kinematics.transform(df_te_clean)

    X_train = df_tr_feat[FEATURE_COLS]
    y_train = df_tr_feat['Class']
    X_test = df_te_feat[FEATURE_COLS]
    y_test = df_te_feat['Class']

    # Binary ground truth: AGGRESSIVE is positive anomaly (1), else 0
    actual_anomaly = (y_test == 'AGGRESSIVE').astype(int)
    results = {}

    # ---------------------------------------------------------
    # Model 1: Supervised Random Forest Classifier
    # ---------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    t0 = time.perf_counter()
    rf_pred_labels = rf.predict(X_test)
    rf_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    rf_anomaly_pred = (rf_pred_labels == 'AGGRESSIVE').astype(int)
    # Probability of AGGRESSIVE class if available
    agg_idx = list(rf.classes_).index('AGGRESSIVE')
    rf_anomaly_scores = rf.predict_proba(X_test)[:, agg_idx]

    results["Supervised_RandomForest"] = {
        **evaluate_anomaly_detection(actual_anomaly, rf_anomaly_pred, rf_anomaly_scores),
        **compute_risk_correlation(y_test, pd.Series(rf_anomaly_scores)),
        "latency_ms_per_sample": rf_latency_ms
    }

    # ---------------------------------------------------------
    # Model 2: Unsupervised Isolation Forest
    # ---------------------------------------------------------
    normal_train = X_train[y_train == 'NORMAL']
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso.fit(normal_train)

    t0 = time.perf_counter()
    iso_decision = iso.decision_function(X_test)
    iso_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    # Invert decision function so higher score = more anomalous
    iso_scores = -iso_decision
    iso_pred = (iso.predict(X_test) == -1).astype(int)

    results["Unsupervised_IsolationForest"] = {
        **evaluate_anomaly_detection(actual_anomaly, iso_pred, iso_scores),
        **compute_risk_correlation(y_test, pd.Series(iso_scores)),
        "latency_ms_per_sample": iso_latency_ms
    }

    # ---------------------------------------------------------
    # Model 3: Unsupervised One-Class SVM
    # ---------------------------------------------------------
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(normal_train)

    t0 = time.perf_counter()
    ocsvm_decision = ocsvm.decision_function(X_test)
    ocsvm_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    ocsvm_scores = -ocsvm_decision
    ocsvm_pred = (ocsvm.predict(X_test) == -1).astype(int)

    results["Unsupervised_OneClassSVM"] = {
        **evaluate_anomaly_detection(actual_anomaly, ocsvm_pred, ocsvm_scores),
        **compute_risk_correlation(y_test, pd.Series(ocsvm_scores)),
        "latency_ms_per_sample": ocsvm_latency_ms
    }

    # ---------------------------------------------------------
    # Model 4: CADA Composite Continuous Architecture
    # ---------------------------------------------------------
    cada = CADACompositeScorer()
    cada.fit(df_tr_feat, y_train=y_train)

    t0 = time.perf_counter()
    df_te_scored = cada.score_batch(df_te_feat)
    cada_latency_ms = ((time.perf_counter() - t0) / len(df_te_feat)) * 1000.0

    cada_scores = df_te_scored['CADA_Score']
    # Threshold at 50 (High & Critical Risk Tiers)
    cada_pred = (cada_scores >= 50.0).astype(int)

    results["CADA_Composite_Architecture"] = {
        **evaluate_anomaly_detection(actual_anomaly, cada_pred, cada_scores),
        **compute_risk_correlation(y_test, cada_scores),
        "latency_ms_per_sample": cada_latency_ms
    }

    # Format into DataFrame
    df_comparison = compare_models(results)
    print("\n================== CADA BENCHMARK RESULTS ==================")
    print(df_comparison.to_string())

    # Save outputs
    report_file = out_dir / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    df_comparison.to_csv(out_dir / "benchmark_comparison.csv")
    print(f"\nSaved benchmark report to: {report_file}")

    return results


if __name__ == "__main__":
    run_full_benchmark()

