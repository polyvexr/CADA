"""
Comprehensive benchmark comparing Supervised, Unsupervised, and CADA architectures.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

from src.config import RAW_DATA_DIR, RESULTS_DIR
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
    Executes benchmark comparison across multiple modeling paradigms on the enhanced feature space.
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
    df_tr_feat = kinematics.transform(df_tr_clean)
    df_te_feat = kinematics.transform(df_te_clean)

    feature_cols = [c for c in df_tr_feat.columns if c not in ['Class', 'Timestamp'] and pd.api.types.is_numeric_dtype(df_tr_feat[c])]
    X_train = df_tr_feat[feature_cols]
    y_train = df_tr_feat['Class']
    X_test = df_te_feat[feature_cols]
    y_test = df_te_feat['Class']

    actual_anomaly = (y_test == 'AGGRESSIVE').astype(int)
    y_train_binary = (y_train == 'AGGRESSIVE').astype(int)
    normal_train = X_train[y_train == 'NORMAL']
    results = {}

    # ---------------------------------------------------------
    # Model 1: Supervised HistGradientBoosting Classifier
    # ---------------------------------------------------------
    hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=9, min_samples_leaf=12, random_state=42)
    hgb.fit(X_train, y_train_binary)

    t0 = time.perf_counter()
    hgb_probs = hgb.predict_proba(X_test)[:, 1]
    hgb_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0
    hgb_pred = (hgb_probs >= 0.5).astype(int)

    results["Supervised_HistGradientBoosting"] = {
        **evaluate_anomaly_detection(actual_anomaly, hgb_pred, hgb_probs),
        **compute_risk_correlation(y_test, pd.Series(hgb_probs)),
        "latency_ms_per_sample": hgb_latency_ms
    }

    # ---------------------------------------------------------
    # Model 2: Supervised Random Forest Classifier
    # ---------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_binary)

    t0 = time.perf_counter()
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0
    rf_pred = (rf_probs >= 0.5).astype(int)

    results["Supervised_RandomForest"] = {
        **evaluate_anomaly_detection(actual_anomaly, rf_pred, rf_probs),
        **compute_risk_correlation(y_test, pd.Series(rf_probs)),
        "latency_ms_per_sample": rf_latency_ms
    }

    # ---------------------------------------------------------
    # Model 3: Unsupervised Isolation Forest
    # ---------------------------------------------------------
    iso = IsolationForest(n_estimators=150, contamination=0.10, random_state=42, n_jobs=-1)
    iso.fit(normal_train)

    t0 = time.perf_counter()
    iso_decision = -iso.decision_function(X_test)
    iso_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0
    iso_norm = np.clip((iso_decision - iso_decision.min()) / (iso_decision.max() - iso_decision.min() + 1e-8), 0, 1)
    iso_pred = (iso.predict(X_test) == -1).astype(int)

    results["Unsupervised_IsolationForest"] = {
        **evaluate_anomaly_detection(actual_anomaly, iso_pred, iso_norm),
        **compute_risk_correlation(y_test, pd.Series(iso_norm)),
        "latency_ms_per_sample": iso_latency_ms
    }

    # ---------------------------------------------------------
    # Model 4: Unsupervised One-Class SVM
    # ---------------------------------------------------------
    scaler = RobustScaler().fit(normal_train)
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(scaler.transform(normal_train))

    t0 = time.perf_counter()
    ocsvm_decision = -ocsvm.decision_function(scaler.transform(X_test))
    ocsvm_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0
    ocsvm_norm = np.clip((ocsvm_decision - ocsvm_decision.min()) / (ocsvm_decision.max() - ocsvm_decision.min() + 1e-8), 0, 1)
    ocsvm_pred = (ocsvm.predict(scaler.transform(X_test)) == -1).astype(int)

    results["Unsupervised_OneClassSVM"] = {
        **evaluate_anomaly_detection(actual_anomaly, ocsvm_pred, ocsvm_norm),
        **compute_risk_correlation(y_test, pd.Series(ocsvm_norm)),
        "latency_ms_per_sample": ocsvm_latency_ms
    }

    # ---------------------------------------------------------
    # Model 5: CADA Composite Continuous Architecture
    # ---------------------------------------------------------
    cada = CADACompositeScorer()
    cada.fit(df_tr_feat, y_train=y_train)

    t0 = time.perf_counter()
    df_te_scored = cada.score_batch(df_te_feat)
    cada_latency_ms = ((time.perf_counter() - t0) / len(df_te_feat)) * 1000.0

    cada_scores = df_te_scored['CADA_Score']
    cada_pred = (cada_scores >= 50.0).astype(int)

    results["CADA_Composite_Architecture"] = {
        **evaluate_anomaly_detection(actual_anomaly, cada_pred, cada_scores / 100.0),
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

