"""
End-to-end model training, calibration, and artifact generation script for CADA.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.config import RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.data.loader import load_motion_data
from src.data.preprocessor import MotionDataPreprocessor
from src.features.kinematics import KinematicFeatureExtractor
from src.scoring.cada_scorer import CADACompositeScorer


def train_cada_models(
    train_csv_path: Optional[Path] = None,
    test_csv_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None
) -> CADACompositeScorer:
    """
    Trains baseline profiler, isolation forest, supervised model, and composite CADA scorer on train_motion_data.csv,
    evaluates on test_motion_data.csv, and serializes artifacts.
    """
    train_path = train_csv_path or (RAW_DATA_DIR / "train_motion_data.csv")
    test_path = test_csv_path or (RAW_DATA_DIR / "test_motion_data.csv")
    out_models_dir = models_dir or MODELS_DIR
    out_results_dir = results_dir or RESULTS_DIR

    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {train_path}...")
    df_train = load_motion_data(train_path, require_target=True)
    
    print(f"Loading test data from {test_path}...")
    df_test = load_motion_data(test_path, require_target=True)

    # Preprocessing
    preprocessor = MotionDataPreprocessor()
    df_train_clean = preprocessor.fit_transform(df_train)
    df_test_clean = preprocessor.transform(df_test)

    # Kinematics Feature Extraction
    kinematics = KinematicFeatureExtractor()
    df_train_feat = kinematics.transform(df_train_clean)
    df_test_feat = kinematics.transform(df_test_clean)

    # Initialize and Train CADA Composite Scorer
    print("Training CADA Composite Scorer...")
    scorer = CADACompositeScorer()
    scorer.fit(df_train_feat, y_train=df_train_feat['Class'])

    # Score Test Dataset
    print("Evaluating CADA Risk Scores on test dataset...")
    df_test_scored = scorer.score_batch(df_test_feat)

    # Metrics
    y_test_bin = (df_test_feat['Class'] == 'AGGRESSIVE').astype(int)
    cada_scores = df_test_scored['CADA_Score']
    cada_preds = (cada_scores >= 50.0).astype(int)

    test_accuracy = float(accuracy_score(y_test_bin, cada_preds))
    test_f1 = float(f1_score(y_test_bin, cada_preds))
    test_roc = float(roc_auc_score(y_test_bin, cada_scores))
    test_precision = float(precision_score(y_test_bin, cada_preds, zero_division=0))
    test_recall = float(recall_score(y_test_bin, cada_preds, zero_division=0))

    tier_distribution = df_test_scored['Risk_Tier'].value_counts().to_dict()
    mean_scores_by_class = df_test_scored.groupby('Class')['CADA_Score'].mean().to_dict()

    summary_metrics = {
        "train_samples": len(df_train_feat),
        "test_samples": len(df_test_feat),
        "test_accuracy": round(test_accuracy, 4),
        "test_f1_score": round(test_f1, 4),
        "test_roc_auc": round(test_roc, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_tier_distribution": tier_distribution,
        "mean_cada_score_by_class": mean_scores_by_class
    }

    print("\n--- Training Summary ---")
    print(json.dumps(summary_metrics, indent=2))

    # Persist Models
    bundle_path = out_models_dir / "cada_model_bundle.joblib"
    scorer.save(bundle_path)
    print(f"\nSaved CADA model bundle to: {bundle_path}")

    scorer.baseline_profiler.save(out_models_dir / "baseline_profiler.joblib")
    scorer.iso_model.save(out_models_dir / "isolation_forest.joblib")

    # Persist training metrics report
    with open(out_results_dir / "training_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    return scorer


if __name__ == "__main__":
    train_cada_models()

