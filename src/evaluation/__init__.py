"""
Model evaluation and benchmarking package.
"""

from src.evaluation.metrics import evaluate_anomaly_detection, compare_models
from src.evaluation.benchmark import run_full_benchmark

__all__ = ["evaluate_anomaly_detection", "compare_models", "run_full_benchmark"]

