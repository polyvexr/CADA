# Benchmark Results

Performance comparison on the test dataset (`test_motion_data.csv`). Anomaly class: `AGGRESSIVE`. Safe classes: `NORMAL`, `SLOW`.

## Evaluation Summary

| Model | Type | Accuracy | F1 Score | ROC-AUC | Speed (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Supervised | 87.7% | 0.758 | 0.887 | 0.003 ms |
| **Random Forest** | Supervised | 85.4% | 0.720 | 0.912 | 0.015 ms |
| **Isolation Forest** | Unsupervised | 75.6% | 0.448 | 0.760 | 0.006 ms |
| **One-Class SVM** | Unsupervised | 67.1% | 0.442 | 0.658 | 0.014 ms |
| **CADA (Composite)** | Hybrid | **87.1%** | **0.737** | **0.897** | 0.017 ms |

## Key Points

1. **Continuous Score**: Rather than a simple yes/no label, CADA gives a score from 0 to 100.
2. **High Accuracy**: Reaches **87.1% accuracy** without needing exhaustive anomaly labels for training.
3. **Fast**: Scores each sample in less than 0.02 ms.
