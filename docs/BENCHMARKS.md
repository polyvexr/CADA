# Benchmark Results

Performance comparison on the test dataset (`test_motion_data.csv`). Anomaly class: `AGGRESSIVE`. Safe classes: `NORMAL`, `SLOW`.

## Evaluation Summary


| Model            | Precision  | Recall | F1 Score   | Accuracy   | ROC AUC | PR AUC | Latency (ms/sample) |
| ---------------- | ---------- | ------ | ---------- | ---------- | ------- | ------ | ------------------- |
| Random Forest    | 0.7313     | 0.7088 | 0.7199     | 0.8544     | 0.9120  | 0.8204 | 0.0148              |
| Isolation Forest | 0.5543     | 0.3759 | 0.4480     | 0.7555     | 0.7598  | 0.5461 | 0.0063              |
| One-Class SVM    | 0.4000     | 0.4939 | 0.4420     | 0.6709     | 0.6579  | 0.5000 | 0.0135              |
| **CADA**         | **0.7952** | 0.6867 | **0.7370** | **0.8706** | 0.8965  | 0.7912 | 0.0173              |


## Key Points

1. **Continuous Score**: Rather than a simple yes/no label, CADA gives a score from 0 to 100.
2. **High Accuracy**: Reaches **87.1% accuracy** without needing exhaustive anomaly labels for training.
3. **Fast**: Scores each sample in less than 0.02 ms.
