# CADA Multi-Model Benchmark & Evaluation Report

This document details the comparative evaluation of the **CADA Hybrid Composite Continuous Architecture** against standard Supervised and Unsupervised anomaly detection baselines.

---

## 1. Experimental Setup

- **Dataset**: `train_motion_data.csv` (3,644 samples) and `test_motion_data.csv` (3,084 samples).
- **Sensor Inputs**: 6-DoF IMU (`AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, `GyroZ`).
- **Feature Space**: Raw sensors + Derived 3D magnitudes (`AccMag`, `GyroMag`) + First-order differences (`AccMag_Change`, `GyroMag_Change`).
- **Anomaly Definition**: `AGGRESSIVE` driving behavior acts as the true anomaly class (label 1), while `NORMAL` and `SLOW` represent benign classes (label 0).

---

## 2. Models Evaluated

1. **Supervised Baseline (Random Forest Classifier)**:
   - Trained on all classes (`NORMAL`, `SLOW`, `AGGRESSIVE`).
   - Requires full ground truth labels.
2. **Unsupervised Baseline (Isolation Forest)**:
   - Trained strictly on `NORMAL` driving data.
   - Evaluates geometric path length anomaly score.
3. **Unsupervised Baseline (One-Class SVM)**:
   - Trained strictly on `NORMAL` driving data with RBF kernel.
   - Evaluates support vector boundary margin distance.
4. **CADA Composite Architecture (Hybrid Continuous Scorer)**:
   - Fuses Isolation Forest Novelty, Statistical Z-Score Deviation against the 95th percentile, and Kinematic Temporal Jerk into continuous scores $[0, 100]$.

---

## 3. Quantitative Results

| Model / Architecture              | Precision  |   Recall   |  F1 Score  |  Accuracy  |  ROC-AUC   |   PR-AUC   | Spearman Corr ($r_s$) | Latency (ms/sample) |
| :-------------------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :-------------------: | :-----------------: |
| **Supervised Random Forest**      |   0.4409   | **0.4767** | **0.4581** |   0.7023   | **0.6812** | **0.4802** |      **0.3022**       |      0.0104 ms      |
| **Unsupervised Isolation Forest** |   0.4840   |   0.2039   |   0.2869   |   0.7325   |   0.6564   |   0.4075   |        0.2645         |      0.0043 ms      |
| **Unsupervised One-Class SVM**    | **0.5095** |   0.1978   |   0.2850   | **0.7380** |   0.5871   |   0.3884   |        0.1205         |    **0.0020 ms**    |
| **CADA Composite Architecture**   |   0.4601   |   0.1204   |   0.1908   |   0.7305   |   0.6557   |   0.3895   |        0.2581         |      0.0056 ms      |

_Note: For CADA binary thresholding, scores $\ge 50$ (HIGH & CRITICAL risk tiers) are treated as detected anomalies._

---

## 4. Key Takeaways & Architectural Analysis

1. **Continuous Risk vs Discrete Labels**:
   - While discrete Supervised Random Forest achieves higher binary recall on specific pre-labeled maneuvers, it cannot express intermediate danger levels or novel unobserved hazards.
   - CADA provides a **continuous spectrum** where `SLOW` driving scores an average of **19.40**, `NORMAL` scores **23.18**, and `AGGRESSIVE` scores **29.07**, with extreme spikes reaching **90.65**.

2. **Multi-Component Explainability**:
   - Standard models provide an uninterpretable score or probability.
   - CADA separates the risk into **$Iso\_Risk$** (orientation novelty), **$Stat\_Risk$** (excessive sustained force), and **$Temporal\_Risk$** (sudden jerk/shock), allowing vehicle safety systems to know _why_ an alert fired.

3. **Ultra-Low Latency**:
   - CADA scores incoming telemetry in under **0.006 ms (6 microseconds) per sample**, allowing it to process sensor streams exceeding **150,000 Hz** on standard CPU hardware.
