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

| Model / Architecture                | Precision  |   Recall   |  F1 Score  |  Accuracy  |  ROC-AUC   |   PR-AUC   | Spearman Corr ($r_s$) | Latency (ms/sample) |
| :---------------------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :-------------------: | :-----------------: |
| **Supervised HistGradientBoosting** |   0.7904   | **0.7273** | **0.7575** | **0.8771** |   0.8867   | **0.8359** |        0.6389         |    **0.0029 ms**    |
| **Supervised Random Forest**        |   0.7313   |   0.7088   |   0.7199   |   0.8544   | **0.9120** |   0.8204   |      **0.6833**       |      0.0148 ms      |
| **Unsupervised Isolation Forest**   |   0.5543   |   0.3759   |   0.4480   |   0.7555   |   0.7598   |   0.5461   |        0.4150         |      0.0063 ms      |
| **Unsupervised One-Class SVM**      |   0.4000   |   0.4939   |   0.4420   |   0.6709   |   0.6579   |   0.5000   |        0.1997         |      0.0135 ms      |
| **CADA Composite Architecture**     | **0.7952** |   0.6867   |   0.7370   | **0.8706** |   0.8965   |   0.7912   |        0.6020         |      0.0173 ms      |

_Note: For CADA binary thresholding, scores $\ge 50$ (HIGH & CRITICAL risk tiers) are treated as detected anomalies._

---

## 4. Key Takeaways & Architectural Analysis

1. **Accuracy Range (80%–90%)**:
   - Upgraded feature representation with multi-scale rolling temporal dynamics ($w \in \{3, 7, 15, 25, 40\}$), orientation kinematics (Pitch, Roll, Yaw rate), and 2nd-order jerk derivatives boosted classification accuracy from **~70–73%** to **87.1% – 87.7%**.
   - CADA composite continuous scoring achieves **87.06% accuracy** with an ROC-AUC of **0.8965** and F1 score of **0.7370**.

2. **Continuous Risk Spectrum**:
   - CADA provides a **continuous spectrum** where `SLOW` driving scores an average of **11.46**, `NORMAL` scores **21.21**, and `AGGRESSIVE` scores **56.77**, with extreme spikes reaching **95.0+**.

3. **Multi-Component Explainability**:
   - CADA transparently isolates risk into **$Iso\_Risk$** (unsupervised orientation novelty), **$Stat\_Risk$** (statistical baseline deviations), **$Temporal\_Risk$** (jerk/volatility shock), and learned maneuver risk.

4. **Ultra-Low Latency**:
   - CADA scores incoming telemetry in under **0.02 ms per sample**, allowing high-frequency streaming over **50,000 Hz** on standard CPU hardware.
