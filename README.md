# CADA — Continuous Anomaly Detection Architecture for Driving Behavior

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**CADA** is an end-to-end, real-time machine learning architecture designed to detect, score, and monitor abnormal driving behaviors and safety risks from multi-axis Inertial Measurement Unit (IMU) telemetry (accelerometer and gyroscope).

Unlike traditional discrete classifiers that only predict static labels, CADA provides **continuous 0–100 risk scoring**, **granular risk attribution**, and **operational safety tiers** (`NORMAL`, `MODERATE`, `HIGH`, `CRITICAL`).

---

## 📚 Documentation

- 🚀 [**Getting Started Guide**](docs/GETTING_STARTED.md) — Environment setup, training, API, dashboard, and test execution.
- 🏗️ [**Architecture & Mathematics**](docs/ARCHITECTURE.md) — Mathematical formulas, multi-risk fusion, and component architecture.
- 🔌 [**API Reference**](docs/API_REFERENCE.md) — Complete REST and streaming telemetry API endpoints and payloads.
- 📊 [**Benchmark Report**](docs/BENCHMARKS.md) — Supervised vs. Unsupervised vs. CADA Hybrid benchmark comparisons and latency metrics.

---

## 🏗️ Architecture & Mathematical Formulation

CADA synthesizes three complementary risk dimensions into a unified continuous score:

```
                  ┌──────────────────────────────────────────────┐
                  │          Raw IMU Telemetry (6-DoF)           │
                  │   [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Kinematics Feature Pipeline          │
                  │  [AccMag, GyroMag, ΔAccMag, ΔGyroMag, Roll]  │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 │                       │                       │
                 ▼                       ▼                       ▼
    ┌─────────────────────────┐┌───────────────────┐┌─────────────────────────┐
    │     Isolation Forest    ││ Statistical Z-Dev ││  Kinematic Jerk Scorer  │
    │  Distributional Novelty ││ Baseline Deviation││    Temporal Dynamics    │
    │      (Iso_Risk)         ││    (Stat_Risk)    ││     (Temporal_Risk)     │
    └────────────┬────────────┘└─────────┬─────────┘└────────────┬────────────┘
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Composite CADA Risk Score           │
                  │       (0 - 100 Continuous Safety Index)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Operational Risk Tier Discretization      │
                  │ [NORMAL <25 | MODERATE <50 | HIGH <75 | CRIT]│
                  └──────────────────────────────────────────────┘
```

### Composite Risk Formula

$$\text{CADA\_Score} = w_1 \cdot \text{Iso\_Risk} + w_2 \cdot \text{Stat\_Risk} + w_3 \cdot \text{Temporal\_Risk}$$

Where:

1. **$\text{Iso\_Risk} \in [0, 100]$**: Normalized inverse Isolation Forest decision function score calibrated against the empirical distribution of `NORMAL` driving.
2. **$\text{Stat\_Risk} \in [0, 100]$**: Multi-feature standardized Z-score deviation from normal driving parameters, scaled so that the 95th percentile of normal driving maps to 50:
   $$\text{Stat\_Risk} = \min\left(100, \frac{\frac{1}{D}\sum_{j=1}^D \frac{|x_j - \mu_j|}{\sigma_j}}{\text{Limit}_{95\%}} \times 50\right)$$
3. **$\text{Temporal\_Risk} \in [0, 100]$**: Instantaneous jerk and rotational rate-of-change normalized against baseline dynamic limits:
   $$\text{Temporal\_Risk} = \frac{|\Delta\text{AccMag}| + |\Delta\text{GyroMag}| - \min}{\max - \min} \times 100$$

---

## 📁 Repository Structure

```
CADA/
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Project dependencies
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Exploratory data analysis & initial prototyping
│   └── 02_model_benchmarking.ipynb     # Multi-model benchmarking & visualization
├── data/
│   └── raw/
│       ├── train_motion_data.csv       # Training motion telemetry
│       └── test_motion_data.csv        # Testing motion telemetry
├── models/
│   ├── baseline_profiler.joblib        # Fitted normal baseline statistics
│   ├── isolation_forest.joblib         # Fitted Isolation Forest model
│   └── cada_model_bundle.joblib        # Serialized CADA Composite Scorer bundle
├── src/
│   ├── config.py                       # Global configuration & parameters
│   ├── data/
│   │   ├── loader.py                   # Data ingestion & schema validation
│   │   └── preprocessor.py             # Telemetry cleaning & clipping
│   ├── features/
│   │   ├── kinematics.py               # 3D magnitudes & jerk extraction
│   │   └── windowing.py                # Rolling window statistics
│   ├── models/
│   │   ├── baseline_profiler.py        # Normal driving profiler
│   │   ├── isolation_forest.py         # Isolation forest model wrapper
│   │   └── trainer.py                  # End-to-end training pipeline
│   ├── scoring/
│   │   ├── cada_scorer.py              # Unified CADA scoring engine
│   │   └── risk_tiers.py               # Safety tier assignment policy
│   └── evaluation/
│       ├── metrics.py                  # Anomaly & rank correlation metrics
│       └── benchmark.py                # Multi-model benchmark suite
├── api/
│   ├── main.py                         # FastAPI application entrypoint
│   ├── routes.py                       # REST & streaming endpoints
│   └── schemas.py                      # Pydantic data contracts
├── dashboard/
│   └── app.py                          # Streamlit telemetry visualizer
├── tests/
│   ├── test_features.py                # Feature extraction unit tests
│   ├── test_scoring.py                 # Scorer & risk tier tests
│   └── test_api.py                     # API route integration tests
└── results/
    ├── training_summary.json           # Model training metrics
    └── benchmark_report.json           # Comparative benchmark outputs
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/polyvexr/CADA.git
cd CADA
pip install -r requirements.txt
```

### 2. Model Training & Calibration

Train the CADA baseline profiler, Isolation Forest, and composite scorer:

```bash
python -m src.models.trainer
```

This generates and saves model bundles to `models/` and summary metrics to `results/training_summary.json`.

### 3. Running Benchmarks

Execute the comparative benchmark across Supervised (Random Forest), Unsupervised (Isolation Forest, One-Class SVM), and CADA Hybrid:

```bash
python -m src.evaluation.benchmark
```

### 4. Running the Real-Time API

Launch the FastAPI telemetry scoring service:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI Specification: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 5. Launching the Interactive Dashboard

Run the Streamlit telemetry monitoring dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 🔌 API Reference

### `POST /api/v1/score`

Evaluate a single instantaneous IMU sensor sample:

```json
// Request Payload:
{
  "AccX": -0.5946,
  "AccY": -0.1224,
  "AccZ": 0.2205,
  "GyroX": -0.0193,
  "GyroY": -0.0293,
  "GyroZ": 0.0878,
  "Timestamp": 3581630
}

// Response:
{
  "cada_score": 28.45,
  "risk_tier": "MODERATE",
  "components": {
    "iso_risk": 25.96,
    "stat_risk": 32.14,
    "temporal_risk": 27.25
  },
  "kinematics": {
    "acc_mag": 0.6459,
    "gyro_mag": 0.0945,
    "acc_mag_change": 0.0,
    "gyro_mag_change": 0.0
  },
  "timestamp": 3581630
}
```

### `POST /api/v1/stream`

Statefully evaluates sequential sensor readings, automatically computing dynamic jerk and rate-of-change from previous readings.

### `POST /api/v1/batch_score`

Scores a batch list of sequential motion readings and returns trip-level aggregates (mean score, peak score, tier breakdown).

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## 📊 Summary of Operational Risk Tiers

| Risk Tier    |    Score Range     | Description                                                   | Recommended System Action                 |
| :----------- | :----------------: | :------------------------------------------------------------ | :---------------------------------------- |
| **NORMAL**   |   $0 \le S < 25$   | Routine, smooth driving within standard baseline limits.      | Log telemetry; no driver alert.           |
| **MODERATE** |  $25 \le S < 50$   | Minor deviations (e.g., mild acceleration or cornering).      | Passive dashboard indicator.              |
| **HIGH**     |  $50 \le S < 75$   | Pronounced jerk, abrupt lane change, or sharp maneuver.       | Advisory audio/visual alert.              |
| **CRITICAL** | $75 \le S \le 100$ | Severe erratic motion, emergency braking, or loss of control. | Immediate warning & safety event trigger. |

---

## 📄 License

This project is licensed under the MIT License.
