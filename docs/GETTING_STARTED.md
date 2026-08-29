# Getting Started with CADA

This guide walks you through setting up the environment, training models, running benchmarks, launching the real-time API, and viewing the interactive dashboard.

---

## 1. Prerequisites & Installation

### Python Environment

CADA requires Python **3.9+** (tested up to Python 3.13).

```bash
# Clone the repository
git clone https://github.com/polyvexr/CADA.git
cd CADA

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Structure

The dataset contains 6-DOF IMU motion sensor telemetry:

- `data/raw/train_motion_data.csv`: 3,644 telemetry samples across 3 classes (`NORMAL`, `SLOW`, `AGGRESSIVE`).
- `data/raw/test_motion_data.csv`: 3,084 telemetry samples for evaluation.

### Expected Telemetry Schema

| Column                    |   Type    | Description                                                    |
| :------------------------ | :-------: | :------------------------------------------------------------- |
| `AccX`, `AccY`, `AccZ`    | `float64` | Tri-axial acceleration (g)                                     |
| `GyroX`, `GyroY`, `GyroZ` | `float64` | Tri-axial angular velocity (rad/s or deg/s)                    |
| `Timestamp`               |  `int64`  | Millisecond timestamp sequence                                 |
| `Class`                   |   `str`   | Driving behavior ground truth (`NORMAL`, `SLOW`, `AGGRESSIVE`) |

---

## 3. End-to-End Workflow

### Step 1: Train CADA Models

Fit the Normal Driving Baseline Profiler and Isolation Forest model on the `NORMAL` driving dataset:

```bash
python -m src.models.trainer
```

**Outputs**:

- `models/cada_model_bundle.joblib` (Master scoring pipeline)
- `models/baseline_profiler.joblib` (Statistical baseline parameters)
- `models/isolation_forest.joblib` (Novelty detector)
- `results/training_summary.json` (Training & scoring distribution summary)

---

### Step 2: Run Multi-Model Benchmarks

Compare Supervised Random Forest, Unsupervised Isolation Forest, One-Class SVM, and CADA Hybrid:

```bash
python -m src.evaluation.benchmark
```

**Outputs**:

- `results/benchmark_report.json`
- `results/benchmark_comparison.csv`

---

### Step 3: Launch Real-Time Telemetry API

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Documentation (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Step 4: Launch Interactive Streamlit Dashboard

Visualize trip telemetry, live speedometer risk gauges, and risk attribution waterfalls:

```bash
streamlit run dashboard/app.py
```

Opens in your browser at `http://localhost:8501`.

---

## 4. Running Automated Tests

Run the test suite to verify data validation, feature extraction, scoring guarantees, and API endpoints:

```bash
# Run feature & scoring unit tests
python tests/run_tests.py

# Run API integration tests
python tests/test_api_unittest.py
```
