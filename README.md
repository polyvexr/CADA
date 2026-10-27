# CADA — Continuous Anomaly Detection Architecture

CADA calculates a real-time driving risk score (0 to 100) using motion sensor data (accelerometer and gyroscope).

Instead of only outputting static tags like "Aggressive" or "Normal", CADA scores driving smoothness continuously and maps it into clear safety tiers: **NORMAL**, **MODERATE**, **HIGH**, and **CRITICAL**.

---

## Architecture

```
Raw Sensor Data (Acc & Gyro)
            │
            ▼
Feature Extraction (Magnitudes & Jerk)
            │
            ├───────────────┬───────────────┐
            ▼               ▼               ▼
       Novelty Risk    Statistical Risk   Temporal Jerk
        (0 - 100)        (0 - 100)          (0 - 100)
            └───────────────┬───────────────┘
                            ▼
                    Combined CADA Score
                         (0 - 100)
                            │
                            ▼
                      Safety Tiers
       [ NORMAL <25 | MODERATE <50 | HIGH <75 | CRITICAL ]
```

---

## Project Structure

```
CADA/
├── api/                  # FastAPI web service
│   ├── main.py           # App entrypoint
│   ├── routes.py         # Scoring endpoints
│   └── schemas.py        # Request & response data models
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── data/
│   ├── raw/              # Original dataset (train & test)
│   └── test/             # Split data parts for testing
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md   # System pipeline & formulas
│   ├── GETTING_STARTED.md# Setup & run instructions
│   ├── API_REFERENCE.md  # Endpoints & payloads
│   └── BENCHMARKS.md     # Model accuracy & speed
├── models/               # Saved model weights
├── notebooks/            # Exploratory notebooks
├── src/                  # Core library code
│   ├── config.py         # Settings & paths
│   ├── data/             # Data loading & cleaning
│   ├── features/         # Kinematics & window calculations
│   ├── models/           # Isolation Forest & Baseline profiler
│   └── scoring/          # CADA scoring & tier logic
├── tests/                # Automated tests
└── requirements.txt      # Python dependencies
```

---

## Quickstart

### 1. Setup

```bash
git clone https://github.com/polyvexr/CADA.git
cd CADA

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models

```bash
python -m src.models.trainer
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Open the Dashboard

```bash
streamlit run dashboard/app.py
```
Dashboard: [http://localhost:8501](http://localhost:8501)

### 5. Run Tests

```bash
pytest tests/
```

---

## Safety Tiers

| Tier | Score Range | Description | Action |
| :--- | :---: | :--- | :--- |
| **NORMAL** | 0 – 24 | Smooth everyday driving | Standard logging |
| **MODERATE** | 25 – 49 | Minor harsh events (rapid acceleration, firm turn) | Information indicator |
| **HIGH** | 50 – 74 | Sharp turns or sudden braking | Driver alert |
| **CRITICAL** | 75 – 100 | Severe shock or loss of vehicle control | Immediate emergency alert |

---

## Documentation Links

- [Architecture & Calculations](docs/ARCHITECTURE.md)
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [API Reference](docs/API_REFERENCE.md)
- [Benchmark Results](docs/BENCHMARKS.md)
