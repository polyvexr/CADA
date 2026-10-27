# Getting Started with CADA

## 1. Installation

```bash
git clone https://github.com/polyvexr/CADA.git
cd CADA

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Dataset

Sensor data files in `data/raw/`:
- `train_motion_data.csv`: Training telemetry.
- `test_motion_data.csv`: Test telemetry.

Columns: `AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, `GyroZ`, `Timestamp`, `Class`.

---

## 3. Running the Project

### Train Models
```bash
python -m src.models.trainer
```
Saves model files to the `models/` directory.

### Start the REST API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 4. Running Tests
```bash
pytest tests/
```
