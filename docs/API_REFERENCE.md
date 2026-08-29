# CADA REST & Streaming API Reference

The CADA API provides low-latency REST endpoints for single-point scoring, batch trip assessment, and stateful telemetry streaming.

- **Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## 1. Endpoints Summary

| Method | Path                   | Description                                    |
| :----- | :--------------------- | :--------------------------------------------- |
| `GET`  | `/health`              | Service health and model readiness check       |
| `POST` | `/api/v1/score`        | Score a single instantaneous telemetry reading |
| `POST` | `/api/v1/stream`       | Stateful sequential telemetry scoring          |
| `POST` | `/api/v1/reset_stream` | Reset streaming history and jerk states        |
| `POST` | `/api/v1/batch_score`  | Score a batch trip of sequential readings      |

---

## 2. Endpoint Details

### 2.1 Health Check

`GET /health`

#### Response (`200 OK`)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### 2.2 Single Sample Scoring

`POST /api/v1/score`

Scores an instantaneous IMU reading.

#### Request Body

```json
{
  "AccX": -0.5946,
  "AccY": -0.1224,
  "AccZ": 0.2205,
  "GyroX": -0.0193,
  "GyroY": -0.0293,
  "GyroZ": 0.0878,
  "Timestamp": 3581630
}
```

#### Response (`200 OK`)

```json
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

#### Python Example

```python
import requests

payload = {
    "AccX": 1.25, "AccY": -0.85, "AccZ": 0.32,
    "GyroX": 0.12, "GyroY": -0.05, "GyroZ": 0.18,
    "Timestamp": 1000
}

response = requests.post("http://localhost:8000/api/v1/score", json=payload)
data = response.json()
print(f"CADA Score: {data['cada_score']} | Tier: {data['risk_tier']}")
```

---

### 2.3 Stateful Streaming Scoring

`POST /api/v1/stream`

Automatically retains previous magnitude in memory to compute continuous jerk ($\Delta \text{AccMag}, \Delta \text{GyroMag}$) across consecutive HTTP calls.

#### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"AccX": 3.5, "AccY": 2.1, "AccZ": 1.0, "GyroX": 0.4, "GyroY": 0.2, "GyroZ": 0.5}'
```

---

### 2.4 Reset Stream State

`POST /api/v1/reset_stream`

Clears internal kinematic difference history for a new trip or driver session.

---

### 2.5 Batch Trip Scoring

`POST /api/v1/batch_score`

Evaluates an array of telemetry readings and computes trip-level safety summaries.

#### Request Body

```json
{
  "readings": [
    {
      "AccX": 0.0,
      "AccY": 0.0,
      "AccZ": 0.0,
      "GyroX": 0.0,
      "GyroY": 0.0,
      "GyroZ": 0.0,
      "Timestamp": 1
    },
    {
      "AccX": 1.2,
      "AccY": -0.4,
      "AccZ": 0.1,
      "GyroX": 0.05,
      "GyroY": -0.02,
      "GyroZ": 0.01,
      "Timestamp": 2
    }
  ]
}
```

#### Response (`200 OK`)

```json
{
  "total_samples": 2,
  "mean_cada_score": 18.25,
  "max_cada_score": 24.50,
  "tier_counts": {
    "NORMAL": 2
  },
  "results": [ ... ]
}
```
