# API Reference

Base URL: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check if API and models are ready |
| `POST` | `/api/v1/score` | Score one sensor reading |
| `POST` | `/api/v1/stream` | Score consecutive readings in a stream |
| `POST` | `/api/v1/reset_stream` | Reset stream memory |
| `POST` | `/api/v1/batch_score` | Score a list of readings |

---

## Example: Score a Single Reading

`POST /api/v1/score`

### Request
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

### Response
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
