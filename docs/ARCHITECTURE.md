# Architecture

CADA converts continuous vehicle motion sensor data (accelerometer and gyroscope) into a real-time safety score between 0 and 100.

## 1. System Pipeline

```
Raw IMU Telemetry (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
                       │
                       ▼
            Preprocessor & Validator
                       │
                       ▼
          Feature Extraction (Magnitudes & Jerk)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   Novelty Risk  Statistical Risk  Temporal Jerk
    (0 - 100)       (0 - 100)        (0 - 100)
         └─────────────┬─────────────┘
                       ▼
             Combined CADA Score
                  (0 - 100)
                       │
                       ▼
                 Safety Tiers
   [ NORMAL <25 | MODERATE <50 | HIGH <75 | CRITICAL ]
```

---

## 2. Feature Extraction

From 3D acceleration $(a_x, a_y, a_z)$ and angular velocity $(\omega_x, \omega_y, \omega_z)$:

- **Total Acceleration Magnitude**:
  $$\text{AccMag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

- **Total Rotation Magnitude**:
  $$\text{GyroMag} = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$$

- **Rate of Change (Jerk)**:
  $$\Delta \text{AccMag}_t = \text{AccMag}_t - \text{AccMag}_{t-1}$$
  $$\Delta \text{GyroMag}_t = \text{GyroMag}_t - \text{GyroMag}_{t-1}$$

---

## 3. Score Calculation

The final score is a weighted blend of three components:

$$\text{CADA\_Score} = w_1 \cdot \text{Iso\_Risk} + w_2 \cdot \text{Stat\_Risk} + w_3 \cdot \text{Temporal\_Risk}$$

1. **Novelty Risk (`Iso_Risk`)**: How unfamiliar the motion pattern looks compared to normal driving data.
2. **Statistical Risk (`Stat_Risk`)**: How far current sensor values stray from normal driving averages.
3. **Temporal Risk (`Temporal_Risk`)**: Sudden shocks, hard turns, or abrupt braking based on rate-of-change.

---

## 4. Safety Tiers

| Tier | Score Range | Meaning | Action |
| :--- | :---: | :--- | :--- |
| **NORMAL** | 0 – 24 | Safe, everyday driving | Log data normally |
| **MODERATE** | 25 – 49 | Minor harsh motion (e.g. brisk acceleration) | Show info status on screen |
| **HIGH** | 50 – 74 | Hard braking or sharp swerving | Driver caution alert |
| **CRITICAL** | 75 – 100 | Severe sudden shock or loss of control | Immediate critical warning |
