# CADA Beginner Guide

This guide is written for someone who is starting from zero and wants to understand the entire project quickly.

The goal of this repository is not just to train a model. It is to build an end-to-end system that:

- reads motion sensor data from a car or driving telemetry stream,
- extracts useful physical features like acceleration magnitude and jerk,
- compares the signal against what is considered normal driving,
- calculates a continuous risk score from 0 to 100,
- exposes that score through an API,
- and visualizes the result in a dashboard.

If you are a fresher, read this document in order: first the big picture, then the project structure, then the code modules, then the theory, and finally the runtime flow.

---

## 1. What problem is this project solving?

The project is called CADA, which stands for Continuous Anomaly Detection Architecture.

The main idea is to detect unusual or risky driving behavior from IMU sensor data. IMU means Inertial Measurement Unit, which usually provides:

- AccX, AccY, AccZ: acceleration along three axes
- GyroX, GyroY, GyroZ: rotational angular velocity around three axes

This project uses these signals to estimate how risky a driving pattern is at a given moment.

Unlike a normal classifier that only says "AGGRESSIVE" or "NORMAL", CADA tries to compute a continuous score from 0 to 100. This is useful because real-world driving is continuous and gradual:

- a mild sudden turn may be slightly risky,
- a sharp braking event may be highly risky,
- a smooth normal drive should stay near the safe zone.

So the idea is to map raw sensor behavior to a risk score and assign safety tiers like:

- NORMAL
- MODERATE
- HIGH
- CRITICAL

---

## 2. The main concept behind CADA

The project combines several signals and ideas:

1. Kinematic features
   - acceleration magnitude
   - rotation magnitude
   - changes in acceleration over time
   - changes in angular velocity over time

2. Outlier detection
   - normal driving is learned from data
   - unusual motion patterns are treated as anomalies

3. Statistical deviation
   - compare current features against normal baseline statistics
   - if the current sample is far from the normal pattern, risk increases

4. Temporal risk
   - sudden changes in motion create spikes in risk
   - this captures jerk, abrupt braking, sharp steering, and instability

5. Composite scoring
   - combine all risk signals into a final score

This is the heart of the project.

---

## 3. How the project is organized

This is the high-level folder structure:

```text
CADA/
├── README.md
├── requirements.txt
├── docs/
├── data/
├── models/
├── src/
├── api/
├── dashboard/
├── tests/
├── results/
└── notebooks/
```

### Folder-by-folder explanation

#### README.md

This is the entry point. It explains the project summary, architecture, and quick start steps.

#### docs/

This contains deeper documentation and explanations.

- GETTING_STARTED.md: how to install and run the project
- ARCHITECTURE.md: mathematical and system-level design
- API_REFERENCE.md: API endpoints and payload formats
- BENCHMARKS.md: comparison of different models
- BEGINNER_GUIDE.md: this file

#### data/

Stores raw training and testing datasets.

- data/raw/train_motion_data.csv
- data/raw/test_motion_data.csv

These CSV files are the main input to the pipeline.

#### models/

Stores trained model artifacts after running training.

Examples:

- baseline_profiler.joblib
- isolation_forest.joblib
- cada_model_bundle.joblib

#### src/

This is the mathematical and ML core of the project.

This is the most important folder for a beginner. Almost all logic lives here.

#### api/

This contains the FastAPI-based application that exposes the model as a web service.

#### dashboard/

This contains the Streamlit app that visualizes driver risk in a browser.

#### tests/

Contains unit tests and integration tests.

#### results/

Stores output reports and training summaries.

#### notebooks/

Contains exploratory and analysis notebooks used for investigation and model benchmarking.

---

## 4. The real code flow: from sensor data to score

A beginner should understand the full runtime pipeline.

### Step 1: Load data

The first place to look is:

- src/data/loader.py

This file validates the input data and loads the CSV file into a pandas DataFrame.

The important function is:

- validate_motion_dataframe()
- load_motion_data()

These functions check:

- required columns exist,
- sensor columns are numeric,
- timestamp is valid if present,
- the file is not empty,
- required target (`Class`) exists if needed.

This is important because all later steps assume the data has a known format.

### Step 2: Clean and arrange data

Next is:

- src/data/preprocessor.py

The MotionDataPreprocessor class performs:

- removing duplicate rows,
- sorting by timestamp,
- filling missing values,
- clipping extreme values (optional),
- preparing clean telemetry for the feature extractor.

This ensures the ML pipeline sees consistent data.

### Step 3: Convert raw sensor data to meaningful engineering features

This is the deepest and most important part of the project.

File:

- src/features/kinematics.py

This file includes the KinematicFeatureExtractor class.

What does it do?

- computes acceleration magnitude: AccMag
- computes gyro magnitude: GyroMag
- computes pitch and roll angles
- computes kinetic energy and rotational energy
- computes differences over time: AccMag_Change, GyroMag_Change
- computes rolling-window statistics across multiple window sizes like 3, 7, 15, 25, 40

This is where raw sensor values become features that can be understood by a risk model.

Examples of feature creation:

- AccMag = sqrt(AccX^2 + AccY^2 + AccZ^2)
- GyroMag = sqrt(GyroX^2 + GyroY^2 + GyroZ^2)
- AccMag_Change = current magnitude - previous magnitude

These are powerful because sudden jumps in these values indicate risky or unstable motion.

### Step 4: Learn the normal baseline

File:

- src/models/baseline_profiler.py

The NormalBaselineProfiler learns a statistical baseline from normal driving samples.

It computes:

- mean of each feature
- standard deviation of each feature
- a deviation score relative to the learned baseline
- a 95th percentile cutoff to represent what is considered normal variance

This gives the model a sense of what typical driving looks like.

If a new sample is far from the baseline, its risk rises.

### Step 5: Learn anomaly behavior with Isolation Forest

File:

- src/models/isolation_forest.py

This wraps sklearn's IsolationForest.

Isolation Forest is an unsupervised anomaly detection algorithm. It learns how to isolate abnormal points by splitting the data in random feature space.

The basic idea:

- normal points are dense and harder to isolate,
- abnormal points are different and easier to isolate,
- the model outputs a raw decision score,
- that score is normalized into a 0-100 anomaly risk.

This component is the "out-of-distribution" detector.

### Step 6: Combine everything in the CADA scorer

File:

- src/scoring/cada_scorer.py

This is the most important model orchestration file.

The class CADACompositeScorer is the master scoring engine. It combines three components:

- Iso_Risk
- Stat_Risk
- Temporal_Risk

It may also add a supervised risk component if labels are available.

The class provides:

- fit()
- score_sample()
- score_batch()
- save()
- load()

This is the central runtime object used in both API and dashboard.

### Step 7: Define risk tiers

File:

- src/scoring/risk_tiers.py

This file decides the final category:

- NORMAL
- MODERATE
- HIGH
- CRITICAL

The score ranges usually look like:

- 0 - 25: NORMAL
- 25 - 50: MODERATE
- 50 - 75: HIGH
- 75 - 100: CRITICAL

This is how the continuous risk score becomes a business-friendly safety label.

### Step 8: Train the model pipeline

File:

- src/models/trainer.py

The train_cada_models() function trains the end-to-end pipeline.

It does:

- loads train and test CSV data,
- cleans and preprocesses them,
- extracts kinematic features,
- initializes CADACompositeScorer,
- fits the scorer,
- evaluates on test data,
- saves model bundle to models/
- writes summary metrics to results/

This is the main training script for the whole architecture.

### Step 9: Build API endpoints

Folder:

- api/

Files:

- api/main.py
- api/routes.py
- api/schemas.py

This exposes the system through FastAPI.

Key concept:

- the server loads the trained model bundle,
- the model scores an incoming sensor reading,
- the API returns the final score and breakdown.

Important endpoints:

- POST /api/v1/score
- POST /api/v1/stream
- POST /api/v1/reset_stream
- POST /api/v1/batch_score
- GET /health

The schemas define the exact request and response format so that client code can call the API reliably.

### Step 10: Visualize in a dashboard

File:

- dashboard/app.py

This is a Streamlit dashboard used for:

- showing CADA risk over time,
- comparing different sensor traces,
- inspecting the gauge for a sample,
- exploring risk component attribution,
- visualizing tier counts.

This is useful for product teams and engineers who want to see how the model behaves in practice.

---

## 5. The most important theoretical ideas

### 5.1 Why use acceleration and gyro magnitude?

Acceleration is a direct indicator of force. When a vehicle suddenly accelerates, brakes, or turns sharply, the magnitude of acceleration changes sharply.

Gyro magnitude captures rotational movement. Sudden changes in rotation often correspond to aggressive steering or loss of control.

Together, they capture motion in both linear and angular space.

### 5.2 Why use jerk?

Jerk is the rate of change of acceleration. It is a common measure of abruptness.

A smooth drive has small jerk. A sudden lane change or hard braking has large jerk.

That is why the project uses AccMag_Change and GyroMag_Change as important warning features.

### 5.3 Why use Isolation Forest?

Not every anomaly has a label in the dataset. Some risky events may be rare and unknown.

Isolation Forest is designed for exactly this case: detect unusual patterns without needing large labeled anomaly examples.

### 5.4 Why use statistical deviations?

Normal driving is not random; it has mean and variance. If a reading deviates strongly from the normal distribution, it is likely abnormal.

The project calculates z-score-like deviation against the normal baseline.

### 5.5 Why combine multiple components?

No single model is perfect.

- Isolation Forest catches unusual outliers.
- Baseline profiler catches deviation from the normal pattern.
- Temporal component catches sudden dynamic spikes.

Combining them gives a more robust total score.

---

## 6. Key formulas used in the project

The repository uses a few simple but important formulas.

### Acceleration magnitude

$$
\text{AccMag} = \sqrt{AccX^2 + AccY^2 + AccZ^2}
$$

### Gyro magnitude

$$
\text{GyroMag} = \sqrt{GyroX^2 + GyroY^2 + GyroZ^2}
$$

### Temporal change

$$
\Delta AccMag_t = AccMag_t - AccMag_{t-1}
$$

$$
\Delta GyroMag_t = GyroMag_t - GyroMag_{t-1}
$$

### Normal baseline risk

The project computes how much a feature differs from the normal mean in units of standard deviation. That is the statistical idea behind z-scores.

### Composite score

The full score is a weighted combination of component risks:

$$
\text{CADA\_Score} = w_1 \cdot \text{Iso\_Risk} + w_2 \cdot \text{Stat\_Risk} + w_3 \cdot \text{Temporal\_Risk}
$$

with the final score normalized to 0-100.

---

## 7. Reading the project like a fresher

Here is a recommended reading order.

### First read

1. README.md
2. docs/GETTING_STARTED.md
3. docs/ARCHITECTURE.md
4. src/config.py

### Second read

5. src/data/loader.py
6. src/data/preprocessor.py
7. src/features/kinematics.py
8. src/models/baseline_profiler.py
9. src/models/isolation_forest.py

### Third read

10. src/scoring/cada_scorer.py
11. src/scoring/risk_tiers.py
12. src/models/trainer.py
13. api/routes.py
14. api/main.py
15. dashboard/app.py

### Last read

16. tests/

If you read in this order, the project will start making sense quickly.

---

## 8. File-by-file reference guide

### src/config.py

Contains global path constants and model configuration.

Important values:

- BASE_DIR
- DATA_DIR
- RAW_DATA_DIR
- MODELS_DIR
- RESULTS_DIR
- RAW_SENSOR_COLS
- FEATURE_COLS
- TIMESTAMP_COL
- TARGET_COL

This file is a central settings file. Most modules reference it.

### src/data/loader.py

Loads and validates CSV sensor data.

This is the entry point for dataset ingestion.

### src/data/preprocessor.py

Cleans raw telemetry; this act prevents bad sensor values from corrupting the model.

### src/features/kinematics.py

Converts raw sensor readings into engineered features.

This is core feature engineering and should be understood well.

### src/models/baseline_profiler.py

Learns expected normal behavior using statistics.

### src/models/isolation_forest.py

Detects unusual motion using anomaly detection.

### src/scoring/cada_scorer.py

Combines multiple risk components into a single score.

This is the main scoring logic.

### src/scoring/risk_tiers.py

Maps numeric risk scores to business-friendly tier labels.

### src/models/trainer.py

Runs the training pipeline and saves the model bundle.

### api/schemas.py

Defines the API request and response contracts using Pydantic models.

### api/routes.py

Contains the application routes and logic for scoring requests.

### api/main.py

Starts the FastAPI app and wires in the router.

### dashboard/app.py

Provides user-friendly visualization and monitoring.

### tests/

Verifies that features, scoring logic, and API endpoints work as expected.

---

## 9. What is the project trying to achieve at a product level?

The repository is not just a toy ML repo. It is designed to simulate a real-world safety monitoring or telematics system.

Possible use cases:

- driver monitoring in fleet systems,
- crash or near-crash risk alerts,
- anomaly detection in telematics,
- analysis of harsh braking and sudden steering,
- dashboarding of safety metrics,
- real-time risk scoring in embedded systems.

This means the project is closer to a production-style ML service than a single script model.

---

## 10. Common beginner confusion points

### Why is there both a training script and an API?

The training script is for model development. The API is for serving the trained model to users or apps.

### Why are there multiple models?

CADA is a hybrid architecture. It combines more than one method:

- unsupervised novelty model,
- statistical baseline model,
- temporal dynamic risk,
- possible supervised learning signal.

This is intentionally more robust than using one single classifier.

### Why do we compute so many features?

Machine learning works better when we provide physically meaningful variables. The raw acceleration signal is noisy and not enough by itself.

Derived features like magnitude, differences, and rolling statistics provide better insight.

### Why is there a dashboard if the API already exists?

The dashboard is for human inspection, debugging, and monitoring. It gives an intuitive view of how the model behaves over time.

---

## 11. Typical development ideas for a fresher

If you want to contribute to the project, these are good beginner tasks:

### Idea 1: add a new feature

Create a new feature in KinematicFeatureExtractor such as:

- speed-like magnitude estimate,
- acceleration variance,
- angular acceleration,
- handling of sign-aware changes.

### Idea 2: improve the score logic

Try changing the weights in config.py or learn how different risk component weights affect performance.

### Idea 3: add a new endpoint

Add a route in api/routes.py to expose a new analysis, such as:

- driver trip summary
- anomaly trend over time
- per-window severity endpoint

### Idea 4: test edge cases

Add tests for:

- missing values,
- extreme sensor values,
- empty request payload,
- invalid timestamp, etc.

### Idea 5: add model benchmarking

Extend the benchmark pipeline in src/evaluation/benchmark.py with more algorithms.

---

## 12. Good debugging approach

When you are reading this project, use this strategy:

1. Start from the training script: src/models/trainer.py
2. Trace how data moves into the pipeline
3. Read the feature extraction step carefully
4. Understand the score assembly in src/scoring/cada_scorer.py
5. Check the API response mapping in api/routes.py
6. Inspect the dashboard to understand the visuals

This top-down understanding is more valuable than reading all files randomly.

---

## 13. Summary for a fresher

The easiest way to think about the project is:

- raw motion sensor data enters the pipeline,
- the data is cleaned and engineered into physical features,
- the system compares those features against normal behavior,
- the system identifies abnormal motion and sudden risky changes,
- the risk components are combined into one score,
- the score is exposed over API and displayed in a dashboard.

In short, this project is an end-to-end risk detection pipeline for driving behavior.

If you understand these steps, you are already understanding the heart of the project.

---

## 14. Final learning checkpoint

Before you consider yourself comfortable with the project, make sure you can answer these questions:

- What does src/data/loader.py do?
- Why is feature engineering important in src/features/kinematics.py?
- How does the baseline profiler calculate deviation from normal behavior?
- What does Isolation Forest detect?
- How does CADACompositeScorer combine the risk signals?
- Which files are responsible for API and dashboard output?
- What is the purpose of the final risk tier mapping?

If you can answer these correctly, you already have a strong beginner-level understanding of the system.

---

## 15. Recommended next steps

- Run the training script once to understand model artifacts.
- Open the model results in results/training_summary.json.
- Call the API using the sample request from README.md.
- Explore the Streamlit dashboard.
- Modify one feature and see how the score changes.

That practical experience will make the code far easier to understand than reading only documentation.

---

This beginner guide is meant to give you a mental map of the project. Once you understand the flow from sensor data to score, the rest of the code becomes much easier to follow.
