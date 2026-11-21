# Real-Time Big Data Credit Card Fraud Detection System

An enterprise-grade, distributed credit card fraud detection system built using **Apache Spark (PySpark MLlib)**, **Spark Structured Streaming**, and **Power BI**. 

The system handles extreme class imbalance (0.172% fraud incidence) through cost-sensitive class weighting, employs a performance-weighted ensemble (combining PR-AUC and ROC-AUC metrics), tunes classification thresholds under real-world business constraints ($\ge 85\%$ recall), and features an authentic **Real-Time Streaming Cockpit** and **Blind Test-Set Explorer** for live evaluation.

---

## 🏗️ Architecture & Pipeline Flow

```
[Kaggle Credit Card Data (284,807 transactions)]
                       │
                       ▼
       ┌───────────────────────────────┐
       │     PySpark Data Loader       │
       │   - Deduplication (1,081 rows)│
       │   - Multi-core Engine (12 CPU)│
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Feature Pipeline & Splits   │
       │   - VectorAssembler (30 feats)│
       │   - Train (80%) / Test (20%)  │
       │   - Train (80%) / Val (20%)   │
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │    Distributed Model Suite    │
       │ 1. Random Forest (50 trees)   │
       │ 2. Gradient-Boosted Trees     │
       │ 3. Weighted Logistic Reg.     │
       │ 4. Decision Tree              │
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Validation & Ensemble Tuning│
       │ - Soft Voting (PR/ROC AUC)    │
       │ - Business Threshold Tuning   │
       │ - Automated Model Persistence │
       └───────────────┬───────────────┘
                       │
         ┌─────────────┼─────────────────────────────┐
         ▼             ▼                             ▼
┌────────────────┐ ┌──────────────────────┐ ┌───────────────────────────┐
│  Batch Scoring │ │ Structured Streaming │ │  Real-Time Web Cockpit    │
│- Enriched CSV  │ │- Micro-batch scoring │ │- Live Payment Stream Feed │
│- Power BI Dash │ │- Real-time alerts    │ │- Blind Test Explorer      │
└────────────────┘ └──────────────────────┘ └───────────────────────────┘
```

---

## 📊 Model Performance & Ensemble Results

### Validation Benchmark (Models & Weights)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Ensemble Weight |
|---|---|---|---|---|---|---|
| **Random Forest** | 92.19% | 72.84% | 81.38% | 0.9686 | 0.8124 | **26.25%** |
| **Gradient-Boosted Trees** | 90.48% | 70.37% | 79.17% | 0.9808 | 0.7891 | **26.33%** |
| **Weighted Logistic Reg.** | 6.14% | 92.59% | 11.51% | 0.9766 | 0.6181 | **24.76%** |
| **Decision Tree** | 88.41% | 75.31% | 81.33% | 0.9495 | 0.7600 | **22.65%** |

### Final Ensemble Test Performance (Threshold = 0.24)

- **True Positives (Fraud Caught):** 69
- **False Positives (False Alarms):** 118
- **False Negatives (Missed Fraud):** 16
- **True Negatives:** 56,177
- **Recall:** **81.18%**
- **ROC-AUC:** **0.9747**
- **PR-AUC:** **0.7010**

---

## 💡 Key Engineering Highlights

1. **Solving Extreme Class Imbalance:**
   - Fraud represents only 492 out of 284,807 transactions (0.172%).
   - Traditional classifiers default to guessing 'legitimate' 99.8% of the time. Using inverse frequency weighting (`classWeight = legitimate / fraud`) on Logistic Regression forced the optimizer to heavily penalize fraud errors, achieving **92.59% validation recall**.
2. **Business-Constrained Threshold Optimization:**
   - Missing a fraudulent charge (False Negative) is exponentially more expensive to banks than verifying a false alarm (False Positive).
   - The decision threshold was tuned against validation data with a business constraint of $\ge 85\%$ recall, identifying **0.24** as the optimal operational boundary.
3. **Model Persistence & Instant Inference:**
   - All 4 Spark ML models and the ensemble configuration (`weights`, `threshold`) are serialized to disk (`models/`).
   - Subsequent executions load pre-trained models from disk in **under 20 seconds**, completely eliminating repetitive re-training.
4. **Live Stream Engine & Blind Test Verification:**
   - Supports continuous micro-batch streaming with real-time financial telemetry ($ Cleared vs $ Prevented).
   - Features a Blind Test Explorer that allows testing raw 30-dimensional vectors with zero label knowledge, proving models are genuinely predicting without hardcoding.

---

## 🚀 How to Run

### 1. Requirements & Setup
- Windows 10/11 or Linux
- Java 17 (JDK)
- Python 3.10–3.14 with virtual environment

Activate environment and install dependencies:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run Batch Scoring Pipeline
```powershell
.\.venv\Scripts\python.exe main.py
```
- Automatically checks for saved models in `models/`.
- Loads the models instantly, evaluates 56,000+ test transactions in ~20 seconds, and writes enriched outputs to `ensemble_risk_predictions.csv`.

### 3. Run Real-Time Web Application (Interactive Dashboard)
```powershell
.\.venv\Scripts\python.exe web_app.py
```
*(Or double-click `run_web_app.bat` on Windows)*.
- Automatically launches your browser at `http://127.0.0.1:5000`.
- **Mode 1 (Real-Time Stream Cockpit):** Stream incoming transactions automatically every 1.4s, view live telemetry ($ Cleared vs $ Prevented), and click "Inject Fraud Attack" to observe spontaneous real-time intercepts.
- **Mode 2 (Blind Test Explorer):** Pick any unlabeled transaction ID at random, run model inference blind, and reveal dataset ground truth to prove zero hardcoded logic.

### 4. Run CLI Interactive Demo
```powershell
.\.venv\Scripts\python.exe predict_interactive.py
```
*(Or double-click `run_cli_demo.bat` on Windows)*.
- Interactive terminal menu for quick evaluator demos.

### 5. Run Spark Structured Streaming Engine
Open two terminal windows:

**Terminal 1 (Start Streaming Detector):**
```powershell
.\.venv\Scripts\python.exe -m src.streaming_detector
```

**Terminal 2 (Start Live Transaction Generator):**
```powershell
.\.venv\Scripts\python.exe -m src.stream_producer
```
Ingests streaming micro-batches and exports high-risk alerts to `data/stream_alerts/`.

### 6. Power BI Dashboard
- Open `Fraud_Detection.pbix` in Power BI Desktop.
- Click **Home > Refresh** to update with newly scored transactions.
- Visualize financial loss prevention, operational review queues, and risk tier distributions (*Low*, *Medium*, *High*, *Critical*).
