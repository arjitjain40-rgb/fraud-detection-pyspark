# Real-Time Big Data Credit Card Fraud Detection System

An enterprise-grade, distributed credit card fraud detection system built using **Apache Spark (PySpark MLlib)**, **Spark Structured Streaming**, and **Power BI**. 

The system handles extreme class imbalance (0.172% fraud incidence) through cost-sensitive class weighting, employs a performance-weighted ensemble (combining PR-AUC and ROC-AUC metrics), tunes classification thresholds under real-world business constraints ($\ge 85\%$ recall), and supports both offline batch analytics and real-time streaming inference.

---

## 🏗️ Architecture & Pipeline Flow

```
[Kaggle Credit Card Data (284,807 transactions)]
                       │
                       ▼
       ┌───────────────────────────────┐
       │     PySpark Data Loader       │
       │   - Deduplication (1,081 rows)│
       │   - Schema Ingestion          │
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
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  Batch Scoring   │       │ Structured Streaming │
│- Enriched CSV    │       │- Micro-batch scoring │
│- Power BI (.pbix)│       │- Real-time alerts    │
└──────────────────┘       └──────────────────────┘
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
   - Fraud represents only 492 out of 284,807 transactions.
   - Traditional classifiers default to predicting the majority class. Using inverse class weighting (`legitimate_count / fraud_count`) for Logistic Regression forced the optimizer to prioritize minority class detection.
2. **Business-Constrained Threshold Optimization:**
   - In fraud detection, missing a fraud case (False Negative) is exponentially more expensive than reviewing a legitimate transaction (False Positive).
   - The threshold is tuned on validation data with a hard constraint of $\ge 85\%$ recall, picking the threshold that maximizes precision under that boundary.
3. **Model Persistence & Instant Inference:**
   - Models are serialized to disk (`models/`) along with ensemble weights and optimal threshold (`ensemble_config.json`).
   - Subsequent executions load in **under 20 seconds** instead of minutes of re-training.
4. **Enriched Reporting for Power BI:**
   - Export preserves `Time` and `Amount` alongside predictions and risk tiers (*Low*, *Medium*, *High*, *Critical*), allowing financial impact dashboards (*Total Dollars Saved* vs *Friction Cost*).

---

## 🚀 How to Run

### 1. Requirements & Setup
- Windows 10/11 or Linux
- Java 17 (JDK)
- Python 3.10–3.14 with virtual environment

Activate environment:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run Batch Pipeline
```powershell
.\.venv\Scripts\python.exe main.py
```
- First run trains and persists models to `models/`.
- Subsequent runs load models instantly and score test data into `ensemble_risk_predictions.csv`.

### 3. Run Web Application (Interactive Presentation UI)
```powershell
.\.venv\Scripts\python.exe web_app.py
```
- Opens a modern browser dashboard at `http://127.0.0.1:5000`.
- Allows 1-click testing of live fraud scenarios, custom transaction amounts, real-time risk gauges, multi-model voting breakdowns, and audit history.

### 4. Run Real-Time Streaming Detection
Open two terminal windows:

**Terminal 1 (Start Streaming Detector):**
```powershell
.\.venv\Scripts\python.exe -m src.streaming_detector
```

**Terminal 2 (Start Live Transaction Generator):**
```powershell
.\.venv\Scripts\python.exe -m src.stream_producer
```
You will observe live micro-batches being ingested and flagged in real-time as high-risk alerts.

### 4. Power BI Dashboard
- Open `Fraud_Detection.pbix` in Power BI Desktop.
- Click **Home > Refresh** to pull in the enriched dataset.
