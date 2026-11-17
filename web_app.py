import os
import sys
import json
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

from src.data_loader import create_spark_session
from src.model_persistence import load_models, load_ensemble_config
from src.ensemble import generate_ensemble_predictions, calculate_ensemble_probability

# Global references
SPARK = None
MODELS = None
WEIGHTS = None
THRESHOLD = None
FULL_DF = None
FRAUD_DF = None
NORMAL_DF = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FraudShield | Big Data Fraud Detection</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.15);
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.15);
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.15);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
    body { background: radial-gradient(circle at 50% 0%, #172554 0%, #0b0f19 50%, #030712 100%); color: var(--text); min-height: 100vh; padding: 24px; }
    .container { max-width: 1200px; margin: 0 auto; }
    
    /* Header */
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 24px; border-bottom: 1px solid var(--card-border); margin-bottom: 24px; }
    .logo-area { display: flex; align-items: center; gap: 14px; }
    .logo-icon { width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #2563eb, #06b6d4); display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4); }
    h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
    .subtitle { font-size: 13px; color: var(--text-muted); }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border); display: flex; align-items: center; gap: 6px; }
    .chip-pulse { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

    /* Layout */
    .grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; margin-bottom: 24px; }
    @media(max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border); border-radius: 16px; padding: 24px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4); }
    .card-title { font-size: 16px; font-weight: 600; margin-bottom: 18px; display: flex; align-items: center; gap: 8px; color: #e5e7eb; }
    
    /* Presets */
    .scenario-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
    .btn-scenario { background: rgba(255, 255, 255, 0.04); border: 1px solid var(--card-border); border-radius: 12px; padding: 12px; text-align: left; color: var(--text); cursor: pointer; transition: all 0.2s; font-size: 13px; }
    .btn-scenario:hover { background: rgba(59, 130, 246, 0.15); border-color: rgba(59, 130, 246, 0.4); transform: translateY(-2px); }
    .btn-scenario .scenario-tag { font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 4px; }
    .tag-fraud { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    .tag-legit { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    
    /* Form */
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 500; color: var(--text-muted); margin-bottom: 6px; }
    input[type="number"], select { width: 100%; padding: 12px 14px; background: rgba(0, 0, 0, 0.3); border: 1px solid var(--card-border); border-radius: 10px; color: #fff; font-size: 15px; outline: none; transition: border-color 0.2s; }
    input[type="number"]:focus, select:focus { border-color: var(--accent-blue); }
    
    .btn-submit { width: 100%; padding: 14px; border-radius: 10px; background: linear-gradient(135deg, #2563eb, #1d4ed8); border: none; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 8px; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4); transition: all 0.2s; }
    .btn-submit:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); transform: translateY(-1px); }
    .btn-submit:disabled { opacity: 0.6; cursor: not-allowed; }

    /* Result Panel */
    .result-box { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 20px 0; }
    .score-circle { width: 140px; height: 140px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 6px solid var(--card-border); margin-bottom: 16px; position: relative; transition: all 0.4s; }
    .score-val { font-size: 38px; font-weight: 800; line-height: 1; }
    .score-lbl { font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-top: 4px; }
    
    .decision-badge { padding: 10px 20px; border-radius: 999px; font-weight: 700; font-size: 14px; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 8px; margin-bottom: 16px; }
    
    .truth-status { font-size: 12px; color: var(--text-muted); margin-bottom: 20px; padding: 6px 14px; background: rgba(255, 255, 255, 0.04); border-radius: 8px; }

    /* Model Breakdown Bars */
    .breakdown-title { width: 100%; text-align: left; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
    .model-row { width: 100%; margin-bottom: 10px; }
    .model-info { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
    .model-bar-bg { width: 100%; height: 7px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; }
    .model-bar-fill { height: 100%; border-radius: 999px; width: 0%; transition: width 0.6s ease-out; }
    
    /* History Table */
    .history-card { margin-top: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 12px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--card-border); font-size: 12px; text-transform: uppercase; }
    td { padding: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
    .pill { padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; display: inline-block; }
    .pill-red { background: var(--danger-bg); color: #f87171; }
    .pill-green { background: var(--success-bg); color: #34d399; }
    .pill-yellow { background: var(--warning-bg); color: #fbbf24; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-area">
        <div class="logo-icon">🛡️</div>
        <div>
          <h1>FraudShield AI</h1>
          <div class="subtitle">Distributed Real-Time Fraud Detection Engine (PySpark MLlib)</div>
        </div>
      </div>
      <div class="chips">
        <div class="chip"><div class="chip-pulse"></div> Apache Spark MLlib</div>
        <div class="chip">Threshold: 24.0%</div>
        <div class="chip">Recall: 81.18%</div>
        <div class="chip">ROC-AUC: 0.9747</div>
      </div>
    </header>

    <div class="grid">
      <!-- Input Column -->
      <div class="card">
        <div class="card-title"><span>⚡</span> Quick Demonstration Scenarios</div>
        <div class="scenario-grid">
          <button class="btn-scenario" onclick="selectScenario('fraud1')">
            <span class="scenario-tag tag-fraud">Fraud Demo 1</span>
            <div><b>Blatant Stolen Card</b></div>
            <div style="color:var(--text-muted); font-size:11px;">$188.78 &bull; Critical (~91% Block)</div>
          </button>
          <button class="btn-scenario" onclick="selectScenario('legit1')">
            <span class="scenario-tag tag-legit">Legitimate 1</span>
            <div><b>Grocery Store POS</b></div>
            <div style="color:var(--text-muted); font-size:11px;">$29.89 &bull; Normal (~1% Approve)</div>
          </button>
          <button class="btn-scenario" onclick="selectScenario('fraud2')">
            <span class="scenario-tag tag-fraud" style="background:rgba(245,158,11,0.2); color:#fbbf24;">Fraud Demo 2</span>
            <div><b>Subtle Sneak Fraud</b></div>
            <div style="color:var(--text-muted); font-size:11px;">$450.00 &bull; Rescued by Weighted LR</div>
          </button>
          <button class="btn-scenario" onclick="selectScenario('random')">
            <span class="scenario-tag" style="background:rgba(59,130,246,0.2); color:#60a5fa;">Random Test</span>
            <div><b>Random Sample</b></div>
            <div style="color:var(--text-muted); font-size:11px;">From 284k Kaggle rows</div>
          </button>
        </div>

        <div class="card-title" style="margin-top: 10px;"><span>⚙️</span> Custom Transaction Parameters</div>
        <div class="form-group">
          <label>Transaction Amount (USD $)</label>
          <input type="number" id="txAmount" value="450.00" step="0.01" min="0.01">
        </div>
        <div class="form-group">
          <label>Behavioral Risk Profile</label>
          <select id="txProfile">
            <option value="fraud">Abnormal Location & Rapid Sequence (Simulated Fraud)</option>
            <option value="normal">Consistent Device & Local Merchant (Simulated Legitimate)</option>
          </select>
        </div>
        <button class="btn-submit" id="btnAnalyze" onclick="analyzeTransaction()">
          <span>🔍</span> Analyze Transaction with Ensemble
        </button>
      </div>

      <!-- Result Column -->
      <div class="card" id="resultCard">
        <div class="card-title"><span>📊</span> Real-Time Risk Intelligence</div>
        <div class="result-box">
          <div class="score-circle" id="circleMeter">
            <div class="score-val" id="scoreVal">--</div>
            <div class="score-lbl">Risk Score</div>
          </div>
          <div class="decision-badge" id="decisionBadge">Awaiting Transaction</div>
          <div class="truth-status" id="truthStatus">Ground Truth in Dataset: None selected yet</div>

          <div class="breakdown-title">Multi-Model Committee Voting</div>
          
          <div class="model-row">
            <div class="model-info">
              <span>Random Forest (Weight 26.25%)</span>
              <span id="rfPct">0%</span>
            </div>
            <div class="model-bar-bg"><div class="model-bar-fill" id="rfBar" style="background:#3b82f6;"></div></div>
          </div>

          <div class="model-row">
            <div class="model-info">
              <span>Gradient-Boosted Trees (Weight 26.33%)</span>
              <span id="gbtPct">0%</span>
            </div>
            <div class="model-bar-bg"><div class="model-bar-fill" id="gbtBar" style="background:#06b6d4;"></div></div>
          </div>

          <div class="model-row">
            <div class="model-info">
              <span>Weighted Logistic Regression (Weight 24.76%)</span>
              <span id="lrPct">0%</span>
            </div>
            <div class="model-bar-bg"><div class="model-bar-fill" id="lrBar" style="background:#8b5cf6;"></div></div>
          </div>

          <div class="model-row">
            <div class="model-info">
              <span>Decision Tree (Weight 22.65%)</span>
              <span id="dtPct">0%</span>
            </div>
            <div class="model-bar-bg"><div class="model-bar-fill" id="dtBar" style="background:#ec4899;"></div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- History Log Table -->
    <div class="card history-card">
      <div class="card-title"><span>🕒</span> Live Audit Trail (Recent Evaluations)</div>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Amount ($)</th>
            <th>Ensemble Probability</th>
            <th>Risk Score</th>
            <th>Tier</th>
            <th>Automated Decision</th>
          </tr>
        </thead>
        <tbody id="auditTable">
          <tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:18px;">No transactions analyzed in this session yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    let activePreset = null;

    function selectScenario(type) {
      activePreset = type;
      if (type === 'fraud1') {
        document.getElementById('txAmount').value = '188.78';
        document.getElementById('txProfile').value = 'fraud';
      } else if (type === 'legit1') {
        document.getElementById('txAmount').value = '29.89';
        document.getElementById('txProfile').value = 'normal';
      } else if (type === 'fraud2') {
        document.getElementById('txAmount').value = '450.00';
        document.getElementById('txProfile').value = 'fraud';
      } else if (type === 'random') {
        document.getElementById('txAmount').value = '120.00';
      }
      analyzeTransaction();
    }

    async function analyzeTransaction() {
      const btn = document.getElementById('btnAnalyze');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳</span> Computing Spark Ensemble...';

      const payload = {
        preset: activePreset,
        amount: parseFloat(document.getElementById('txAmount').value) || 100.0,
        profile: document.getElementById('txProfile').value
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert('Error connecting to backend: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>🔍</span> Analyze Transaction with Ensemble';
        activePreset = null; // reset
      }
    }

    function renderResult(data) {
      const score = data.risk_score;
      const meter = document.getElementById('circleMeter');
      const badge = document.getElementById('decisionBadge');
      const scoreVal = document.getElementById('scoreVal');

      scoreVal.innerText = score.toFixed(1);

      if (score >= 80) {
        meter.style.borderColor = '#ef4444';
        badge.style.background = 'rgba(239, 68, 68, 0.2)';
        badge.style.color = '#f87171';
        badge.innerHTML = '🚨 ACTION: BLOCK TRANSACTION';
      } else if (score >= 24) {
        meter.style.borderColor = '#f59e0b';
        badge.style.background = 'rgba(245, 158, 11, 0.2)';
        badge.style.color = '#fbbf24';
        badge.innerHTML = '⚠️ ACTION: OTP / MANUAL REVIEW';
      } else {
        meter.style.borderColor = '#10b981';
        badge.style.background = 'rgba(16, 185, 129, 0.2)';
        badge.style.color = '#34d399';
        badge.innerHTML = '✅ ACTION: APPROVE TRANSACTION';
      }

      // Truth
      const truth = document.getElementById('truthStatus');
      const isTruth = data.actual_label === 1 ? 'FRAUDULENT' : 'LEGITIMATE';
      const accurate = (data.ensemble_probability >= 0.24 && data.actual_label === 1) || (data.ensemble_probability < 0.24 && data.actual_label === 0);
      truth.innerHTML = `Ground Truth Label: <b>${isTruth}</b> &bull; Ensemble Verdict: <b style="color:${accurate ? '#34d399' : '#f87171'}">${accurate ? 'ACCURATE' : 'MISCLASSIFIED'}</b>`;

      // Model Bars
      setBar('rf', data.prob_rf * 100);
      setBar('gbt', data.prob_gbt * 100);
      setBar('lr', data.prob_lr * 100);
      setBar('dt', data.prob_dt * 100);

      // Audit Log Table
      addToAudit(data);
    }

    function setBar(id, val) {
      document.getElementById(id + 'Pct').innerText = val.toFixed(1) + '%';
      document.getElementById(id + 'Bar').style.width = Math.min(100, Math.max(0, val)) + '%';
    }

    function addToAudit(data) {
      const table = document.getElementById('auditTable');
      if (table.innerHTML.includes('No transactions analyzed')) {
        table.innerHTML = '';
      }
      const timeStr = new Date().toLocaleTimeString();
      const pillClass = data.risk_score >= 80 ? 'pill-red' : (data.risk_score >= 24 ? 'pill-yellow' : 'pill-green');
      const decisionText = data.risk_score >= 80 ? 'BLOCK' : (data.risk_score >= 24 ? 'FLAGGED' : 'APPROVED');

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${timeStr}</td>
        <td>$${data.amount.toFixed(2)}</td>
        <td>${(data.ensemble_probability * 100).toFixed(2)}%</td>
        <td><b>${data.risk_score.toFixed(1)}</b> / 100</td>
        <td><span class="pill ${pillClass}">${data.risk_category}</span></td>
        <td><b>${decisionText}</b></td>
      `;
      table.prepend(tr);
    }
  </script>
</body>
</html>
"""


class FraudRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet down server logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            params = json.loads(body) if body else {}

            preset = params.get("preset")
            amount = float(params.get("amount", 100.0))
            profile = params.get("profile", "normal")

            # Select sample row
            if preset == "fraud1":
                # Blatant high-confidence fraud case (All models agree, score ~91.2% RED BLOCK)
                sample = FULL_DF.loc[[10891]].copy()
                actual_label = 1
            elif preset == "fraud2":
                # Subtle / borderline fraud (Rescued by Weighted LR at 99.1%, overall ~25.9% REVIEW)
                sample = FRAUD_DF.sample(n=1, random_state=42).copy()
                sample["Amount"] = 450.0
                actual_label = 1
            elif preset == "legit1":
                # Clean legitimate purchase (Score ~1.2% GREEN APPROVE)
                sample = FULL_DF.loc[[40]].copy()
                actual_label = 0
            elif preset == "random":
                sample = FULL_DF.sample(n=1).copy()
                actual_label = int(sample["Class"].values[0])
            elif profile == "fraud":
                sample = FULL_DF.loc[[10891]].copy()
                sample["Amount"] = amount
                actual_label = 1
            else:
                sample = FULL_DF.loc[[40]].copy()
                sample["Amount"] = amount
                actual_label = 0

            # Run Spark transformation & ensemble
            spark_df = SPARK.createDataFrame(sample)
            feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df_ml = assembler.transform(spark_df)

            preds = generate_ensemble_predictions(df_ml, MODELS, WEIGHTS)
            preds = calculate_ensemble_probability(preds, WEIGHTS)

            row = preds.toPandas().iloc[0].to_dict()

            ens_prob = float(row.get("ensemble_probability", 0.0))
            risk_score = ens_prob * 100

            if risk_score >= 80:
                category = "CRITICAL"
                action = "BLOCK"
            elif risk_score >= 24:
                category = "MEDIUM / HIGH"
                action = "REVIEW"
            else:
                category = "LOW"
                action = "APPROVE"

            response = {
                "amount": float(sample["Amount"].values[0]),
                "time": float(sample["Time"].values[0]),
                "ensemble_probability": ens_prob,
                "risk_score": risk_score,
                "risk_category": category,
                "action": action,
                "actual_label": actual_label,
                "prob_rf": float(row.get("prob_random_forest", 0.0)),
                "prob_gbt": float(row.get("prob_gradient_boosted_trees", 0.0)),
                "prob_lr": float(row.get("prob_weighted_logistic_regression", 0.0)),
                "prob_dt": float(row.get("prob_decision_tree", 0.0))
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=5000):
    global SPARK, MODELS, WEIGHTS, THRESHOLD, FULL_DF, FRAUD_DF, NORMAL_DF

    print("\n" + "=" * 65)
    print("      🚀 INITIALIZING FRAUDSHIELD AI WEB APPLICATION 🚀")
    print("=" * 65)
    print("[*] Starting Apache Spark session...")
    SPARK = create_spark_session()

    print("[*] Loading pre-trained models from disk...")
    MODELS = load_models()
    WEIGHTS, THRESHOLD = load_ensemble_config()

    print("[*] Loading transaction samples from data/creditcard.csv...")
    FULL_DF = pd.read_csv(os.path.join("data", "creditcard.csv"))
    FRAUD_DF = FULL_DF[FULL_DF["Class"] == 1].reset_index(drop=True)
    NORMAL_DF = FULL_DF[FULL_DF["Class"] == 0].reset_index(drop=True)

    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, FraudRequestHandler)

    url = f"http://127.0.0.1:{port}"
    print(f"\n[SUCCESS] Web application is live at: {url}")
    print("[*] Opening your web browser automatically...")
    print("[*] Press Ctrl+C in this terminal to stop the server.\n")

    # Automatically launch the user's browser
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FraudShield Web Server...")
    finally:
        SPARK.stop()
        print("Spark stopped. Server closed.")


if __name__ == "__main__":
    start_server()
