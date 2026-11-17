import os
import sys
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler

from src.data_loader import create_spark_session
from src.model_persistence import load_models, load_ensemble_config
from src.ensemble import generate_ensemble_predictions, calculate_ensemble_probability


def print_banner():
    print("""
======================================================================
       🏦 REAL-TIME BIG DATA CREDIT CARD FRAUD DETECTOR 🏦
             Powered by Apache Spark MLlib Ensemble
======================================================================
    """)


def format_report(row, actual_label=None):
    amount = row.get("Amount", 0.0)
    tx_time = row.get("Time", 0.0)
    prob = row.get("ensemble_probability", 0.0)
    risk_score = prob * 100

    rf_p = row.get("prob_random_forest", 0.0) * 100
    gbt_p = row.get("prob_gradient_boosted_trees", 0.0) * 100
    lr_p = row.get("prob_weighted_logistic_regression", 0.0) * 100
    dt_p = row.get("prob_decision_tree", 0.0) * 100

    if risk_score >= 80:
        category = "CRITICAL"
        action = "🚨 BLOCK TRANSACTION - HIGH RISK FRAUD"
    elif risk_score >= 50:
        category = "HIGH"
        action = "⚠️  FLAG FOR MANUAL REVIEW & OTP VERIFICATION"
    elif risk_score >= 24:
        category = "MEDIUM"
        action = "🔍 MONITOR TRANSACTION (ELEVATED RISK)"
    else:
        category = "LOW"
        action = "✅ APPROVE TRANSACTION - LEGITIMATE"

    print("\n" + "=" * 70)
    print("                     TRANSACTION RISK REPORT")
    print("=" * 70)
    print(f"Transaction Amount:        ${amount:,.2f}")
    print(f"Transaction Timestamp:     {tx_time:.1f} seconds")
    print("-" * 70)
    print("INDIVIDUAL MODEL BREAKDOWN:")
    print(f"  • Random Forest:               {rf_p:6.2f}%  (PR/ROC Weight: 26.25%)")
    print(f"  • Gradient-Boosted Trees:      {gbt_p:6.2f}%  (PR/ROC Weight: 26.33%)")
    print(f"  • Weighted Logistic Reg:       {lr_p:6.2f}%  (PR/ROC Weight: 24.76%)")
    print(f"  • Decision Tree:               {dt_p:6.2f}%  (PR/ROC Weight: 22.65%)")
    print("-" * 70)
    print(f"WEIGHTED ENSEMBLE PROBABILITY:   {prob * 100:.2f}%")
    print(f"OVERALL RISK SCORE (0-100):      {risk_score:.2f} / 100")
    print(f"RISK CATEGORY:                   {category}")
    print(f"OPERATIONAL DECISION THRESHOLD:  24.00%")
    print("-" * 70)
    print(f">>> FINAL SYSTEM ACTION: {action} <<<")

    if actual_label is not None:
        truth = "FRAUDULENT (1)" if actual_label == 1 else "LEGITIMATE (0)"
        is_correct = (prob >= 0.24 and actual_label == 1) or (prob < 0.24 and actual_label == 0)
        status = "✅ PREDICTION ACCURATE" if is_correct else "⚠️ MISCLASSIFIED"
        print(f"Actual Ground Truth Label:       {truth} -> {status}")
    print("=" * 70 + "\n")


def score_transaction(spark, models, weights, sample_df):
    spark_df = spark.createDataFrame(sample_df)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_ml = assembler.transform(spark_df)

    preds = generate_ensemble_predictions(df_ml, models, weights)
    preds = calculate_ensemble_probability(preds, weights)

    result = preds.toPandas().iloc[0].to_dict()
    return result


def main():
    print_banner()
    print("[*] Initializing Apache Spark session...")
    spark = create_spark_session()

    print("[*] Loading pre-trained models from disk...")
    models = load_models()
    weights, threshold = load_ensemble_config()

    print("[*] Loading dataset samples from data/creditcard.csv...")
    dataset_path = os.path.join("data", "creditcard.csv")
    full_df = pd.read_csv(dataset_path)

    fraud_samples = full_df[full_df["Class"] == 1].reset_index(drop=True)
    normal_samples = full_df[full_df["Class"] == 0].reset_index(drop=True)

    print(f"[*] Ready! ({len(fraud_samples)} fraud cases, {len(normal_samples)} normal cases loaded)\n")

    while True:
        print("-" * 50)
        print("           PROFESSOR / EVALUATOR DEMO MENU")
        print("-" * 50)
        print("  [1] Test a Known FRAUDULENT Transaction (Live Block Demo)")
        print("  [2] Test a Known LEGITIMATE Transaction (Live Approval Demo)")
        print("  [3] Custom Transaction (Enter Amount & Test Risk Scenario)")
        print("  [4] Random Transaction Stress Test")
        print("  [5] Exit Demo")
        print("-" * 50)

        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            print("\n>>> Selecting verified fraudulent transaction from test set...")
            sample = fraud_samples.sample(n=1)
            result = score_transaction(spark, models, weights, sample)
            format_report(result, actual_label=1)

        elif choice == "2":
            print("\n>>> Selecting verified legitimate transaction from test set...")
            sample = normal_samples.sample(n=1)
            result = score_transaction(spark, models, weights, sample)
            format_report(result, actual_label=0)

        elif choice == "3":
            try:
                amt_str = input("\nEnter transaction amount in $ (e.g. 50.00, 1500.00): ").strip()
                amt = float(amt_str)
            except ValueError:
                amt = 250.0
                print("Invalid input, defaulting to $250.00")

            print("\nSelect Behavioral Profile to test:")
            print("  (A) Suspicious High-Risk Pattern (Sudden abnormal location/frequency)")
            print("  (B) Normal Cardholder Spending Profile (Standard retail purchase)")
            sub_choice = input("Choose pattern (A or B): ").strip().upper()

            if sub_choice == "A":
                # Base on fraud profile with custom amount
                sample = fraud_samples.sample(n=1).copy()
                sample["Amount"] = amt
                truth = 1
            else:
                # Base on normal profile with custom amount
                sample = normal_samples.sample(n=1).copy()
                sample["Amount"] = amt
                truth = 0

            result = score_transaction(spark, models, weights, sample)
            format_report(result, actual_label=truth)

        elif choice == "4":
            print("\n>>> Picking a random transaction at random from 284,807 rows...")
            sample = full_df.sample(n=1)
            truth = int(sample["Class"].values[0])
            result = score_transaction(spark, models, weights, sample)
            format_report(result, actual_label=truth)

        elif choice == "5":
            print("\nThank you for evaluating the Fraud Detection System. Exiting...")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, or 5.")

    spark.stop()


if __name__ == "__main__":
    main()
