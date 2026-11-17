import os
import time
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler

from src.data_loader import create_spark_session
from src.model_persistence import load_models, load_ensemble_config
from src.ensemble import (
    generate_ensemble_predictions,
    calculate_ensemble_probability
)


def run_streaming_fraud_detector():
    spark = create_spark_session()

    input_dir = "data/stream_input"
    output_dir = "data/stream_alerts"
    checkpoint_dir = "data/stream_checkpoint"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n========== INITIALIZING REAL-TIME FRAUD DETECTOR ==========")
    print("Loading pre-trained ensemble models from disk...")
    models = load_models()
    weights, ensemble_threshold = load_ensemble_config()

    print("\nLoaded Ensemble Configuration:")
    for model_name, weight in weights.items():
        print(f"  - {model_name}: {weight:.4f}")
    print(f"Operational Decision Threshold: {ensemble_threshold:.4f}\n")

    # Define transaction schema
    fields = [StructField("Time", DoubleType(), True)]
    fields += [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
    fields += [
        StructField("Amount", DoubleType(), True),
        StructField("Class", DoubleType(), True)
    ]
    schema = StructType(fields)

    # Monitor incoming streaming directory for transaction files
    stream_df = (
        spark.readStream
        .schema(schema)
        .option("header", True)
        .csv(input_dir)
    )

    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    stream_ml = assembler.transform(stream_df)
    if "Class" in stream_ml.columns:
        stream_ml = stream_ml.withColumn("label", col("Class").cast("double"))

    def process_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            return

        tx_count = batch_df.count()
        print(f"\n[Micro-Batch {batch_id}] Ingested {tx_count} live transactions. Scoring...")

        preds = generate_ensemble_predictions(batch_df, models, weights)
        preds = calculate_ensemble_probability(preds, weights)

        scored = preds.withColumn(
            "risk_score", col("ensemble_probability") * 100
        ).withColumn(
            "risk_category",
            when(col("risk_score") < 30, "Low")
            .when(col("risk_score") < 60, "Medium")
            .when(col("risk_score") < 80, "High")
            .otherwise("Critical")
        ).withColumn(
            "action",
            when(col("risk_score") >= 80, "BLOCK")
            .when(col("risk_score") >= 50, "FLAG_FOR_REVIEW")
            .otherwise("APPROVE")
        )

        # High-risk alerts
        alerts = scored.filter(col("risk_score") >= 50).select(
            "Time", "Amount", "risk_score", "risk_category", "action"
        )

        alert_count = alerts.count()
        if alert_count > 0:
            print(f"⚠️  ALERT: {alert_count} HIGH-RISK / FRAUD TRANSACTION(S) DETECTED!")
            alerts.show(truncate=False)

            alert_path = os.path.join(
                output_dir, f"alerts_batch_{batch_id}_{int(time.time())}.csv"
            )
            alerts.toPandas().to_csv(alert_path, index=False)
            print(f"Alert written to {alert_path}")
        else:
            print(f"✓ All {tx_count} transactions verified clean (Low/Medium risk).")

    query = (
        stream_ml.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", checkpoint_dir)
        .start()
    )

    print(f"[*] Real-time streaming engine active!")
    print(f"[*] Drop transaction CSV files into: {input_dir}")
    print("[*] Press Ctrl+C to stop.\n")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_fraud_detector()
