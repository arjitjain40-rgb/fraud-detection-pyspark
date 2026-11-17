
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, sum as spark_sum, udf, when
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType

def evaluate_model(model, test_df, model_name):
    # Generate predictions using the trained model
    predictions = model.transform(test_df)


    counts = predictions.agg(
        spark_sum(
            ((col("label") == 1) & (col("prediction") == 1)).cast("int")
        ).alias("TP"),
        spark_sum(
            ((col("label") == 0) & (col("prediction") == 1)).cast("int")
        ).alias("FP"),
        spark_sum(
            ((col("label") == 1) & (col("prediction") == 0)).cast("int")
        ).alias("FN"),
        spark_sum(
            ((col("label") == 0) & (col("prediction") == 0)).cast("int")
        ).alias("TN")
    ).first()

    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print(f"Precision: {precision:.4%}")
    print(f"Recall: {recall:.4%}")
    print(f"F1-score: {f1:.4%}")

    # Display actual labels versus predicted labels
    predictions.groupBy(
        "label", "prediction"
    ).count().orderBy(
        "label", "prediction"
    ).show()

    # Evaluate ROC-AUC and PR-AUC
    # Choose the correct score column for AUC evaluation
    score_column = (
        "rawPrediction"
        if model_name == "Linear SVM"
        else "probability"
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol=score_column
    )

    roc_auc = evaluator.evaluate(
        predictions,
        {evaluator.metricName: "areaUnderROC"}
    )

    pr_auc = evaluator.evaluate(
        predictions,
        {evaluator.metricName: "areaUnderPR"}
    )

    return {
    "Model": model_name,
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc
}



def select_best_threshold(model, validation_df):

    predictions = model.transform(validation_df)

    get_fraud_probability = udf(
        lambda probability: float(probability[1]),
        DoubleType()
    )

    predictions = predictions.withColumn(
        "fraud_probability",
        get_fraud_probability(col("probability"))
    )

    total_frauds = predictions.filter(col("label") == 1).count()

    thresholds = [round(x * 0.01, 2) for x in range(50, 100, 2)]

    best_threshold = None
    best_precision = -1
    best_recall = 0

    print("\nThreshold tuning (minimum recall: 85%)")
    print("-" * 55)

    for threshold in thresholds:
        evaluated = predictions.withColumn(
            "prediction",
            (col("fraud_probability") >= threshold).cast("double")
        )

        counts = evaluated.agg(
            spark_sum(
                ((col("label") == 1) & (col("prediction") == 1)).cast("int")
            ).alias("TP"),
            spark_sum(
                ((col("label") == 0) & (col("prediction") == 1)).cast("int")
            ).alias("FP"),
            spark_sum(
                ((col("label") == 1) & (col("prediction") == 0)).cast("int")
            ).alias("FN")
        ).first()

        tp = counts["TP"] or 0
        fp = counts["FP"] or 0
        fn = counts["FN"] or 0

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        # print(
        #     f"Threshold: {threshold:.2f} | "
        #     f"Precision: {precision:.4f} | "
        #     f"Recall: {recall:.4f}"
        # )

        if recall >= 0.85 and precision > best_precision:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    if best_threshold is None:
        raise ValueError(
            "No tested threshold achieved the minimum recall of 85%."
        )

    print(f"\nSelected threshold: {best_threshold:.2f}")
    print(f"Validation precision: {best_precision:.4f}")
    print(f"Validation recall: {best_recall:.4f}")

    return best_threshold




def add_risk_score(predictions):
    predictions = predictions.withColumn(
        "fraud_probability_pct",
        vector_to_array(col("probability"))[1] * 100
    )

    predictions = predictions.withColumn(
        "risk_category",
        when(col("fraud_probability_pct") < 30, "Low")
        .when(col("fraud_probability_pct") < 60, "Medium")
        .when(col("fraud_probability_pct") < 80, "High")
        .otherwise("Critical")
    )

    return predictions


def select_ensemble_threshold(ensemble_predictions):

    total_frauds = ensemble_predictions.filter(
        col("label") == 1
    ).count()

    thresholds = [
        round(x * 0.01, 2)
        for x in range(1, 100)
    ]

    best_threshold = None
    best_precision = -1
    best_recall = 0

    print("\nEnsemble threshold tuning (minimum recall: 85%)")

    for threshold in thresholds:

        evaluated = ensemble_predictions.withColumn(
            "ensemble_prediction",
            (col("ensemble_probability") >= threshold).cast("double")
        )

        counts = evaluated.agg(
            spark_sum(
                ((col("label") == 1) &
                 (col("ensemble_prediction") == 1)).cast("int")
            ).alias("TP"),

            spark_sum(
                ((col("label") == 0) &
                 (col("ensemble_prediction") == 1)).cast("int")
            ).alias("FP"),

            spark_sum(
                ((col("label") == 1) &
                 (col("ensemble_prediction") == 0)).cast("int")
            ).alias("FN")
        ).first()

        tp = counts["TP"] or 0
        fp = counts["FP"] or 0
        fn = counts["FN"] or 0

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        if recall >= 0.85 and precision > best_precision:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    if best_threshold is None:
        raise ValueError(
            "No threshold achieved the minimum recall of 85%."
        )

    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation precision: {best_precision:.4f}")
    print(f"Validation recall: {best_recall:.4f}")

    return best_threshold

