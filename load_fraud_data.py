import os

os.environ["PYSPARK_PYTHON"] = os.path.abspath(".venv\\Scripts\\python.exe")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.path.abspath(".venv\\Scripts\\python.exe")

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("FraudDetection")
    .master("local[2]")
    .getOrCreate()
)

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# create Spark
# read CSV

print("Number of rows:", df.count())

df.printSchema()

df.show(5)

df.groupBy("Class").count().show()

print("Null values:")

df.select([
    df[col].isNull().cast("int").alias(col)
    for col in df.columns
]).groupBy().sum().show()

unique_count = df.dropDuplicates().count()

print("Original rows:", df.count())
print("Rows after removing duplicates:", unique_count)
print("Duplicate rows:", df.count() - unique_count)
duplicates = df.groupBy(df.columns).count().filter("count > 1")

# duplicates.groupBy("Class").sum("count").show()

# df.select("Time", "Amount").describe().show()
# df_clean = df.dropDuplicates()

# feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# from pyspark.ml.feature import VectorAssembler

# assembler = VectorAssembler(
#     inputCols=feature_cols,
#     outputCol="features"
# )
# df_features = assembler.transform(df_clean)

# df_ml = df_features.select(
#     "features",
#     df_features["Class"].cast("double").alias("label")
# )

# train_df, test_df = df_ml.randomSplit(
#     [0.8, 0.2],
#     seed=42
# )

# # print("Training rows:", train_df.count())
# # print("Testing rows:", test_df.count())

# # print("Training class distribution:")
# # train_df.groupBy("label").count().show()

# # print("Testing class distribution:")
# # test_df.groupBy("label").count().show()

# from pyspark.ml.classification import LogisticRegression

# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# from pyspark.sql.functions import when

# total = train_df.count()

# class_counts = train_df.groupBy("label").count()

# # class_counts.show()

# train_df = train_df.join(
#     class_counts.withColumnRenamed("count", "class_count"),
#     on="label"
# )

# train_df = train_df.withColumn(
#     "weight",
#     total / (2.0 * train_df["class_count"])
# )

# # train_df.select(
# #     "label",
# #     "class_count",
# #     "weight"
# # ).distinct().show()

# lr_weighted = LogisticRegression(
#     featuresCol="features",
#     labelCol="label",
#     weightCol="weight"
# )
# lr_weighted_model = lr_weighted.fit(train_df)

# # print("WEIGHTED LOGISTIC REGRESSION TRAINING COMPLETED")

# weighted_predictions = lr_weighted_model.transform(test_df)

# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# for score_col in ["rawPrediction", "probability"]:
#     for metric in ["areaUnderROC", "areaUnderPR"]:
#         evaluator = BinaryClassificationEvaluator(
#             labelCol="label",
#             rawPredictionCol=score_col,
#             metricName=metric
#         )

#         score = evaluator.evaluate(weighted_predictions)
#         print(f"Weighted LR | {score_col} | {metric}: {score:.4f}")


# weighted_predictions.select(
#     "label",
#     "prediction",
#     "probability"
# ).show(10, truncate=False)

# weighted_predictions.groupBy(
#     "label",
#     "prediction"
# ).count().orderBy(
#     "label",
#     "prediction"
# ).show()

# weighted_binary_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction"
# )

# weighted_roc_auc = weighted_binary_evaluator.evaluate(
#     weighted_predictions,
#     {weighted_binary_evaluator.metricName: "areaUnderROC"}
# )

# weighted_pr_auc = weighted_binary_evaluator.evaluate(
#     weighted_predictions,
#     {weighted_binary_evaluator.metricName: "areaUnderPR"}
# )

# print("Weighted Logistic Regression ROC-AUC:", weighted_roc_auc)
# print("Weighted Logistic Regression PR-AUC:", weighted_pr_auc)

from pyspark.sql.functions import col

from pyspark.ml.functions import vector_to_array

# threshold_predictions = weighted_predictions.withColumn(
#     "fraud_probability",
#     vector_to_array("probability")[1]
# )
# threshold_predictions.select(
#     "label",
#     "fraud_probability",
#     "prediction"
# ).show(10, truncate=False)

from pyspark.sql.functions import when

# thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]

# for threshold in thresholds:

#     temp = threshold_predictions.withColumn(
#         "threshold_prediction",
#         when(
#             col("fraud_probability") >= threshold,
#             1.0
#         ).otherwise(0.0)
#     )

    # print("\nThreshold:", threshold)

    # temp.groupBy(
    #     "label",
    #     "threshold_prediction"
    # ).count().orderBy(
    #     "label",
    #     "threshold_prediction"
    # ).show()


# from pyspark.ml.classification import DecisionTreeClassifier

# dt = DecisionTreeClassifier(
#     featuresCol="features",
#     labelCol="label",
#     maxDepth=5,
#     seed=42
# )

# dt_model = dt.fit(train_df)

# dt_predictions = dt_model.transform(test_df)

# from pyspark.ml.evaluation import BinaryClassificationEvaluator


# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# dt_roc_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="probability",
#     metricName="areaUnderROC"
# )

# dt_pr_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="probability",
#     metricName="areaUnderPR"
# )

# dt_roc_auc = dt_roc_evaluator.evaluate(dt_predictions)
# dt_pr_auc = dt_pr_evaluator.evaluate(dt_predictions)

# print(f"Decision Tree ROC-AUC: {dt_roc_auc:.4f}")
# print(f"Decision Tree PR-AUC: {dt_pr_auc:.4f}")


# dt_predictions.select(
#     "label",
#     "prediction",
#     "probability"
# ).show(10, truncate=False)

# dt_predictions.groupBy(
#     "label",
#     "prediction"
# ).count().orderBy(
#     "label",
#     "prediction"
# ).show()

# print("DECISION TREE TRAINING COMPLETED")

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# dt_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction"
# )

# dt_roc_auc = dt_evaluator.evaluate(
#     dt_predictions,
#     {dt_evaluator.metricName: "areaUnderROC"}
# )

# dt_pr_auc = dt_evaluator.evaluate(
#     dt_predictions,
#     {dt_evaluator.metricName: "areaUnderPR"}
# )

# print("Decision Tree ROC-AUC:", dt_roc_auc)
# print("Decision Tree PR-AUC:", dt_pr_auc)

from pyspark.ml.classification import RandomForestClassifier
# rf = RandomForestClassifier(
#     featuresCol="features",
#     labelCol="label",
#     numTrees=50,
#     maxDepth=5,
#     seed=42
# )
# rf_model = rf.fit(train_df)

# # # # print("RANDOM FOREST TRAINING COMPLETED")

# rf_predictions = rf_model.transform(test_df)

# rf_predictions.select(
#     "label",
#     "prediction",
#     "probability"
# ).show(10, truncate=False)

# rf_predictions.groupBy(
#     "label",
#     "prediction"
# ).count().orderBy(
#     "label",
#     "prediction"
# ).show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# rf_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction"
# )

# rf_roc_auc = rf_evaluator.evaluate(
#     rf_predictions,
#     {rf_evaluator.metricName: "areaUnderROC"}
# )

# rf_pr_auc = rf_evaluator.evaluate(
#     rf_predictions,
#     {rf_evaluator.metricName: "areaUnderPR"}
# )

# print("Random Forest ROC-AUC:", rf_roc_auc)
# print("Random Forest PR-AUC:", rf_pr_auc)

from pyspark.ml.classification import GBTClassifier

# gbt = GBTClassifier(
#     featuresCol="features",
#     labelCol="label",
#     maxIter=30,
#     maxDepth=3,
#     seed=42
# )

# gbt_model = gbt.fit(train_df)

# # # print("GRADIENT-BOOSTED TREES TRAINING COMPLETED")

# gbt_predictions = gbt_model.transform(test_df)

# gbt_predictions.groupBy(
#     "label",
#     "prediction"
# ).count().orderBy(
#     "label",
#     "prediction"
# ).show()

# gbt_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction"
# )

# gbt_roc_auc = gbt_evaluator.evaluate(
#     gbt_predictions,
#     {gbt_evaluator.metricName: "areaUnderROC"}
# )

# gbt_pr_auc = gbt_evaluator.evaluate(
#     gbt_predictions,
#     {gbt_evaluator.metricName: "areaUnderPR"}
# )

# print("GBT ROC-AUC:", gbt_roc_auc)
# print("GBT PR-AUC:", gbt_pr_auc)

from pyspark.ml.classification import LinearSVC

# svm = LinearSVC(
#     featuresCol="features",
#     labelCol="label",
#     maxIter=50,
#     regParam=0.01
# )

# svm_model = svm.fit(train_df)

# # # print("LINEAR SVM TRAINING COMPLETED")

# svm_predictions = svm_model.transform(test_df)

# svm_predictions.groupBy(
#     "label",
#     "prediction"
# ).count().orderBy(
#     "label",
#     "prediction"
# ).show()

# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# svm_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction"
# )

# svm_roc_auc = svm_evaluator.evaluate(
#     svm_predictions,
#     {svm_evaluator.metricName: "areaUnderROC"}
# )

# svm_pr_auc = svm_evaluator.evaluate(
#     svm_predictions,
#     {svm_evaluator.metricName: "areaUnderPR"}
# )

# print("Linear SVM ROC-AUC:", svm_roc_auc)
# print("Linear SVM PR-AUC:", svm_pr_auc)


from pyspark.sql.functions import col, sum as spark_sum
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# models = [
#     ("Logistic Regression", weighted_predictions),
#     ("Decision Tree", dt_predictions),
#     ("Random Forest", rf_predictions),
#     ("Gradient-Boosted Trees", gbt_predictions),
#     ("Linear SVM", svm_predictions)
# ]

# results = []

# for model_name, model_predictions in models:
#     counts = model_predictions.agg(
#         spark_sum(
#             ((col("label") == 1.0) & (col("prediction") == 1.0)).cast("int")
#         ).alias("TP"),
#         spark_sum(
#             ((col("label") == 0.0) & (col("prediction") == 1.0)).cast("int")
#         ).alias("FP"),
#         spark_sum(
#             ((col("label") == 1.0) & (col("prediction") == 0.0)).cast("int")
#         ).alias("FN"),
#         spark_sum(
#             ((col("label") == 0.0) & (col("prediction") == 0.0)).cast("int")
#         ).alias("TN")
#     ).first()

#     tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

#     precision = tp / (tp + fp) if tp + fp else 0.0
#     recall = tp / (tp + fn) if tp + fn else 0.0
#     f1 = (
#         2 * precision * recall / (precision + recall)
#         if precision + recall else 0.0
#     )

# #     
#     score_col = (
#         "rawPrediction"
#         if model_name == "Linear SVM"
#         else "probability"
#     )

#     roc_auc = BinaryClassificationEvaluator(
#         labelCol="label",
#         rawPredictionCol=score_col,
#         metricName="areaUnderROC"
#     ).evaluate(model_predictions)

#     pr_auc = BinaryClassificationEvaluator(
#         labelCol="label",
#         rawPredictionCol=score_col,
#         metricName="areaUnderPR"
#     ).evaluate(model_predictions)

#     results.append((
#         model_name, tp, fp, fn, tn,
#         precision, recall, f1, roc_auc, pr_auc
#     ))

# print("\nMODEL COMPARISON")
# print("-" * 115)
# print(
#     f"{'Model':25} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>7} "
#     f"{'Precision':>11} {'Recall':>9} {'F1':>9} "
#     f"{'ROC-AUC':>9} {'PR-AUC':>9}"
# )

# for row in results:
#     name, tp, fp, fn, tn, precision, recall, f1, roc_auc, pr_auc = row
#     print(
#         f"{name:25} {tp:5} {fp:5} {fn:5} {tn:7} "
#         f"{precision:10.2%} {recall:8.2%} {f1:8.2%} "
#         f"{roc_auc:9.4f} {pr_auc:9.4f}"
#     )


spark.stop()