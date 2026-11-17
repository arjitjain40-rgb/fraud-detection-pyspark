from src.data_loader import create_spark_session
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.sql import Row
import os
import shutil

spark = create_spark_session()

try:
    data = [
        Row(features=[0.0, 1.0], label=0.0),
        Row(features=[1.0, 0.0], label=1.0),
        Row(features=[0.0, 0.5], label=0.0),
        Row(features=[1.0, 0.5], label=1.0),
    ]

    from pyspark.ml.linalg import Vectors

    df = spark.createDataFrame([
        (Vectors.dense(r.features), r.label) for r in data
    ], ["features", "label"])

    model = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=2
    ).fit(df)

    path = "models/persistence_test"

    model.write().overwrite().save(path)
    print("MODEL SAVED SUCCESSFULLY")

    loaded_model = DecisionTreeClassificationModel.load(path)
    print("MODEL LOADED SUCCESSFULLY")

finally:
    spark.stop()