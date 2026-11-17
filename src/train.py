
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC


def train_random_forest(train_df):
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=50,
        maxDepth=5,
        seed=42
    )

    rf_model = rf.fit(train_df.select("features", "label"))

    return rf_model

def train_logistic_regression(train_df):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100
    )

    lr_model = lr.fit(train_df.select("features", "label"))
    return lr_model

def train_weighted_logistic_regression(train_df):
    # Count each class in the training data
    class_counts = {
        row["label"]: row["count"]
        for row in train_df.groupBy("label").count().collect()
    }

    fraud_count = class_counts.get(1.0, 0)
    legitimate_count = class_counts.get(0.0, 0)

    if fraud_count == 0 or legitimate_count == 0:
        raise ValueError("Training data must contain both classes.")

    # Assign greater weight to the minority fraud class
    fraud_weight = legitimate_count / fraud_count

    from pyspark.sql.functions import when, col

    weighted_train_df = train_df.withColumn(
        "classWeight",
        when(col("label") == 1.0, fraud_weight).otherwise(1.0)
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=100
    )

    return lr.fit(
        weighted_train_df.select("features", "label", "classWeight")
    )

def train_decision_tree(train_df):
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=5,
        seed=42
    )

    dt_model = dt.fit(train_df.select("features", "label"))
    return dt_model

def train_gradient_boosted_trees(train_df):
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=30,
        maxDepth=3,
        seed=42
    )

    gbt_model = gbt.fit(train_df.select("features", "label"))
    return gbt_model

def train_linear_svm(train_df):
    svm = LinearSVC(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01
    )

    svm_model = svm.fit(train_df.select("features", "label"))
    return svm_model