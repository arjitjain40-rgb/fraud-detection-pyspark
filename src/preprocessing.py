
def remove_duplicates(df):
    original_count = df.count()

    df_clean = df.dropDuplicates()

    cleaned_count = df_clean.count()
    duplicates_removed = original_count - cleaned_count

    # print("Original rows:", original_count)
    # print("Rows after removing duplicates:", cleaned_count)
    # print("Duplicates removed:", duplicates_removed)

    return df_clean

# def check_class_distribution(df):
#     df.groupBy("Class").count().show()

# def show_dataset_summary(df):
#     print("Dataset summary:")
#     print("Total transactions:", df.count())
#     print("Total features:", len(df.columns))
#     print("Fraud transactions:", df.filter(df.Class == 1).count())
#     print("Legitimate transactions:", df.filter(df.Class == 0).count())

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col


def prepare_features(df):
    feature_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    df = df.withColumn("label", col("Class").cast("double"))

    df_ml = assembler.transform(df)

    # Retain Time and Amount alongside features and label for Power BI & BI reporting
    return df_ml.select("Time", "Amount", "features", "label")

def split_data(df_ml):
    train_df, test_df = df_ml.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    # print("Training rows:", train_df.count())
    # print("Testing rows:", test_df.count())

    return train_df, test_df

# def check_split_distribution(train_df, test_df):
#     # print("Training class distribution:")
#     # train_df.groupBy("label").count().show()

#     # print("Testing class distribution:")
#     # test_df.groupBy("label").count().show()

def split_training_validation(train_df):
    model_train_df, validation_df = train_df.randomSplit(
        [0.8, 0.2],
        seed=123
    )

    print("Model training rows:", model_train_df.count())
    print("Validation rows:", validation_df.count())

    return model_train_df, validation_df