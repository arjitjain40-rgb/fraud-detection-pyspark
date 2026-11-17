
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum



# Configure Python and environment paths for Spark workers
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.name == "nt":
    python_exec = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Java\jdk-17.0.18")
    os.environ.setdefault("HADOOP_HOME", r"D:\hadoop")
    if r"D:\hadoop\bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = r"D:\hadoop\bin;" + os.environ.get("PATH", "")
else:
    python_exec = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

os.environ["PYSPARK_PYTHON"] = python_exec
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec


def create_spark_session():
    builder = (
        SparkSession.builder
        .appName("FraudDetection")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
    )
    if os.name == "nt":
        builder = (
            builder
            .config(
                "spark.driver.extraJavaOptions",
                "-Dhadoop.home.dir=D:/hadoop -Djava.library.path=D:/hadoop/bin"
            )
            .config(
                "spark.executor.extraJavaOptions",
                "-Dhadoop.home.dir=D:/hadoop -Djava.library.path=D:/hadoop/bin"
            )
            .config(
                "spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.RawLocalFileSystem"
            )
        )
    return builder.getOrCreate()


def load_fraud_data(spark):
    data_path = os.path.join(
        PROJECT_ROOT, "data", "creditcard.csv"
    )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(data_path)
    )

    return df

def validate_fraud_data(df):
    print("Total rows:", df.count())
    print("Total columns:", len(df.columns))

    # Count missing values in each column
    null_counts = df.select([
        spark_sum(col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])

    print("\nMissing values per column:")
    null_counts.show(truncate=False)

    # Count exact duplicate rows
    duplicate_count = df.count() - df.dropDuplicates().count()
    print("Duplicate rows:", duplicate_count)


