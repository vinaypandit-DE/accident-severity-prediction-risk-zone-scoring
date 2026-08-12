from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ----------------------------------------------------
# Create Spark Session
# ----------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Partial Oversampling")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# ----------------------------------------------------
# Load Training Data
# ----------------------------------------------------
train_df = spark.read.parquet("../data/train_data")

print(f"\nOriginal Training Records: {train_df.count()}")

# ----------------------------------------------------
# Show Label Distribution
# ----------------------------------------------------
print("\nLabel Distribution:")

label_counts = (
    train_df
    .groupBy("label")
    .count()
    .orderBy("count", ascending=False)
)

label_counts.show()

# ----------------------------------------------------
# Identify Majority and Minority Classes
# ----------------------------------------------------
counts = label_counts.collect()

majority_label = counts[0]["label"]
middle_label = counts[1]["label"]
minority_label = counts[2]["label"]

majority_df = train_df.filter(col("label") == majority_label)
middle_df = train_df.filter(col("label") == middle_label)
minority_df = train_df.filter(col("label") == minority_label)

print("\nBefore Oversampling")

print("Majority :", majority_df.count())
print("Middle   :", middle_df.count())
print("Minority :", minority_df.count())

# ----------------------------------------------------
# Partial Oversampling
# ----------------------------------------------------
middle_over = middle_df.sample(
    withReplacement=True,
    fraction=2.0,
    seed=42
)

minority_over = minority_df.sample(
    withReplacement=True,
    fraction=5.0,
    seed=42
)

balanced_train = (
    majority_df
    .union(middle_df)
    .union(minority_df)
    .union(middle_over)
    .union(minority_over)
)

print("\nAfter Oversampling")

print(
    "Majority :",
    balanced_train.filter(col("label") == majority_label).count()
)

print(
    "Middle   :",
    balanced_train.filter(col("label") == middle_label).count()
)

print(
    "Minority :",
    balanced_train.filter(col("label") == minority_label).count()
)

# ----------------------------------------------------
# Save Dataset
# ----------------------------------------------------
balanced_train.write.mode("overwrite").parquet(
    "../data/balanced_train"
)

print("\nBalanced dataset saved successfully.")

spark.stop()