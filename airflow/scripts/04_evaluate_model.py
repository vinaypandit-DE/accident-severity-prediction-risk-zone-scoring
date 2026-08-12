from pyspark.sql import SparkSession

from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os

# ----------------------------------------------------
# Create Spark Session
# ----------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Evaluate Random Forest")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
model = RandomForestClassificationModel.load(
    "../models/random_forest_model"
)

# ----------------------------------------------------
# Load Test Data
# ----------------------------------------------------
test_df = spark.read.parquet("../data/test_data")

print(f"Testing Records : {test_df.count()}")

# ----------------------------------------------------
# Predictions
# ----------------------------------------------------
predictions = model.transform(test_df)

# ----------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------
accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

# ----------------------------------------------------
# Print Metrics
# ----------------------------------------------------
print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------
print("\n========== CONFUSION MATRIX ==========")

predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
).show(50)

# ----------------------------------------------------
# Save Metrics
# ----------------------------------------------------
os.makedirs("../metrics", exist_ok=True)

with open("../metrics/model_metrics.txt", "w") as f:

    f.write("Random Forest Model Evaluation\n")
    f.write("===============================\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")

print("\nMetrics saved to:")
print("../metrics/model_metrics.txt")

spark.stop()