from pyspark.sql import SparkSession

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import os


# Create Spark Session
# -------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Evaluate XGBoost Model")
    .getOrCreate()
)

# -------------------------------------------------
# Load Test Data
# -------------------------------------------------

test_df = spark.read.parquet("../data/test_data")

print(f"Test Records : {test_df.count()}")

# -------------------------------------------------
# Convert to Pandas
# -------------------------------------------------

pdf = test_df.toPandas()

# Convert Spark DenseVector to list
pdf["features"] = pdf["features"].apply(lambda x: x.toArray())

# Expand feature vector into separate columns
X_test = pd.DataFrame(
    pdf["features"].tolist()
)

y_test = pdf["label"]

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = joblib.load("../models/xgboost_model.pkl")

print("XGBoost model loaded successfully.")

# -------------------------------------------------
# Predictions
# -------------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# -------------------------------------------------
# Print Metrics
# -------------------------------------------------

print("\nModel Performance")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# -------------------------------------------------
# Save Metrics
# -------------------------------------------------

os.makedirs("../metrics", exist_ok=True)

with open("../metrics/xgb_metrics.txt", "w") as f:

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print("\nMetrics saved successfully.")
print("Location : ../metrics/xgb_metrics.txt")

# -------------------------------------------------
# Stop Spark
# -------------------------------------------------

spark.stop()