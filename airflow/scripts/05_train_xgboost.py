from pyspark.sql import SparkSession
import pandas as pd

import xgboost as xgb
import joblib
import os

# -------------------------------------------------
# Create Spark Session
# -------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Train XGBoost Model")
    .getOrCreate()
)

# -------------------------------------------------
# Load Balanced Training Data
# -------------------------------------------------

train_df = spark.read.parquet("../data/balanced_train")

total_records = train_df.count()
print(f"Total Training Records : {total_records}")

#
sample_size = 100000

fraction = min(sample_size / total_records, 1.0)

train_df = train_df.sample(
    withReplacement=False,
    fraction=fraction,
    seed=42
)

sample_records = train_df.count()
print(f"Sampled Training Records : {sample_records}")

# -------------------------------------------------
# Convert Spark DataFrame to Pandas
# -------------------------------------------------

pdf = train_df.toPandas()

# Convert Spark DenseVector to list
pdf["features"] = pdf["features"].apply(lambda x: x.toArray())

# Expand feature vector into columns
X = pd.DataFrame(
    pdf["features"].tolist()
)

y = pdf["label"]

print(f"Number of Features : {X.shape[1]}")
print(f"Number of Classes  : {len(y.unique())}")

# -------------------------------------------------
# Train XGBoost Model
# -------------------------------------------------

model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(y.unique()),
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

print("\nTraining XGBoost Model...")

model.fit(X, y)

print("Training Completed Successfully.")

# -------------------------------------------------
# Save Model
# -------------------------------------------------

os.makedirs("../models", exist_ok=True)

joblib.dump(
    model,
    "../models/xgboost_model.pkl"
)

print("\nModel saved successfully.")
print("Location : ../models/xgboost_model.pkl")

# -------------------------------------------------
# Stop Spark
# -------------------------------------------------

spark.stop()