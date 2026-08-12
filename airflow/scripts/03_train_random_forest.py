from pyspark.sql import SparkSession

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ----------------------------------------------------
# Create Spark Session
# ----------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Train Random Forest")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
train_df = spark.read.parquet("../data/balanced_train")
test_df = spark.read.parquet("../data/test_data")

print(f"Training Records : {train_df.count()}")
print(f"Testing Records  : {test_df.count()}")

# ----------------------------------------------------
# Train Random Forest
# ----------------------------------------------------
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    seed=42
)

model = rf.fit(train_df)

print("\nModel training completed.")

# ----------------------------------------------------
# Save Model
# ----------------------------------------------------
model.write().overwrite().save("../models/random_forest_model")

print("Model saved successfully.")

# ----------------------------------------------------
# Predict on Test Data
# ----------------------------------------------------
predictions = model.transform(test_df)

predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)

# ----------------------------------------------------
# Accuracy
# ----------------------------------------------------
accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

print(f"\nAccuracy : {accuracy:.4f}")

spark.stop()