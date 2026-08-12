from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

# ----------------------------------------------------
# Create Spark Session
# ----------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Feature Engineering")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# ----------------------------------------------------
# Read Cleaned Dataset
# ----------------------------------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("../../cleaned_data/road_accident_cleaned")
)

print(f"Total Records: {df.count()}")

# ----------------------------------------------------
# Features
# ----------------------------------------------------
feature_cols = [
    "Month",
    "Day_of_Week",
    "Junction_Control",
    "Junction_Detail",
    "Light_Conditions",
    "Carriageway_Hazards",
    "Number_of_Casualties",
    "Number_of_Vehicles",
    "Road_Surface_Conditions",
    "Road_Type",
    "Speed_limit",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type"
]

target_col = "Accident_Severity"

# ----------------------------------------------------
# Categorical & Numerical Columns
# ----------------------------------------------------
categorical_cols = [
    "Month",
    "Day_of_Week",
    "Junction_Control",
    "Junction_Detail",
    "Light_Conditions",
    "Carriageway_Hazards",
    "Road_Surface_Conditions",
    "Road_Type",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type"
]

numeric_cols = [
    "Number_of_Casualties",
    "Number_of_Vehicles",
    "Speed_limit"
]

# ----------------------------------------------------
# String Indexers
# ----------------------------------------------------
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"
    )
    for col in categorical_cols
]

label_indexer = StringIndexer(
    inputCol=target_col,
    outputCol="label",
    handleInvalid="keep"
)

# ----------------------------------------------------
# One Hot Encoding
# ----------------------------------------------------
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_vec" for c in categorical_cols]
)

# ----------------------------------------------------
# Assemble Features
# ----------------------------------------------------
assembler = VectorAssembler(
    inputCols=[f"{c}_vec" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)

# ----------------------------------------------------
# Build Pipeline
# ----------------------------------------------------
pipeline = Pipeline(
    stages=indexers + [label_indexer, encoder, assembler]
)

pipeline_model = pipeline.fit(df)

processed_df = pipeline_model.transform(df)

label_indexer_model = pipeline_model.stages[len(indexers)]

print("Label Mapping:")
for i, label in enumerate(label_indexer_model.labels):
    print(f"{i} -> {label}")
# ----------------------------------------------------
# Keep Required Columns
# ----------------------------------------------------
processed_df = processed_df.select(
    "features",
    "label"
)

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------
train_df, test_df = processed_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(f"Training Records : {train_df.count()}")
print(f"Testing Records  : {test_df.count()}")

# ----------------------------------------------------
# Save Output
# ----------------------------------------------------
train_df.write.mode("overwrite").parquet(
    "../data/train_data"
)

test_df.write.mode("overwrite").parquet(
    "../data/test_data"
)

print("Feature Engineering Completed Successfully.")

spark.stop()