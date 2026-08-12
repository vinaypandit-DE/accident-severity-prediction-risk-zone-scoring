# Intelligent Road Safety Analytics

An end-to-end big data and machine learning project for analyzing road accident data, predicting accident severity, and identifying high-risk geographic zones.

The project processes **300K+ road accident records** using PySpark and SparkSQL, performs data cleaning and feature engineering, trains and evaluates machine learning models for accident severity prediction, calculates geographic risk-zone scores, and orchestrates the workflow using Apache Airflow.

## Project Objectives

* Analyze large-scale road accident data using Apache Spark.
* Perform data cleaning, preprocessing, and feature engineering using PySpark.
* Use SparkSQL for analytical queries and data exploration.
* Build multiclass machine learning models to predict accident severity.
* Compare Random Forest and XGBoost models.
* Address class imbalance using sampling and model-weighting techniques.
* Develop a geographic risk-zone scoring framework to identify accident-prone locations.
* Automate the end-to-end workflow using Apache Airflow.
* Provide an interactive Streamlit application for model-based predictions.

## Technology Stack

| Category               | Technologies                  |
| ---------------------- | ----------------------------- |
| Programming            | Python                        |
| Big Data Processing    | Apache Spark, PySpark         |
| Query Engine           | SparkSQL                      |
| Machine Learning       | Scikit-learn, XGBoost         |
| Workflow Orchestration | Apache Airflow                |
| Data Analysis          | Pandas, NumPy                 |
| Visualization / App    | Streamlit                     |
| Data Format            | CSV, Parquet                  |
| Development            | Jupyter Notebook, Git, GitHub |

## Architecture

```text
                    Road Accident Dataset
                            |
                            v
                  Data Loading & Cleaning
                            |
                            v
                    PySpark Processing
                            |
             +--------------+--------------+
             |                             |
             v                             v
       SparkSQL Analysis             Feature Engineering
                                           |
                                           v
                                  Sampling / Preprocessing
                                           |
                            +--------------+--------------+
                            |                             |
                            v                             v
                     Random Forest                    XGBoost
                            |                             |
                            +--------------+--------------+
                                           |
                                           v
                                  Model Evaluation
                                           |
                                           v
                                  Model Comparison
                                           |
                            +--------------+--------------+
                            |                             |
                            v                             v
                  Severity Prediction             Risk Zone Scoring
                            |                             |
                            +--------------+--------------+
                                           |
                                           v
                                  Reports / Results
                                           |
                                           v
                                      Streamlit
```

## Data Pipeline

The project follows a Spark-based processing workflow:

1. Load the road accident dataset into the Spark environment.
2. Clean and validate the source data.
3. Handle missing values and prepare analytical features.
4. Perform exploratory analysis and SparkSQL-based analysis.
5. Engineer features required for machine learning.
6. Prepare training and testing datasets.
7. Apply partial oversampling and/or sample weighting to address class imbalance.
8. Train Random Forest and XGBoost models.
9. Evaluate model performance using classification metrics.
10. Compare model performance.
11. Calculate geographic risk scores using accident frequency, severity, casualty statistics, and geographic coordinates.
12. Generate reports and outputs.
13. Orchestrate the workflow through Apache Airflow.
14. Serve predictions through the Streamlit application.

## Machine Learning

The project treats accident severity as a **multiclass classification problem**.

### Models

* Random Forest
* XGBoost

### Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Model comparison results are generated as part of the automated pipeline.

## Class Imbalance Handling

Accident severity classes can be imbalanced, which can cause a model to favor the majority class.

The project experiments with techniques including:

* Partial oversampling
* Sample weighting

These approaches are incorporated into the training workflow and evaluated through the resulting model performance.

## Risk Zone Scoring

In addition to severity prediction, the project identifies potentially high-risk geographic locations.

The risk-zone framework considers factors including:

* Accident frequency
* Accident severity
* Casualty statistics
* Geographic coordinates

The resulting scores can be used to identify accident-prone areas and support data-driven road safety planning.

## Apache Airflow

Apache Airflow is used to automate and orchestrate the machine learning workflow.

The repository contains DAGs for:

* Random Forest pipeline
* Model comparison pipeline

The workflow covers processing, feature engineering, model training, evaluation, comparison, and reporting.

## Streamlit Application

The project includes a Streamlit application that provides an interactive interface for using the trained machine learning models.

The application uses the trained preprocessing components and model artifacts to support prediction.

## Repository Structure

```text
accident-severity-prediction-risk-zone-scoring/
│
├── airflow/
│   ├── dags/
│   │   ├── road_safety_model_comparison_pipeline.py
│   │   └── road_safety_rf_pipeline.py
│   │
│   ├── scripts/
│   │   ├── 01_feature_engineering.py
│   │   ├── 02_partial_oversampling.py
│   │   ├── 03_train_random_forest.py
│   │   ├── 04_evaluate_model.py
│   │   ├── 05_train_xgboost.py
│   │   ├── 06_evaluate_xgboost.py
│   │   └── 07_compare_models.py
│   │
│   ├── data/
│   ├── metrics/
│   ├── models/
│   └── reports/
│
├── cleaned_data/
│
├── data/
│   └── Road Accident Data.csv
│
├── notebooks/
│   ├── 01_Data_Loading_and_Cleaning.ipynb
│   ├── 02_SparkSQL_Analysis.ipynb
│   ├── 03_Exploratory Data Analysis (EDA).ipynb
│   ├── 04_Accident_Severity_ML_model.ipynb
│   ├── 05_XGBoost_sample weight.ipynb
│   ├── 06_rf_partialsampling.ipynb
│   └── 06_risk zone score.ipynb
│
├── outputs/
│
├── streamlit/
│
├── create_streamlit_models.py
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Install the required software:

* Python 3.x
* Apache Spark
* Apache Airflow
* Jupyter Notebook
* Git

Install the Python dependencies used by the project:

```bash
pip install pyspark pandas numpy scikit-learn xgboost streamlit
```

### Run the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Execute the notebooks in sequence to reproduce the data processing, analysis, machine learning, and risk-zone scoring workflow.

### Run the Airflow Pipeline

Configure your Airflow environment and copy the project DAGs into the Airflow DAG directory.

The main pipelines are:

```text
road_safety_rf_pipeline.py
road_safety_model_comparison_pipeline.py
```

After starting the Airflow scheduler and webserver, enable the required DAG from the Airflow UI.

### Run the Streamlit Application

From the project root:

```bash
streamlit run streamlit/app.py
```

If the application entry-point filename differs in your local project, use the corresponding Streamlit Python file.

## Key Outcomes

This project demonstrates an end-to-end workflow combining:

**Big Data Processing → Data Engineering → Machine Learning → Workflow Orchestration → Risk Analytics → Interactive Application**

It demonstrates practical experience with PySpark, SparkSQL, Apache Airflow, Scikit-learn, XGBoost, data preprocessing, feature engineering, ML evaluation, geographic analytics, and automated data workflows.

## Future Improvements

Potential extensions include:

* Deploying the pipeline on a cloud platform.
* Moving datasets to cloud object storage.
* Adding model monitoring and data-quality checks.
* Implementing experiment tracking.
* Containerizing the Airflow and Spark environment.
* Adding automated CI/CD for the project.
* Deploying the Streamlit application.
