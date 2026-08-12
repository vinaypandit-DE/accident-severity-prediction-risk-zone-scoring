from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "vinay",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="road_safety_rf_pipeline",
    description="Road Safety Analytics Random Forest Pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["Road Safety", "PySpark", "ML"],
) as dag:

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="""
cd /home/dockeradmin/projects/Intelligent-Road-Safety-Analytics/airflow/scripts
python3 01_feature_engineering.py
"""
    )

    partial_oversampling = BashOperator(
        task_id="partial_oversampling",
        bash_command="""
cd /home/dockeradmin/projects/Intelligent-Road-Safety-Analytics/airflow/scripts
python3 02_partial_oversampling.py
"""
    )

    train_random_forest = BashOperator(
        task_id="train_random_forest",
        bash_command="""
cd /home/dockeradmin/projects/Intelligent-Road-Safety-Analytics/airflow/scripts
python3 03_train_random_forest.py
"""
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="""
cd /home/dockeradmin/projects/Intelligent-Road-Safety-Analytics/airflow/scripts
python3 04_evaluate_model.py
"""
    )

    (
        feature_engineering
        >> partial_oversampling
        >> train_random_forest
        >> evaluate_model
    )