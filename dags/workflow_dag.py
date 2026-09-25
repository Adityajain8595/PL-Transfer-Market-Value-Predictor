from datetime import datetime, timedelta, timezone

from airflow.operators.python import PythonOperator

from airflow import DAG
from src.data.download import download_data
from src.data.dvc_sync import sync_dvc
from src.features.transform import transform_features
from src.models.train import run_train_pipeline
from src.monitoring.drift_detector import run_drift_pipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Airflow MLOps pipeline DAG
with DAG(
    dag_id="pl_market_value_prediction_pipeline",
    default_args=default_args,
    description="Automated Kaggle Ingestion, DuckDB, MLflow Pipeline",
    schedule="@monthly",
    catchup=False,
    tags=["mlops", "duckdb", "mlflow", "transfermarket"],
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=download_data,
    )

    task_features = PythonOperator(
        task_id="duckdb_feature_engineering",
        python_callable=transform_features,
    )

    task_dvc = PythonOperator(
        task_id="sync_dvc_remote",
        python_callable=sync_dvc,
        op_kwargs={"action": "push"},
    )

    task_train = PythonOperator(
        task_id="train_tournament_champion",
        python_callable=run_train_pipeline,
    )

    task_drift = PythonOperator(
        task_id="audit_data_drift",
        python_callable=run_drift_pipeline,
    )

    # Pipeline execution flow: ingest -> transform -> dvc sync -> train -> drift audit
    task_ingest >> task_features >> task_dvc >> task_train >> task_drift