from datetime import datetime, timedelta, timezone

from airflow.operators.python import PythonOperator

from airflow import DAG


def _task_ingest():
    from src.data.download import download_data
    download_data()


def _task_features():
    from src.features.transform import transform_features
    transform_features()


def _task_dvc(action: str = "push"):
    from src.data.dvc_sync import sync_dvc
    sync_dvc(action=action)


def _task_train():
    from src.models.train import run_train_pipeline
    run_train_pipeline()


def _task_drift():
    from src.monitoring.drift_detector import run_drift_pipeline
    run_drift_pipeline()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

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
        python_callable=_task_ingest,
    )

    task_features = PythonOperator(
        task_id="duckdb_feature_engineering",
        python_callable=_task_features,
    )

    task_dvc = PythonOperator(
        task_id="sync_dvc_remote",
        python_callable=_task_dvc,
        op_kwargs={"action": "push"},
    )

    task_train = PythonOperator(
        task_id="train_tournament_champion",
        python_callable=_task_train,
    )

    task_drift = PythonOperator(
        task_id="audit_data_drift",
        python_callable=_task_drift,
    )

    task_ingest >> task_features >> task_dvc >> task_train >> task_drift