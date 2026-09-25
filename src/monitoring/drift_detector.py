import json
import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import dagshub
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
)

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.paths import CONFIG_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

DATA_CONFIG_PATH = CONFIG_DIR / "data_config.yaml"
DRIFT_CONFIG_PATH = CONFIG_DIR / "drift_config.yaml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
MODEL_PATH = MODELS_DIR / "champion_model.joblib"

# Load datasets and configs
def load_datasets_configs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    try:
        with open(DATA_CONFIG_PATH, "r") as f:
            data_cfg = yaml.safe_load(f)
        with open(DRIFT_CONFIG_PATH, "r") as f:
            drift_cfg = yaml.safe_load(f)

        parquet_path = PROCESSED_DIR / data_cfg["processed_data"]["parquet_filename"]
        df = pd.read_parquet(parquet_path)
        df["valuation_date"] = pd.to_datetime(df["valuation_date"])

        val_months = data_cfg["rolling_split"]["val_months"]
        test_months = data_cfg["rolling_split"]["test_months"]

        max_date = df["valuation_date"].max()
        test_start = max_date - pd.DateOffset(months=test_months)
        val_start = test_start - pd.DateOffset(months=val_months)

        ref_mask = df["valuation_date"] < val_start
        cur_mask = df["valuation_date"] >= test_start

        ref_df = df[ref_mask].copy()
        cur_df = df[cur_mask].copy()

        logger.info(f"Loaded: Ref={len(ref_df)}, Cur={len(cur_df)} rows.")
        return ref_df, cur_df, drift_cfg

    except Exception as err:
        logger.error("Failed loading drift data.")
        raise CustomException(err, sys) from err

# Attach predictions for drift
def attach_predictions(model, ref_df: pd.DataFrame, cur_df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ref_preds = model.predict(ref_df[cols])
        cur_preds = model.predict(cur_df[cols])

        ref_df["prediction"] = np.clip(np.expm1(ref_preds), a_min=0, a_max=None)
        cur_df["prediction"] = np.clip(np.expm1(cur_preds), a_min=0, a_max=None)

        return ref_df, cur_df
    except Exception as err:
        logger.error("Failed generating drift predictions.")
        raise CustomException(err, sys) from err

# Run drift audit pipeline
def run_drift_pipeline() -> bool:
    try:
        ref_df, cur_df, drift_cfg = load_datasets_configs()
        drift_data = drift_cfg["drift"]

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        num_cols = drift_data["features_to_monitor"]["numerical"]
        cat_cols = drift_data["features_to_monitor"]["categorical"]
        all_cols = num_cols + cat_cols

        ref_df, cur_df = attach_predictions(model, ref_df, cur_df, all_cols)

        # Configure column mapping
        col_mapping = ColumnMapping(
            prediction="prediction",
            target="target_market_value_eur",
            numerical_features=num_cols,
            categorical_features=cat_cols
        )

        out_dir = REPORTS_DIR / "drift"
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / drift_data["html_report_filename"]
        json_path = out_dir / drift_data["json_summary_filename"]

        # Run Evidently drift report
        logger.info("Computing Evidently drift report...")
        report = Report(metrics=[
            DataDriftPreset(
                drift_share=drift_data["thresholds"]["drift_share_threshold"],
                num_stattest=drift_data["thresholds"]["numerical"]["stat_test"],
                cat_stattest=drift_data["thresholds"]["categorical"]["stat_test"],
                num_stattest_threshold=drift_data["thresholds"]["numerical"]["threshold"],
                cat_stattest_threshold=drift_data["thresholds"]["categorical"]["threshold"]
            ),
            TargetDriftPreset()
        ])

        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)
        report.save_html(str(html_path))
        logger.info(f"Saved drift HTML: {html_path}")

        # Run test suite gates
        threshold = drift_data["thresholds"]["drift_share_threshold"]
        drift_tests = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=threshold),
            TestColumnDrift(column_name="prediction", stattest="wasserstein", stattest_threshold=0.15)
        ])

        drift_tests.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)
        suite_dict = drift_tests.as_dict()

        with open(json_path, "w") as f:
            json.dump(suite_dict, f, indent=2)
        logger.info(f"Saved drift summary: {json_path}")

        all_passed = suite_dict["summary"]["all_passed"]

        # Log drift audit MLflow
        if MODEL_CONFIG_PATH.exists():
            with open(MODEL_CONFIG_PATH, "r") as f:
                model_cfg = yaml.safe_load(f)
            if model_cfg.get("mlflow", {}).get("use_dagshub", False):
                dagshub.init(
                    repo_owner=model_cfg["mlflow"]["dagshub_repo_owner"],
                    repo_name=model_cfg["mlflow"]["dagshub_repo_name"],
                    mlflow=True
                )
            else:
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment("PL_Market_Value_Monitoring")

        with mlflow.start_run(run_name="Evidently_Data_Drift_Audit"):
            mlflow.log_artifact(str(html_path), artifact_path="drift_reports")
            mlflow.log_artifact(str(json_path), artifact_path="drift_reports")

            report_dict = report.as_dict()
            metrics = report_dict["metrics"][0]["result"]
            mlflow.log_metric("drifted_features_count", metrics["number_of_drifted_columns"])
            mlflow.log_metric("drift_share", metrics["share_of_drifted_columns"])
            mlflow.log_metric("dataset_drift_detected", int(metrics["dataset_drift"]))

        return all_passed

    except Exception as err:
        logger.error("Drift pipeline failed.")
        raise CustomException(err, sys) from err

if __name__ == "__main__":
    run_drift_pipeline()