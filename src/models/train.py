import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import dagshub
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.models.evaluate import evaluate_preds
from src.models.helpers import get_models, get_preprocessor, load_split_data
from src.models.register_model import register_model
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.paths import MODELS_DIR


# Train tournament and champion
def run_train_pipeline():
    try:
        train_df, val_df, test_df, data_cfg, model_cfg = load_split_data()

        # Connect MLflow tracking
        if model_cfg["mlflow"].get("use_dagshub", False):
            logger.info("Connecting to DagsHub MLflow...")
            dagshub.init(
                repo_owner=model_cfg["mlflow"]["dagshub_repo_owner"],
                repo_name=model_cfg["mlflow"]["dagshub_repo_name"],
                mlflow=True
            )
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment(model_cfg["mlflow"]["experiment_name"])

        cat_cols = data_cfg["features"]["categorical"]
        num_cols = data_cfg["features"]["numerical"]
        features = num_cols + cat_cols
        target = data_cfg["features"]["target"]

        x_train, y_train = train_df[features], train_df[target].values
        x_val, y_val = val_df[features], val_df[target].values
        x_test, y_test = test_df[features], test_df[target].values

        preprocessor = get_preprocessor(cat_cols, num_cols)
        models = get_models(model_cfg)

        best_mae = float("inf")
        champ_name = None
        champ_instance = None

        # Execute tournament runs
        logger.info(f"Running tournament across {len(models)} models...")
        for name, instance in models.items():
            with mlflow.start_run(run_name=f"Tournament_{name}"):
                pipe = Pipeline([("preprocessor", clone(preprocessor)), ("regressor", clone(instance))])
                pipe.fit(x_train, y_train)

                val_preds = pipe.predict(x_val)
                metrics = evaluate_preds(y_val, val_preds)

                for k, v in metrics.items():
                    mlflow.log_metric(f"val_{k}", v)

                logger.info(f"[{name}] Val MAE: EUR {metrics['MAE']:,.0f} | R2: {metrics['R2']:.4f}")
                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    champ_name = name
                    champ_instance = instance

        logger.info(f"Tournament Champion: [{champ_name}] (Val MAE: EUR {best_mae:,.0f})")

        if champ_instance is None or champ_name is None:
            raise ValueError("No champion model was selected during the tournament.")

        # Retrain champion on train_val
        x_full = pd.concat([x_train, x_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        champ_pipe = Pipeline([("preprocessor", clone(preprocessor)), ("regressor", clone(champ_instance))])
        champ_pipe.fit(x_full, y_full)

        test_preds = champ_pipe.predict(x_test)
        test_metrics = evaluate_preds(y_test, test_preds)

        # Champion vs Challenger gate
        champion_path = MODELS_DIR / "champion_model.joblib"
        should_promote = True
        if champion_path.exists():
            try:
                curr_model = joblib.load(champion_path)
                curr_preds = curr_model.predict(x_test)
                curr_metrics = evaluate_preds(y_test, curr_preds)
                logger.info(f"Existing Champion Test MAE: EUR {curr_metrics['MAE']:,.0f} vs New Candidate MAE: EUR {test_metrics['MAE']:,.0f}")
                if test_metrics["MAE"] > curr_metrics["MAE"] * 1.05:
                    logger.warning("New candidate degraded by >5% over current champion. Promotion skipped.")
                    should_promote = False
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Challenger benchmark warning: {e}")

        # Log champion model run and conditionally promote
        with mlflow.start_run(run_name=f"Champion_{champ_name}"):
            if hasattr(champ_instance, "get_params"):
                mlflow.log_params(champ_instance.get_params())
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            logger.info(
                f"Champion [{champ_name}] Test Performance:\n"
                f"  MAE:  EUR {test_metrics['MAE']:,.0f}\n"
                f"  RMSE: EUR {test_metrics['RMSE']:,.0f}\n"
                f"  R2:   {test_metrics['R2']:.4f}"
            )

            if should_promote:
                model_info = mlflow.sklearn.log_model(
                    sk_model=champ_pipe,
                    name="model_pipeline",
                    serialization_format="cloudpickle"
                )

                register_model(
                    model_uri=model_info.model_uri,
                    model_name=model_cfg["mlflow"]["registered_model_name"]
                )

                # Persist local champion binary and metrics
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(champ_pipe, champion_path)
                import json
                metrics_data = {
                    "model_name": champ_name,
                    "test_mae": float(test_metrics["MAE"]),
                    "test_rmse": float(test_metrics["RMSE"]),
                    "test_r2": float(test_metrics["R2"]),
                }
                with open(MODELS_DIR / "champion_metrics.json", "w") as mf:
                    json.dump(metrics_data, mf, indent=2)
                logger.info("Saved models/champion_model.joblib and champion_metrics.json.")

    except Exception as err:
        logger.error("Training pipeline failed.")
        raise CustomException(err, sys) from err

if __name__ == "__main__":
    run_train_pipeline()