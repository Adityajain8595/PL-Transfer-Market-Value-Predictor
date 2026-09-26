import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils.paths import CONFIG_DIR, MODELS_DIR, PROCESSED_DIR

DATA_CONFIG_PATH = CONFIG_DIR / "data_config.yaml"
MODEL_PATH = MODELS_DIR / "champion_model.joblib"

MAX_MAE = 3_500_000.0
MIN_R2 = 0.90


# Model regression quality gate tests
def test_champion_artifact_exists():
    assert MODEL_PATH.exists(), f"Champion model missing: {MODEL_PATH}"


def test_model_performance_regression_gate():
    with open(DATA_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    parquet_path = PROCESSED_DIR / cfg["processed_data"]["parquet_filename"]
    df = pd.read_parquet(parquet_path)
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])

    test_months = cfg["rolling_split"]["test_months"]
    test_start = df["valuation_date"].max() - pd.DateOffset(months=test_months)
    test_df = df[df["valuation_date"] >= test_start]
    assert len(test_df) > 0, "Test set contains 0 samples."

    features = cfg["features"]["numerical"] + cfg["features"]["categorical"]
    target_col = cfg["features"]["target"]

    x_test = test_df[features]
    y_test_log = test_df[target_col].values
    y_test_eur = np.expm1(y_test_log)

    model = joblib.load(MODEL_PATH)
    y_pred_log = model.predict(x_test)
    y_pred_eur = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

    mae = mean_absolute_error(y_test_eur, y_pred_eur)
    r2 = r2_score(y_test_eur, y_pred_eur)

    assert mae <= MAX_MAE, f"Model degraded! MAE EUR {mae:,.0f} > EUR {MAX_MAE:,.0f}"
    assert r2 >= MIN_R2, f"Model degraded! R2 {r2:.4f} < {MIN_R2}"