import pandas as pd
import yaml
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.utils.paths import CONFIG_DIR, PROCESSED_DIR

DATA_CONFIG = CONFIG_DIR / "data_config.yaml"
MODEL_CONFIG = CONFIG_DIR / "model_config.yaml"

# Load and split dataset
def load_split_data():
    with open(DATA_CONFIG, "r") as f:
        data_cfg = yaml.safe_load(f)

    with open(MODEL_CONFIG, "r") as f:
        model_cfg = yaml.safe_load(f)

    parquet_path = PROCESSED_DIR / data_cfg["processed_data"]["parquet_filename"]
    df = pd.read_parquet(parquet_path)
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])

    # Deduplicate and filter out inactive/retired records with zero or null match stats
    df = df.drop_duplicates(subset=["player_id", "valuation_date"])
    if "minutes_since_last_val" in df.columns:
        df = df[df["minutes_since_last_val"] > 0]
    req_cols = data_cfg["features"]["numerical"] + data_cfg["features"]["categorical"] + [data_cfg["features"]["target"]]
    df = df.dropna(subset=[col for col in req_cols if col in df.columns])

    val_months = data_cfg["rolling_split"]["val_months"]
    test_months = data_cfg["rolling_split"]["test_months"]

    max_date = df["valuation_date"].max()
    test_start = max_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train_mask = df["valuation_date"] < val_start
    val_mask = (df["valuation_date"] >= val_start) & (df["valuation_date"] < test_start)
    test_mask = df["valuation_date"] >= test_start

    return df[train_mask], df[val_mask], df[test_mask], data_cfg, model_cfg

# Feature preprocessing pipeline
def get_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# Candidate model registry
def get_models(model_cfg: dict) -> dict:
    models = {}
    cfg = model_cfg["models"]
    if "LightGBM" in cfg:
        models["LightGBM"] = LGBMRegressor(**cfg["LightGBM"])
    if "XGBoost" in cfg:
        models["XGBoost"] = XGBRegressor(**cfg["XGBoost"])
    if "CatBoost" in cfg:
        models["CatBoost"] = CatBoostRegressor(**cfg["CatBoost"])
    if "RandomForest" in cfg:
        models["RandomForest"] = RandomForestRegressor(**cfg["RandomForest"])
    if "GradientBoosting" in cfg:
        models["GradientBoosting"] = GradientBoostingRegressor(**cfg["GradientBoosting"])
    return models