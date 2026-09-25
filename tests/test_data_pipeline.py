import pandas as pd
import pytest
import yaml

from src.utils.paths import CONFIG_DIR, PROCESSED_DIR

CONFIG_PATH = CONFIG_DIR / "data_config.yaml"

# Fixture data configuration
@pytest.fixture(scope="module")
def data_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# Fixture processed features dataframe
@pytest.fixture(scope="module")
def processed_dataframe(data_config):
    parquet_path = PROCESSED_DIR / data_config["processed_data"]["parquet_filename"]
    assert parquet_path.exists(), f"Processed parquet missing: {parquet_path}"
    return pd.read_parquet(parquet_path)

# Validate dataset non-empty
def test_processed_dataset_not_empty(processed_dataframe):
    assert len(processed_dataframe) > 0, "Processed feature dataset is empty."

# Validate feature schema columns
def test_expected_features_exist(processed_dataframe, data_config):
    expected_cols = (
        data_config["features"]["numerical"]
        + data_config["features"]["categorical"]
        + [data_config["features"]["target"]]
        + ["valuation_date", "prev_valuation_date", "player_id"]
    )
    for col in expected_cols:
        assert col in processed_dataframe.columns, f"Missing column: {col}"

# Validate zero null values
def test_no_nulls_in_critical_columns(processed_dataframe, data_config):
    critical_cols = (
        data_config["features"]["numerical"]
        + data_config["features"]["categorical"]
        + [data_config["features"]["target"]]
    )
    null_counts = processed_dataframe[critical_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    assert len(null_cols) == 0, f"Found nulls:\n{null_cols}"

# Validate temporal sequencing
def test_no_temporal_leakage_in_dates(processed_dataframe):
    dates_valid = (
        pd.to_datetime(processed_dataframe["valuation_date"])
        > pd.to_datetime(processed_dataframe["prev_valuation_date"])
    ).all()
    assert dates_valid, "Temporal leakage detected."

# Validate duration boundaries
def test_days_between_valuations_bounds(processed_dataframe, data_config):
    min_days = data_config["filters"]["min_days_between"]
    max_days = data_config["filters"]["max_days_between"]
    assert (processed_dataframe["days_between_valuations"] >= min_days).all()
    assert (processed_dataframe["days_between_valuations"] <= max_days).all()

# Validate target positivity
def test_target_market_value_positivity(processed_dataframe):
    assert (processed_dataframe["target_market_value_eur"] > 0).all()
    assert (processed_dataframe["log_target_market_value"] > 0).all()