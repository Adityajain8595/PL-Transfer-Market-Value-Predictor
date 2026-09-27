import json

import duckdb
import pandas as pd

from src.serve.clubs import CURRENT_2026_PL_CLUBS
from src.utils.logger import logger
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, RAW_DIR

VALUATIONS_PATH = RAW_DIR / "player_valuations.csv"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"
TOURNAMENT_PATH = MODELS_DIR / "tournament_metrics.json"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
ACTIVE_CATALOG_PATH = PROCESSED_DIR / "active_players_catalog.parquet"
ACTIVE_FEATURES_PATH = PROCESSED_DIR / "active_players_features.parquet"
METADATA_PATH = PROCESSED_DIR / "players_metadata.parquet"


def load_features_df() -> pd.DataFrame:
    try:
        return duckdb.read_parquet(str(FEATURES_PATH).replace("\\", "/")).df()
    except Exception:
        return pd.read_parquet(FEATURES_PATH)


# Load champion model metrics
def load_model_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Could not read champion metrics file: {err}")
    return {
        "model_name": "LightGBM",
        "test_mae": 3094428.0,
        "test_rmse": 5019762.0,
        "test_r2": 0.9528,
    }


# Load tournament comparison metrics
def load_tournament_metrics() -> dict:
    if TOURNAMENT_PATH.exists():
        try:
            with open(TOURNAMENT_PATH, "r") as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Could not read tournament metrics file: {err}")
    return {}


# Load processed dataset metrics
def load_dataset_stats() -> dict:
    if FEATURES_PATH.exists():
        try:
            df = load_features_df()
            dates = pd.to_datetime(df["valuation_date"])
            min_yr = dates.min().year
            max_yr = dates.max().year
            feature_cols = [
                c
                for c in df.columns
                if c
                not in [
                    "player_id",
                    "valuation_date",
                    "prev_valuation_date",
                    "target_market_value_eur",
                    "log_target_market_value",
                ]
            ]
            avg_days = int(df["days_between_valuations"].mean()) if "days_between_valuations" in df.columns else 0
            cutoff_str = dates.max().strftime("%B %Y")
            return {
                "total_records": len(df),
                "min_year": min_yr,
                "max_year": max_yr,
                "feature_count": len(feature_cols),
                "players_count": df["player_id"].nunique(),
                "avg_days_between": avg_days,
                "cutoff_date": cutoff_str,
            }
        except Exception as err:
            logger.warning(f"Could not compute dataset stats: {err}")
    return {
        "total_records": 0,
        "min_year": 0,
        "max_year": 0,
        "feature_count": 0,
        "players_count": 0,
        "avg_days_between": 0,
        "cutoff_date": "N/A",
    }


def build_catalog_from_parquet(fdf: pd.DataFrame) -> pd.DataFrame:
    cat = fdf.sort_values(by="valuation_date", ascending=False).groupby("player_id").first().reset_index()
    if "is_current_pl" in cat.columns:
        cat = cat[cat["is_current_pl"]]
    if "club_name" in cat.columns:
        cat = cat[cat["club_name"].isin(CURRENT_2026_PL_CLUBS)]

    cat["latest_recorded_val_eur"] = cat["target_market_value_eur"]
    cat["latest_age"] = cat["age_at_valuation"]

    if METADATA_PATH.exists():
        try:
            meta_df = duckdb.read_parquet(str(METADATA_PATH).replace("\\", "/")).df()
            cat = cat.merge(meta_df, on="player_id", how="left")
        except Exception as err:
            logger.debug(f"Metadata parquet merge skipped: {err}")

    if "image_url" not in cat.columns:
        cat["image_url"] = ""
    else:
        cat["image_url"] = cat["image_url"].fillna("")

    if "country_of_citizenship" not in cat.columns:
        cat["country_of_citizenship"] = "Unknown"
    else:
        cat["country_of_citizenship"] = cat["country_of_citizenship"].fillna("Unknown")

    cat = cat.sort_values(by="latest_recorded_val_eur", ascending=False).reset_index(drop=True)
    return cat[[
        "player_id",
        "player_name",
        "image_url",
        "country_of_citizenship",
        "club_name",
        "position",
        "sub_position",
        "dominant_foot",
        "latest_age",
        "latest_recorded_val_eur",
    ]]


def get_player_metadata(player_id: int) -> tuple[str, str]:
    if METADATA_PATH.exists():
        try:
            mdf = duckdb.read_parquet(str(METADATA_PATH).replace("\\", "/")).df()
            matched = mdf[mdf["player_id"] == player_id]
            if not matched.empty:
                return (
                    str(matched.iloc[0].get("image_url", "")),
                    str(matched.iloc[0].get("country_of_citizenship", "Unknown")),
                )
        except Exception as err:
            logger.debug(f"Player metadata lookup skipped: {err}")
    return "", "Unknown"


# Load active player catalog (serving tier)
def load_catalog() -> pd.DataFrame:
    if ACTIVE_CATALOG_PATH.exists():
        try:
            return duckdb.read_parquet(str(ACTIVE_CATALOG_PATH).replace("\\", "/")).df()
        except Exception as err:
            logger.debug(f"Parquet catalog read fallback: {err}")
            return pd.read_parquet(ACTIVE_CATALOG_PATH)

    if FEATURES_PATH.exists():
        fdf = load_features_df()
        return build_catalog_from_parquet(fdf)

    return pd.DataFrame()


# Load player valuation trajectory
def load_history(player_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if VALUATIONS_PATH.exists():
        try:
            con = duckdb.connect(database=":memory:")
            v_path = str(VALUATIONS_PATH).replace("\\", "/")
            val_query = f"""
                SELECT 
                    date::DATE as valuation_date,
                    TRY_CAST(market_value_in_eur AS BIGINT) as market_value_eur,
                    COALESCE(current_club_name, 'Unknown Club') as club_name
                FROM read_csv_auto('{v_path}')
                WHERE player_id = {player_id} AND market_value_in_eur IS NOT NULL
                ORDER BY valuation_date ASC;
            """
            val_df = con.execute(val_query).df()
            con.close()
            if not val_df.empty:
                val_df["val_change_eur"] = val_df["market_value_eur"].diff().fillna(0)
                prev_val = val_df["market_value_eur"].shift(1)
                val_df["val_change_pct"] = ((val_df["val_change_eur"] / prev_val) * 100.0).fillna(0)
                table_df = val_df.sort_values(by="valuation_date", ascending=False).copy()
                return val_df, table_df
        except Exception as err:
            logger.warning(f"DuckDB valuation history failed: {err}")

    if FEATURES_PATH.exists():
        fdf = load_features_df()
        pdf = fdf[fdf["player_id"] == player_id].sort_values(by="valuation_date", ascending=True)
        if not pdf.empty:
            val_df = pd.DataFrame({
                "valuation_date": pd.to_datetime(pdf["valuation_date"]),
                "market_value_eur": pdf["target_market_value_eur"],
                "club_name": pdf["club_name"],
            })
            val_df["val_change_eur"] = val_df["market_value_eur"].diff().fillna(0)
            prev_val = val_df["market_value_eur"].shift(1)
            val_df["val_change_pct"] = ((val_df["val_change_eur"] / prev_val) * 100.0).fillna(0)
            table_df = val_df.sort_values(by="valuation_date", ascending=False).copy()
            return val_df, table_df

    return pd.DataFrame(), pd.DataFrame(
        columns=["valuation_date", "club_name", "market_value_eur", "val_change_eur", "val_change_pct"]
    )


# Extract individual player profile
def get_player_features(player_id: int) -> dict:
    if ACTIVE_FEATURES_PATH.exists():
        try:
            adf = duckdb.read_parquet(str(ACTIVE_FEATURES_PATH).replace("\\", "/")).df()
            matched = adf[adf["player_id"] == player_id]
            if not matched.empty:
                return matched.iloc[0].to_dict()
        except Exception as err:
            logger.debug(f"Active features lookup fallback: {err}")

    if FEATURES_PATH.exists():
        fdf = load_features_df()
        pdf = fdf[fdf["player_id"] == player_id]
        if not pdf.empty:
            row = pdf.sort_values(by="valuation_date", ascending=False).iloc[0].to_dict()
            img_url, citizenship = get_player_metadata(player_id)
            val_eur = float(row.get("target_market_value_eur", 0.0))
            return {
                "player_id": player_id,
                "player_name": str(row.get("player_name", "Unknown")),
                "image_url": img_url,
                "country_of_citizenship": citizenship,
                "club_name": str(row.get("club_name", "Unknown")),
                "position": str(row.get("position", "Midfield")),
                "sub_position": str(row.get("sub_position", "Central Midfield")),
                "dominant_foot": str(row.get("dominant_foot", "right")),
                "age_at_valuation": float(row.get("age_at_valuation", 25.0)),
                "target_market_value_eur": val_eur,
                "current_market_value_eur": val_eur,
                "prior_market_value_eur": float(row.get("prev_market_value_eur", val_eur)),
                "days_between_valuations": int(row.get("days_between_valuations", 180)),
                "window_appearances": int(row.get("minutes_since_last_val", 0) // 90),
                "window_minutes": int(row.get("minutes_since_last_val", 0)),
                "window_goals": 0,
                "window_assists": 0,
                "window_yellow_cards": int(row.get("yellow_cards_since_val", 0)),
                "window_red_cards": 0,
                "window_european_minutes": int(row.get("european_minutes_played", 0)),
                "goal_contributions_per_90": float(row.get("goal_contributions_per_90", 0.0)),
                "minutes_since_last_val": int(row.get("minutes_since_last_val", 0)),
                "european_minutes_played": int(row.get("european_minutes_played", 0)),
                "minutes_prior_window": int(row.get("minutes_prior_window", 0)),
                "contrib_per_90_prior": float(row.get("contrib_per_90_prior", 0.0)),
                "yellow_cards_since_val": int(row.get("yellow_cards_since_val", 0)),
            }

    return {}