import json

import duckdb
import pandas as pd

from src.serve.clubs import CURRENT_2026_PL_CLUBS
from src.utils.logger import logger
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, RAW_DIR

STATS_PATH = PROCESSED_DIR / "player_latest_stats.parquet"
PLAYERS_PATH = RAW_DIR / "players.csv"
VALUATIONS_PATH = RAW_DIR / "player_valuations.csv"
CLUBS_PATH = RAW_DIR / "clubs.csv"
GAMES_PATH = RAW_DIR / "games.csv"
APPEARANCES_PATH = RAW_DIR / "appearances.csv"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"
TOURNAMENT_PATH = MODELS_DIR / "tournament_metrics.json"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
ACTIVE_CATALOG_PATH = PROCESSED_DIR / "active_players_catalog.parquet"
ACTIVE_FEATURES_PATH = PROCESSED_DIR / "active_players_features.parquet"


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
        "test_r2": 0.9528
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
                c for c in df.columns
                if c not in ["player_id", "valuation_date", "prev_valuation_date", "target_market_value_eur", "log_target_market_value"]
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
        "cutoff_date": "June 2026",
    }






def build_catalog_from_parquet(fdf: pd.DataFrame) -> pd.DataFrame:
    cat = fdf.sort_values(by="valuation_date", ascending=False).groupby("player_id").first().reset_index()
    if "is_current_pl" in cat.columns:
        cat = cat[cat["is_current_pl"]]
    if "club_name" in cat.columns:
        cat = cat[cat["club_name"].isin(CURRENT_2026_PL_CLUBS)]

    cat["latest_recorded_val_eur"] = cat["target_market_value_eur"]
    cat["latest_age"] = cat["age_at_valuation"]

    meta_path = PROCESSED_DIR / "players_metadata.parquet"
    if meta_path.exists():
        try:
            meta_df = duckdb.read_parquet(str(meta_path).replace("\\", "/")).df()
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
        "player_id", "player_name", "image_url", "country_of_citizenship",
        "club_name", "position", "sub_position", "dominant_foot", "latest_age", "latest_recorded_val_eur"
    ]]


def get_player_metadata(player_id: int) -> tuple[str, str]:
    meta_path = PROCESSED_DIR / "players_metadata.parquet"
    if meta_path.exists():
        try:
            mdf = duckdb.read_parquet(str(meta_path).replace("\\", "/")).df()
            matched = mdf[mdf["player_id"] == player_id]
            if not matched.empty:
                return (
                    str(matched.iloc[0].get("image_url", "")),
                    str(matched.iloc[0].get("country_of_citizenship", "Unknown"))
                )
        except Exception as err:
            logger.debug(f"Player metadata lookup skipped: {err}")
    return "", "Unknown"


# Load active player catalog
def load_catalog() -> pd.DataFrame:
    if ACTIVE_CATALOG_PATH.exists():
        try:
            return duckdb.read_parquet(str(ACTIVE_CATALOG_PATH).replace("\\", "/")).df()
        except Exception as err:
            logger.debug(f"Parquet catalog read fallback: {err}")
            return pd.read_parquet(ACTIVE_CATALOG_PATH)

    if not PLAYERS_PATH.exists() or not VALUATIONS_PATH.exists():
        if FEATURES_PATH.exists():
            fdf = load_features_df()
            return build_catalog_from_parquet(fdf)
        return pd.DataFrame()


    con = duckdb.connect(database=":memory:")
    p_path = str(PLAYERS_PATH).replace("\\", "/")
    v_path = str(VALUATIONS_PATH).replace("\\", "/")
    a_path = str(APPEARANCES_PATH).replace("\\", "/")
    c_path = str(CLUBS_PATH).replace("\\", "/")
    s_path = str(STATS_PATH).replace("\\", "/")

    clubs_sql = "', '".join(CURRENT_2026_PL_CLUBS)

    query = f"""
        WITH latest_val AS (
            SELECT 
                player_id,
                date as val_date,
                TRY_CAST(market_value_in_eur AS BIGINT) as val_eur,
                current_club_name as val_club
            FROM read_csv_auto('{v_path}')
            WHERE market_value_in_eur IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) = 1
        ),
        latest_app AS (
            SELECT 
                a.player_id,
                a.date::DATE as app_date,
                c.name as app_club
            FROM read_csv_auto('{a_path}') a
            LEFT JOIN read_csv_auto('{c_path}') c ON a.player_club_id = c.club_id
            QUALIFY ROW_NUMBER() OVER (PARTITION BY a.player_id ORDER BY a.date::DATE DESC) = 1
        )
        SELECT 
            p.player_id,
            p.name as player_name,
            COALESCE(p.image_url, '') as image_url,
            COALESCE(p.country_of_citizenship, 'Unknown') as country_of_citizenship,
            COALESCE(lv.val_club, la.app_club, p.current_club_name) as club_name,
            COALESCE(p.position, 'Midfield') as position,
            COALESCE(p.sub_position, 'Central Midfield') as sub_position,
            COALESCE(p.foot, 'right') as dominant_foot,
            ROUND(DATEDIFF('day', p.date_of_birth::DATE, CURRENT_DATE) / 365.25, 1) as latest_age,
            COALESCE(lv.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) as latest_recorded_val_eur
        FROM read_csv_auto('{p_path}') p
        LEFT JOIN latest_val lv ON p.player_id = lv.player_id
        LEFT JOIN latest_app la ON p.player_id = la.player_id
        LEFT JOIN read_parquet('{s_path}') s ON p.player_id = s.player_id
        WHERE COALESCE(lv.val_club, la.app_club, p.current_club_name) IN ('{clubs_sql}')
          AND COALESCE(lv.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) > 0
          AND p.last_season >= 2025
        ORDER BY latest_recorded_val_eur DESC;
    """

    try:
        df = con.execute(query).df()
        con.close()
        return df
    except Exception as err:
        logger.warning(f"DuckDB catalog query failed, falling back to features.parquet: {err}")
        con.close()
        if FEATURES_PATH.exists():
            fdf = load_features_df()
            return build_catalog_from_parquet(fdf)
        return pd.DataFrame()


# Load player valuation trajectory
def load_history(player_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not VALUATIONS_PATH.exists():
        if FEATURES_PATH.exists():
            fdf = load_features_df()
            pdf = fdf[fdf["player_id"] == player_id].sort_values(by="valuation_date", ascending=True)
            if not pdf.empty:
                val_df = pd.DataFrame({
                    "valuation_date": pd.to_datetime(pdf["valuation_date"]),
                    "market_value_eur": pdf["target_market_value_eur"],
                    "club_name": pdf["club_name"]
                })
                val_df["val_change_eur"] = val_df["market_value_eur"].diff().fillna(0)
                prev_val = val_df["market_value_eur"].shift(1)
                val_df["val_change_pct"] = ((val_df["val_change_eur"] / prev_val) * 100.0).fillna(0)
                table_df = val_df.sort_values(by="valuation_date", ascending=False).copy()
                return val_df, table_df
        return pd.DataFrame(), pd.DataFrame()

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
    try:
        val_df = con.execute(val_query).df()
        con.close()
    except Exception as err:
        logger.warning(f"DuckDB valuation history failed: {err}")
        con.close()
        val_df = pd.DataFrame()

    if not val_df.empty:
        val_df["val_change_eur"] = val_df["market_value_eur"].diff().fillna(0)
        prev_val = val_df["market_value_eur"].shift(1)
        val_df["val_change_pct"] = ((val_df["val_change_eur"] / prev_val) * 100.0).fillna(0)

        table_df = val_df.sort_values(by="valuation_date", ascending=False).copy()
    else:
        table_df = pd.DataFrame(columns=["valuation_date", "club_name", "market_value_eur", "val_change_eur", "val_change_pct"])

    return val_df, table_df


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

    if not PLAYERS_PATH.exists() or not VALUATIONS_PATH.exists():
        if FEATURES_PATH.exists():
            fdf = load_features_df()
            pdf = fdf[fdf["player_id"] == player_id]
            if not pdf.empty:
                row = pdf.sort_values(by="valuation_date", ascending=False).iloc[0].to_dict()
                img_url, citizenship = get_player_metadata(player_id)
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
                    "target_market_value_eur": float(row.get("target_market_value_eur", 0.0)),
                    "current_market_value_eur": float(row.get("target_market_value_eur", 0.0)),
                    "prior_market_value_eur": float(row.get("prev_market_value_eur", row.get("target_market_value_eur", 0.0))),
                    "days_between_valuations": int(row.get("days_between_valuations", 180)),
                    "window_appearances": int(row.get("minutes_since_last_val", 900) // 90),
                    "window_minutes": int(row.get("minutes_since_last_val", 900)),
                    "window_goals": 0,
                    "window_assists": 0,
                    "window_yellow_cards": int(row.get("yellow_cards_since_val", 0)),
                    "window_red_cards": 0,
                    "window_european_minutes": int(row.get("european_minutes_played", 0)),
                    "goal_contributions_per_90": float(row.get("goal_contributions_per_90", 0.0)),
                    "minutes_since_last_val": int(row.get("minutes_since_last_val", 1000)),
                    "european_minutes_played": int(row.get("european_minutes_played", 0)),
                    "minutes_prior_window": int(row.get("minutes_prior_window", 1000)),
                    "contrib_per_90_prior": float(row.get("contrib_per_90_prior", 0.0)),
                    "yellow_cards_since_val": int(row.get("yellow_cards_since_val", 0)),
                }
        return {}

    con = duckdb.connect(database=":memory:")
    p_path = str(PLAYERS_PATH).replace("\\", "/")
    v_path = str(VALUATIONS_PATH).replace("\\", "/")
    a_path = str(APPEARANCES_PATH).replace("\\", "/")
    c_path = str(CLUBS_PATH).replace("\\", "/")
    s_path = str(STATS_PATH).replace("\\", "/")

    query = f"""
        WITH ranked_vals AS (
            SELECT 
                player_id,
                date as val_date,
                TRY_CAST(market_value_in_eur AS BIGINT) as val_eur,
                current_club_name as val_club,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM read_csv_auto('{v_path}')
            WHERE player_id = {player_id} AND market_value_in_eur IS NOT NULL
        ),
        v1 AS (
            SELECT * FROM ranked_vals WHERE rn = 1
        ),
        v2 AS (
            SELECT * FROM ranked_vals WHERE rn = 2
        ),
        latest_app AS (
            SELECT 
                a.player_id,
                c.name as app_club,
                c.domestic_competition_id as app_club_comp
            FROM read_csv_auto('{a_path}') a
            LEFT JOIN read_csv_auto('{c_path}') c ON a.player_club_id = c.club_id
            WHERE a.player_id = {player_id}
            ORDER BY a.date::DATE DESC
            LIMIT 1
        )
        SELECT 
            p.player_id,
            p.name as player_name,
            COALESCE(p.image_url, '') as image_url,
            COALESCE(p.country_of_citizenship, 'Unknown') as country_of_citizenship,
            COALESCE(v1.val_club, la.app_club, p.current_club_name) as club_name,
            COALESCE(p.position, 'Midfield') as position,
            COALESCE(p.sub_position, 'Central Midfield') as sub_position,
            COALESCE(p.foot, 'right') as dominant_foot,
            ROUND(DATEDIFF('day', p.date_of_birth::DATE, CURRENT_DATE) / 365.25, 1) as age_at_valuation,
            COALESCE(v1.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) as target_market_value_eur,
            COALESCE(v1.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) as current_market_value_eur,
            COALESCE(v2.val_eur, v1.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) as prior_market_value_eur,
            COALESCE(DATEDIFF('day', v2.val_date::DATE, v1.val_date::DATE), 180) as days_between_valuations,
            COALESCE(s.window_appearances, 0) as window_appearances,
            COALESCE(s.window_minutes, 0) as window_minutes,
            COALESCE(s.window_goals, 0) as window_goals,
            COALESCE(s.window_assists, 0) as window_assists,
            COALESCE(s.window_yellow_cards, 0) as window_yellow_cards,
            COALESCE(s.window_red_cards, 0) as window_red_cards,
            COALESCE(s.window_european_minutes, 0) as window_european_minutes
        FROM read_csv_auto('{p_path}') p
        LEFT JOIN v1 ON p.player_id = v1.player_id
        LEFT JOIN v2 ON p.player_id = v2.player_id
        LEFT JOIN latest_app la ON p.player_id = la.player_id
        LEFT JOIN read_parquet('{s_path}') s ON p.player_id = s.player_id
        WHERE p.player_id = {player_id}
        LIMIT 1;
    """
    try:
        df = con.execute(query).df()
        con.close()
    except Exception as err:
        logger.warning(f"DuckDB player lookup failed: {err}")
        con.close()
        if FEATURES_PATH.exists():
            fdf = load_features_df()
            pdf = fdf[fdf["player_id"] == player_id]
            if not pdf.empty:
                row = pdf.sort_values(by="valuation_date", ascending=False).iloc[0].to_dict()
                img_url, citizenship = get_player_metadata(player_id)
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
                    "target_market_value_eur": float(row.get("target_market_value_eur", 0.0)),
                    "current_market_value_eur": float(row.get("target_market_value_eur", 0.0)),
                    "prior_market_value_eur": float(row.get("prev_market_value_eur", row.get("target_market_value_eur", 0.0))),
                    "days_between_valuations": int(row.get("days_between_valuations", 180)),
                    "window_appearances": int(row.get("minutes_since_last_val", 900) // 90),
                    "window_minutes": int(row.get("minutes_since_last_val", 900)),
                    "window_goals": 0,
                    "window_assists": 0,
                    "window_yellow_cards": int(row.get("yellow_cards_since_val", 0)),
                    "window_red_cards": 0,
                    "window_european_minutes": int(row.get("european_minutes_played", 0)),
                    "goal_contributions_per_90": float(row.get("goal_contributions_per_90", 0.0)),
                    "minutes_since_last_val": int(row.get("minutes_since_last_val", 1000)),
                    "european_minutes_played": int(row.get("european_minutes_played", 0)),
                    "minutes_prior_window": int(row.get("minutes_prior_window", 1000)),
                    "contrib_per_90_prior": float(row.get("contrib_per_90_prior", 0.0)),
                    "yellow_cards_since_val": int(row.get("yellow_cards_since_val", 0)),
                }
        return {}

    if df.empty:
        return {}

    data = df.iloc[0].to_dict()

    # Compute contribution rate
    minutes = float(data.get("window_minutes", 0))
    goals = float(data.get("window_goals", 0))
    assists = float(data.get("window_assists", 0))
    if minutes >= 90:
        data["goal_contributions_per_90"] = min(round(((goals + assists) * 90.0) / minutes, 4), 4.0)
    else:
        data["goal_contributions_per_90"] = 0.0

    data["minutes_since_last_val"] = int(data.get("window_minutes", 1000))
    data["european_minutes_played"] = int(data.get("window_european_minutes", 0))
    data["yellow_cards_since_val"] = int(data.get("window_yellow_cards", 0))
    data["minutes_prior_window"] = 1000
    data["contrib_per_90_prior"] = 0.0

    if FEATURES_PATH.exists():
        try:
            fdf = load_features_df()
            pdf = fdf[fdf["player_id"] == player_id]
            if not pdf.empty:
                latest_f = pdf.sort_values(by="valuation_date", ascending=False).iloc[0]
                data["minutes_prior_window"] = int(latest_f.get("minutes_prior_window", 1000))
                data["contrib_per_90_prior"] = float(latest_f.get("contrib_per_90_prior", 0.0))
        except Exception as err:
            logger.debug(f"Prior window features lookup skipped: {err}")

    return data