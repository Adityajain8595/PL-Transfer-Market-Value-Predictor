import json

import duckdb
import pandas as pd

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


# Load champion model metrics dynamically
def load_model_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Could not read champion metrics file: {err}")
    return {
        "model_name": "LightGBM",
        "test_mae": 3094428.0,
        "test_rmse": 5019762.0,
        "test_r2": 0.9528
    }


# Load tournament comparison metrics dynamically
def load_tournament_metrics() -> dict:
    if TOURNAMENT_PATH.exists():
        try:
            with open(TOURNAMENT_PATH, "r") as f:
                return json.load(f)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Could not read tournament metrics file: {err}")
    return {
        "LightGBM": {"val_mae": 2618511.0, "val_rmse": 4593406.0, "val_r2": 0.9587},
        "XGBoost": {"val_mae": 2642194.0, "val_rmse": 4714736.0, "val_r2": 0.9565},
        "RandomForest": {"val_mae": 2700654.0, "val_rmse": 4908480.0, "val_r2": 0.9528},
        "GradientBoosting": {"val_mae": 2786100.0, "val_rmse": 4938663.0, "val_r2": 0.9523},
        "CatBoost": {"val_mae": 2836798.0, "val_rmse": 5124272.0, "val_r2": 0.9486},
    }


# Load dataset summary statistics dynamically directly from processed features
def load_dataset_stats() -> dict:
    if FEATURES_PATH.exists():
        try:
            df = pd.read_parquet(FEATURES_PATH)
            dates = pd.to_datetime(df["valuation_date"])
            min_yr = int(dates.min().year)
            max_yr = int(dates.max().year)
            feature_cols = [
                c for c in df.columns
                if c not in ["player_id", "valuation_date", "prev_valuation_date", "target_market_value_eur", "log_target_market_value"]
            ]
            avg_days = int(df["days_between_valuations"].mean()) if "days_between_valuations" in df.columns else 194
            return {
                "total_records": len(df),
                "min_year": min_yr,
                "max_year": max_yr,
                "feature_count": len(feature_cols),
                "players_count": int(df["player_id"].nunique()),
                "avg_days_between": avg_days,
            }
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Could not compute dataset stats: {err}")
    return {
        "total_records": 12476,
        "min_year": 2013,
        "max_year": 2026,
        "feature_count": 13,
        "players_count": 1850,
        "avg_days_between": 194,
    }



CURRENT_2026_PL_CLUBS = [
    "AFC Bournemouth",
    "Arsenal FC",
    "Aston Villa",
    "Brentford FC",
    "Brighton & Hove Albion",
    "Chelsea FC",
    "Coventry City",          # Promoted for 2026/27
    "Crystal Palace",
    "Everton FC",
    "Fulham FC",
    "Hull City",              # Promoted for 2026/27
    "Ipswich Town",           # Promoted for 2026/27
    "Leeds United",
    "Liverpool FC",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland AFC",
    "Tottenham Hotspur",
]


# Load verified active Premier League player catalog cross-checking latest valuations & matches
def load_catalog() -> pd.DataFrame:
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
          AND NOT (COALESCE(s.window_appearances, 0) = 0 AND COALESCE(s.window_minutes, 0) = 0 AND (ROUND(DATEDIFF('day', p.date_of_birth::DATE, CURRENT_DATE) / 365.25, 1) >= 35 OR COALESCE(lv.val_eur, 0) < 1000000))
        ORDER BY latest_recorded_val_eur DESC;
    """

    df = con.execute(query).df()
    con.close()
    return df


# Load player historical valuation records with club progression
def load_history(player_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    else:
        table_df = pd.DataFrame(columns=["valuation_date", "club_name", "market_value_eur", "val_change_eur", "val_change_pct"])

    return val_df, table_df


# Get latest verified player profile, true latest club, real match statistics, target value, and prior value
def get_player_features(player_id: int) -> dict:
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
    df = con.execute(query).df()
    con.close()

    if df.empty:
        return {}

    data = df.iloc[0].to_dict()

    # Calculate G+A / 90 rate
    minutes = float(data.get("window_minutes", 0))
    goals = float(data.get("window_goals", 0))
    assists = float(data.get("window_assists", 0))
    if minutes >= 90:
        data["goal_contributions_per_90"] = min(round(((goals + assists) * 90.0) / minutes, 4), 4.0)
    else:
        data["goal_contributions_per_90"] = 0.0

    return data