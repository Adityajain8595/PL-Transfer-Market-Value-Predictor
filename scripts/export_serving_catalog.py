
import duckdb

from src.serve.clubs import CURRENT_2026_PL_CLUBS
from src.utils.logger import logger
from src.utils.paths import PROCESSED_DIR, RAW_DIR


def export_active_catalog_and_features() -> None:
    """Pre-materialize active player catalog and feature cache for fast, consistent serving."""
    players_path = str(RAW_DIR / "players.csv").replace("\\", "/")
    valuations_path = str(RAW_DIR / "player_valuations.csv").replace("\\", "/")
    appearances_path = str(RAW_DIR / "appearances.csv").replace("\\", "/")
    clubs_path = str(RAW_DIR / "clubs.csv").replace("\\", "/")
    stats_path = str(PROCESSED_DIR / "player_latest_stats.parquet").replace("\\", "/")

    clubs_sql = "', '".join(CURRENT_2026_PL_CLUBS)
    con = duckdb.connect(database=":memory:")

    catalog_query = f"""
        WITH latest_val AS (
            SELECT 
                player_id,
                date as val_date,
                TRY_CAST(market_value_in_eur AS BIGINT) as val_eur,
                current_club_name as val_club
            FROM read_csv_auto('{valuations_path}')
            WHERE market_value_in_eur IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) = 1
        ),
        latest_app AS (
            SELECT 
                a.player_id,
                a.date::DATE as app_date,
                c.name as app_club
            FROM read_csv_auto('{appearances_path}') a
            LEFT JOIN read_csv_auto('{clubs_path}') c ON a.player_club_id = c.club_id
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
        FROM read_csv_auto('{players_path}') p
        LEFT JOIN latest_val lv ON p.player_id = lv.player_id
        LEFT JOIN latest_app la ON p.player_id = la.player_id
        LEFT JOIN read_parquet('{stats_path}') s ON p.player_id = s.player_id
        WHERE COALESCE(lv.val_club, la.app_club, p.current_club_name) IN ('{clubs_sql}')
          AND COALESCE(lv.val_eur, TRY_CAST(p.market_value_in_eur AS BIGINT)) > 0
          AND NOT (COALESCE(s.window_appearances, 0) = 0 AND COALESCE(s.window_minutes, 0) = 0 AND (ROUND(DATEDIFF('day', p.date_of_birth::DATE, CURRENT_DATE) / 365.25, 1) >= 35 OR COALESCE(lv.val_eur, 0) < 1000000))
        ORDER BY latest_recorded_val_eur DESC;
    """

    cat_df = con.execute(catalog_query).df()
    cat_path = PROCESSED_DIR / "active_players_catalog.parquet"
    cat_df.to_parquet(cat_path, index=False)
    logger.info(f"Saved active catalog ({len(cat_df)} players) to {cat_path}")

    pids_sql = ", ".join(str(int(pid)) for pid in cat_df["player_id"])
    features_query = f"""
        WITH ranked_vals AS (
            SELECT 
                player_id,
                date as val_date,
                TRY_CAST(market_value_in_eur AS BIGINT) as val_eur,
                current_club_name as val_club,
                ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) as rn
            FROM read_csv_auto('{valuations_path}')
            WHERE player_id IN ({pids_sql}) AND market_value_in_eur IS NOT NULL
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
            FROM read_csv_auto('{appearances_path}') a
            LEFT JOIN read_csv_auto('{clubs_path}') c ON a.player_club_id = c.club_id
            WHERE a.player_id IN ({pids_sql})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY a.player_id ORDER BY a.date::DATE DESC) = 1
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
        FROM read_csv_auto('{players_path}') p
        LEFT JOIN v1 ON p.player_id = v1.player_id
        LEFT JOIN v2 ON p.player_id = v2.player_id
        LEFT JOIN latest_app la ON p.player_id = la.player_id
        LEFT JOIN read_parquet('{stats_path}') s ON p.player_id = s.player_id
        WHERE p.player_id IN ({pids_sql});
    """

    features_df = con.execute(features_query).df()

    minutes = features_df["window_minutes"].astype(float)
    goals = features_df["window_goals"].astype(float)
    assists = features_df["window_assists"].astype(float)

    features_df["goal_contributions_per_90"] = 0.0
    valid_mask = minutes >= 90
    features_df.loc[valid_mask, "goal_contributions_per_90"] = (
        ((goals[valid_mask] + assists[valid_mask]) * 90.0) / minutes[valid_mask]
    ).round(4).clip(upper=4.0)

    features_df["minutes_since_last_val"] = features_df["window_minutes"].astype(int)
    features_df["european_minutes_played"] = features_df["window_european_minutes"].astype(int)
    features_df["yellow_cards_since_val"] = features_df["window_yellow_cards"].astype(int)
    features_df["minutes_prior_window"] = features_df["minutes_since_last_val"]
    features_df["contrib_per_90_prior"] = features_df["goal_contributions_per_90"]

    feat_path = PROCESSED_DIR / "active_players_features.parquet"
    features_df.to_parquet(feat_path, index=False)
    logger.info(f"Saved active features ({len(features_df)} records) to {feat_path}")
    con.close()


if __name__ == "__main__":
    export_active_catalog_and_features()
