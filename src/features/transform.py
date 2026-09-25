import sys
import time

import duckdb
import yaml

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.paths import CONFIG_DIR, PROCESSED_DIR, RAW_DIR

CONFIG_PATH = CONFIG_DIR / "data_config.yaml"

# Feature transformation pipeline
def transform_features() -> None:
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        raw_dir = str(RAW_DIR).replace("\\", "/")
        out_dir = PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = str(out_dir / cfg["processed_data"]["parquet_filename"]).replace("\\", "/")

        logger.info("Starting match-based feature engineering with DuckDB...")
        start_time = time.time()
        con = duckdb.connect()

        min_val_year = cfg["filters"]["min_valuation_year"]
        raw_cutoff = cfg["filters"]["raw_valuation_cutoff"]
        competition_id = cfg["filters"]["competition_id"]
        min_age = cfg["filters"]["min_age"]
        max_age = cfg["filters"]["max_age"]
        min_days = cfg["filters"]["min_days_between"]
        max_days = cfg["filters"]["max_days_between"]

        query = f"""
        WITH val_prep AS (
            SELECT 
                TRY_CAST(player_id AS BIGINT) as player_id,
                date::DATE as valuation_date,
                TRY_CAST(market_value_in_eur AS BIGINT) as target_market_value_eur
            FROM read_csv_auto('{raw_dir}/player_valuations.csv')
            WHERE market_value_in_eur IS NOT NULL 
              AND TRY_CAST(market_value_in_eur AS BIGINT) > 0
              AND date::DATE >= '{raw_cutoff}'
        ),
        player_prep AS (
            SELECT 
                TRY_CAST(player_id AS BIGINT) as player_id,
                name as player_name,
                date_of_birth::DATE as date_of_birth,
                position,
                sub_position,
                COALESCE(foot, 'unknown') as dominant_foot,
                (COALESCE(current_club_domestic_competition_id, '') = '{competition_id}') as is_current_pl
            FROM read_csv_auto('{raw_dir}/players.csv')
            WHERE date_of_birth IS NOT NULL
        ),
        joined_val AS (
            SELECT 
                v.player_id,
                v.valuation_date,
                v.target_market_value_eur,
                p.player_name,
                p.position,
                p.sub_position,
                p.dominant_foot,
                p.is_current_pl,
                ROUND(DATEDIFF('day', p.date_of_birth, v.valuation_date) / 365.25, 2) as age_at_valuation
            FROM val_prep v
            JOIN player_prep p ON v.player_id = p.player_id
        ),
        ranked_val AS (
            SELECT 
                *,
                LAG(target_market_value_eur, 1) OVER (PARTITION BY player_id ORDER BY valuation_date) as prev_market_value_eur,
                LAG(valuation_date, 1) OVER (PARTITION BY player_id ORDER BY valuation_date) as prev_valuation_date
            FROM joined_val
        ),
        app_in_window AS (
            SELECT 
                rv.player_id,
                rv.valuation_date,
                a.player_club_id,
                a.competition_id,
                a.minutes_played,
                a.goals,
                a.assists,
                a.yellow_cards,
                ROW_NUMBER() OVER (PARTITION BY rv.player_id, rv.valuation_date ORDER BY a.date::DATE DESC) as rn
            FROM ranked_val rv
            JOIN read_csv_auto('{raw_dir}/appearances.csv') a
              ON a.player_id = rv.player_id
             AND a.date::DATE > rv.prev_valuation_date
             AND a.date::DATE <= rv.valuation_date
        ),
        window_stats AS (
            SELECT 
                player_id,
                valuation_date,
                MAX(CASE WHEN rn = 1 THEN player_club_id END) as window_club_id,
                SUM(CASE WHEN competition_id = '{competition_id}' THEN 1 ELSE 0 END) as gb1_appearances,
                SUM(minutes_played) as minutes_since_last_val,
                SUM(goals) as goals_since_last_val,
                SUM(assists) as assists_since_last_val,
                SUM(yellow_cards) as yellow_cards_since_val,
                SUM(CASE WHEN competition_id IN ('CL', 'EL', 'ECL') THEN minutes_played ELSE 0 END) as european_minutes_played
            FROM app_in_window
            GROUP BY player_id, valuation_date
            HAVING SUM(CASE WHEN competition_id = '{competition_id}' THEN 1 ELSE 0 END) > 0
        ),
        club_prep AS (
            SELECT 
                TRY_CAST(club_id AS BIGINT) as club_id,
                name as club_name
            FROM read_csv_auto('{raw_dir}/clubs.csv')
        ),
        filtered_pairs AS (
            SELECT 
                r.*,
                COALESCE(c.club_name, 'Unknown Club') as club_name,
                DATEDIFF('day', r.prev_valuation_date, r.valuation_date) as days_between_valuations,
                ws.minutes_since_last_val,
                ws.goals_since_last_val,
                ws.assists_since_last_val,
                ws.yellow_cards_since_val,
                ws.european_minutes_played
            FROM ranked_val r
            JOIN window_stats ws ON r.player_id = ws.player_id AND r.valuation_date = ws.valuation_date
            LEFT JOIN club_prep c ON ws.window_club_id = c.club_id
            WHERE r.prev_market_value_eur IS NOT NULL
              AND r.valuation_date >= '{min_val_year}'
              AND DATEDIFF('day', r.prev_valuation_date, r.valuation_date) BETWEEN {min_days} AND {max_days}
              AND r.age_at_valuation BETWEEN {min_age} AND {max_age}
        ),
        stats_prior AS (
            SELECT 
                fp.player_id,
                fp.valuation_date,
                COALESCE(SUM(a.minutes_played), 0) as minutes_prior_window,
                COALESCE(SUM(a.goals), 0) as goals_prior_window,
                COALESCE(SUM(a.assists), 0) as assists_prior_window
            FROM filtered_pairs fp
            LEFT JOIN read_csv_auto('{raw_dir}/appearances.csv') a 
              ON fp.player_id = a.player_id 
             AND a.date::DATE > (fp.prev_valuation_date - INTERVAL 365 DAY)
             AND a.date::DATE <= fp.prev_valuation_date
            GROUP BY fp.player_id, fp.valuation_date
        )
        SELECT DISTINCT
            fp.player_id,
            fp.player_name,
            fp.valuation_date,
            fp.prev_valuation_date,
            fp.target_market_value_eur,
            fp.prev_market_value_eur,
            fp.position,
            fp.sub_position,
            fp.dominant_foot,
            fp.club_name,
            fp.age_at_valuation,
            fp.days_between_valuations,
            ROUND(LN(fp.prev_market_value_eur + 1), 4) as log_last_known_value,
            ROUND(LN(fp.target_market_value_eur + 1), 4) as log_target_market_value,
            fp.minutes_since_last_val,
            fp.yellow_cards_since_val,
            fp.european_minutes_played,
            CASE 
                WHEN fp.minutes_since_last_val >= 90 
                THEN LEAST(ROUND(((fp.goals_since_last_val + fp.assists_since_last_val) * 90.0) / fp.minutes_since_last_val, 4), 4.0)
                ELSE 0.0 
            END as goal_contributions_per_90,
            sp.minutes_prior_window,
            CASE 
                WHEN sp.minutes_prior_window >= 90 
                THEN LEAST(ROUND(((sp.goals_prior_window + sp.assists_prior_window) * 90.0) / sp.minutes_prior_window, 4), 4.0)
                ELSE 0.0 
            END as contrib_per_90_prior,
            fp.is_current_pl
        FROM filtered_pairs fp
        JOIN stats_prior sp ON fp.player_id = sp.player_id AND fp.valuation_date = sp.valuation_date
        WHERE fp.minutes_since_last_val > 0
          AND fp.target_market_value_eur > 0
          AND fp.prev_market_value_eur > 0
        QUALIFY ROW_NUMBER() OVER (PARTITION BY fp.player_id, fp.valuation_date ORDER BY fp.prev_valuation_date DESC) = 1
        ORDER BY fp.valuation_date
        """

        con.execute(f"COPY ({query}) TO '{parquet_path}' (FORMAT PARQUET)")
        con.close()

        elapsed = time.time() - start_time
        logger.info(f"Match-based transformation complete in {elapsed:.2f}s -> {parquet_path}")

    except Exception as err:
        logger.error("Feature engineering failed.")
        raise CustomException(err, sys) from err

if __name__ == "__main__":
    transform_features()