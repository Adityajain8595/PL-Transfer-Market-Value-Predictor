from pathlib import Path
import duckdb

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "premier_league_features.parquet"


def run_feature_pipeline():
    con = duckdb.connect(database=":memory:")
    print("[INFO] Registering raw CSV files into DuckDB in-memory engine...")

    # Load relevant tables
    con.execute(f"""
    CREATE VIEW players AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/players.csv', normalize_names=True);

    CREATE VIEW valuations AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/player_valuations.csv', normalize_names=True);

    CREATE VIEW appearances AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/appearances.csv', normalize_names=True);

    CREATE VIEW games AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/games.csv', normalize_names=True);
    
    CREATE VIEW clubs AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/clubs.csv', normalize_names=True);

    CREATE VIEW competitions AS
    SELECT * FROM read_csv_auto('{RAW_DIR}/competitions.csv', normalize_names=True);
    """)

    print("[INFO] Constructing point-in-time temporal features matrix...")

    feature_query = f"""
        CREATE TABLE final_feature_matrix AS
        WITH target_valuations AS (
            SELECT
                v.player_id,
                v.date::DATE AS valuation_date,
                TRY_CAST(v.market_value_in_eur AS BIGINT) AS target_market_value_eur,
                v.current_club_id,
                p._name AS player_name,
                p.date_of_birth::DATE AS date_of_birth,
                p.position,
                p.sub_position,
                p.foot,
                TRY_CAST(p.height_in_cm AS INTEGER) AS height_in_cm,
                ROUND(DATEDIFF('day', p.date_of_birth::DATE, v.date::DATE) / 365.25, 2) AS age_at_valuation
            FROM valuations v
            JOIN players p ON v.player_id = p.player_id
            WHERE v.market_value_in_eur IS NOT NULL
                AND TRY_CAST(v.market_value_in_eur AS BIGINT) > 0
                AND p.date_of_birth IS NOT NULL
                AND v.date >= '2015-01-01'
        ),
        
        player_club_context AS (
            SELECT
                tv.*,
                c._name AS club_name,
                c.domestic_competition_id,
                TRY_CAST(c.total_market_value AS BIGINT) AS club_total_market_value
            FROM target_valuations tv
            LEFT JOIN clubs c ON tv.current_club_id = c.club_id
            WHERE c.domestic_competition_id = 'GB1'
        ),

        appearance_aggregates AS (
            SELECT 
                pcc.player_id,
                pcc.valuation_date,
                COALESCE(COUNT(a.appearance_id), 0) AS appearances_last_365d,
                COALESCE(SUM(TRY_CAST(a.minutes_played AS INTEGER)), 0) AS minutes_played_last_365d,

                COALESCE(SUM(TRY_CAST(a.goals AS INTEGER)), 0) AS goals_last_365d,
                COALESCE(SUM(TRY_CAST(a.assists AS INTEGER)), 0) AS assists_last_365d,

                COALESCE(SUM(TRY_CAST(a.yellow_cards AS INTEGER)), 0) AS yellow_cards_last_365d,
                COALESCE(SUM(TRY_CAST(a.red_cards AS INTEGER)), 0) AS red_cards_last_365d,

                COALESCE(SUM(CASE 
                    WHEN g.competition_id IN ('CL', 'EL', 'ECL') THEN TRY_CAST(a.minutes_played AS INTEGER)
                    ELSE 0
                END), 0) AS european_minutes_played_last_365d

            FROM player_club_context pcc
            LEFT JOIN appearances a
                ON pcc.player_id = a.player_id
                AND a.date::DATE < pcc.valuation_date
                AND a.date::DATE >= (pcc.valuation_date - INTERVAL 365 DAY)
            LEFT JOIN games g
                ON a.game_id = g.game_id
            GROUP BY pcc.player_id, pcc.valuation_date
        )

        SELECT 
            pcc.player_id,
            pcc.player_name,
            pcc.valuation_date,
            pcc.age_at_valuation,
            pcc.position,
            pcc.sub_position,
            COALESCE(pcc.foot, 'unknown') AS dominant_foot,
            COALESCE(pcc.height_in_cm, 182) AS height_in_cm,
            pcc.club_name,

            aa.appearances_last_365d,
            aa.minutes_played_last_365d,
            aa.goals_last_365d,
            aa.assists_last_365d,
            aa.yellow_cards_last_365d,
            aa.red_cards_last_365d,
            aa.european_minutes_played_last_365d,

            ROUND(CASE 
                WHEN aa.minutes_played_last_365d > 0 THEN (aa.goals_last_365d * 90.0) / aa.minutes_played_last_365d
                ELSE 0.0
            END, 4) AS goals_per_90_last_365d,

            ROUND(CASE 
                WHEN aa.minutes_played_last_365d > 0 THEN (aa.assists_last_365d * 90.0) / aa.minutes_played_last_365d
                ELSE 0.0
            END, 4) AS assists_per_90_last_365d,

            ROUND(CASE 
                WHEN aa.minutes_played_last_365d > 0 THEN ((aa.goals_last_365d + aa.assists_last_365d) * 90.0) / aa.minutes_played_last_365d
                ELSE 0.0
            END, 4) AS goal_contributions_per_90_last_365d,
            
            pcc.target_market_value_eur,
            ROUND(LN(pcc.target_market_value_eur), 4) AS log_target_market_value
        FROM player_club_context pcc
        JOIN appearance_aggregates aa 
            ON pcc.player_id = aa.player_id
            AND pcc.valuation_date = aa.valuation_date
        WHERE pcc.age_at_valuation BETWEEN 15 AND 42
        ORDER BY pcc.valuation_date ASC;
        """

    con.execute(feature_query)
    summary_stats = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT player_id) AS unique_players,
            MIN(valuation_date) AS earliest_date,
            MAX(valuation_date) AS latest_date,
            ROUND(AVG(target_market_value_eur), 0) AS avg_value_eur,
            ROUND(MEDIAN(target_market_value_eur), 0) AS median_value_eur    
        FROM final_feature_matrix;
        """).df()

    print("\n[INFO] Summary statistics of the final feature matrix:")
    print(summary_stats.to_string(index=False))

    print(f"\n[INFO] Exporting feature dataset to {OUTPUT_FILE}...")
    con.execute(f"COPY final_feature_matrix TO '{OUTPUT_FILE}' (FORMAT PARQUET, COMPRESSION SNAPPY);")

    con.close()
    print("[SUCCESS] Feature pipeline completed!")

if __name__ == "__main__":
    run_feature_pipeline()