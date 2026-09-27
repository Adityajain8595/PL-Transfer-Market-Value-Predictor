import base64
import os
import sys
from pathlib import Path

# Add project root to sys.path for cloud deployment
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.serve.dashboard_utils import (
    get_player_features,
    load_catalog,
    load_dataset_stats,
    load_history,
    load_model_metrics,
    load_tournament_metrics,
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# Assets
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "pl_logo.png"
FAVICON_PATH = ASSETS_DIR / "pl_favicon.png"
BACKDROP_PATH = ASSETS_DIR / "backdrop.png"

fav_icon = Image.open(FAVICON_PATH) if FAVICON_PATH.exists() else None

# Page Configuration
st.set_page_config(
    page_title="Premier League Transfer Valuation",
    page_icon=fav_icon if fav_icon else "PL",
    layout="wide",
    initial_sidebar_state="collapsed",
)

backdrop_css = ""
if BACKDROP_PATH.exists():
    b64_img = base64.b64encode(BACKDROP_PATH.read_bytes()).decode()
    backdrop_css = f"""
    .stApp {{
        background: linear-gradient(135deg, rgba(7, 11, 20, 0.92) 0%, rgba(11, 16, 29, 0.95) 100%),
                    url("data:image/png;base64,{b64_img}") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: top right !important;
    }}
    """

# UI Styling and Theme
st.markdown(f"""
    <style>
    {backdrop_css}

    /* Core typography & layout */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #F1F5F9;
    }}
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }}

    .hero-title {{
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }}
    .hero-subtitle {{
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 3px;
        margin-bottom: 10px;
    }}

    /* Stat Badges */
    .stat-badge {{
        display: inline-block;
        padding: 4px 12px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 6px;
        color: #E2E8F0;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 8px;
        letter-spacing: 0.3px;
    }}
    .stat-badge-accent {{
        background: rgba(0, 242, 254, 0.1);
        border-color: rgba(0, 242, 254, 0.3);
        color: #00f2fe;
    }}
    .stat-badge-green {{
        background: rgba(0, 255, 135, 0.1);
        border-color: rgba(0, 255, 135, 0.3);
        color: #00FF87;
    }}

    /* Clear Line Separators */
    .divider-line {{
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        margin: 1.8rem 0;
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        background: #0052d4 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        height: 42px !important;
        transition: all 0.2s ease !important;
    }}
    .stButton>button:hover {{
        background: #0066ff !important;
        border-color: #00f2fe !important;
    }}

    /* Back Button */
    .back-btn>button {{
        background: transparent !important;
        color: #00f2fe !important;
        border: 1px solid rgba(0, 242, 254, 0.35) !important;
        border-radius: 6px !important;
        height: 36px !important;
        font-weight: 600 !important;
        width: auto !important;
        padding: 0 16px !important;
        margin-bottom: 12px !important;
    }}
    .back-btn>button:hover {{
        background: rgba(0, 242, 254, 0.1) !important;
    }}

    /* Inputs */
    div[data-baseweb="input"] {{
        background-color: rgba(18, 24, 38, 0.9) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: rgba(18, 24, 38, 0.9) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.8rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    </style>
""", unsafe_allow_html=True)


# Data Access Helpers
@st.cache_data(show_spinner=False)
def cached_catalog():
    return load_catalog()


@st.cache_data(show_spinner=False)
def cached_history(pid: int):
    return load_history(player_id=pid)


@st.cache_data(show_spinner=False)
def cached_features(pid: int):
    return get_player_features(player_id=pid)


# Catalog and model metrics
catalog = cached_catalog()
model_metrics = load_model_metrics()
dataset_stats = load_dataset_stats()

test_mae = float(model_metrics.get("test_mae", 3094428.0))
test_r2 = float(model_metrics.get("test_r2", 0.9528))
model_name = str(model_metrics.get("model_name", "LightGBM"))
cutoff_date_str = str(dataset_stats.get("cutoff_date", "June 2026"))

# Selection state
if "selected_player_id" not in st.session_state:
    st.session_state.selected_player_id = None


# Header Section
header_col1, header_col2, header_col3 = st.columns([1, 8.2, 3.8])

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=72)

with header_col2:
    st.markdown('<div class="hero-title">Premier League Transfer Valuation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Live Valuation Service for Active Premier League Squads</div>', unsafe_allow_html=True)

    total_active_players = len(catalog)
    total_active_clubs = catalog["club_name"].nunique() if not catalog.empty else 20
    st.markdown(f"""
        <div>
            <span class="stat-badge stat-badge-accent">PLAYERS: {total_active_players:,}</span>
            <span class="stat-badge">CLUBS: {total_active_clubs}</span>
            <span class="stat-badge stat-badge-green">MODEL: {model_name}</span>
            <span class="stat-badge">MAE: ±EUR {test_mae:,.0f}</span>
            <span class="stat-badge stat-badge-green">FIT: {test_r2 * 100:.1f}% R²</span>
        </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: flex-end; justify-content: flex-start; height: 100%; padding-top: 4px;">
            <div style="background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.28); border-radius: 8px; padding: 6px 14px; text-align: right;">
                <div style="color: #94A3B8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">DATASET CUTOFF</div>
                <div style="color: #00f2fe; font-size: 0.95rem; font-weight: 700;">{cutoff_date_str}</div>
            </div>
            <div style="color: #64748B; font-size: 0.72rem; margin-top: 4px; text-align: right; line-height: 1.35;">
                Valuations & squads as of season close.<br>Transfers post-cutoff not reflected.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


# Directory View
if st.session_state.selected_player_id is None:
    # Pipeline Overview and Architecture
    with st.expander("Pipeline Insights & Architecture", expanded=True):
        tab_pipe, tab_data, tab_model, tab_stack = st.tabs([
            "Pipeline Architecture",
            "Dataset Insights",
            "Model Tournament",
            "Tech Stack"
        ])

        with tab_pipe:
            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 24px; padding-bottom: 8px;">
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00f2fe; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 1: Ingestion & Curation</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">Final Processed Dataset</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">12,476 curated evaluation windows across 1,904 historical players, with active match appearances and strict deduplication.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00f2fe; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 2: Feature Store</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">DuckDB Match Windows</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Analytical SQL calculating log valuation anchors, G+A/90, discipline, European volume, and deduplication.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00f2fe; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 3: Validation</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">Rolling Temporal Split</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Strict chronological separation (12-mo val, 12-mo test) preventing lookahead and target leakage.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00FF87; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 4: Tournament</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">5-Candidate Arena</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Automated tournament across LightGBM, XGBoost, CatBoost, Random Forest, and Gradient Boosting.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00FF87; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 5: Governance</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">DagsHub MLflow Registry</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Regression quality gate (&le; 5% degradation check) with automated semantic model versioning.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                    <div style="color: #00f2fe; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;">Stage 6: Serving</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-top: 2px;">FastAPI & Drift Guard</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">High-throughput async inference microservice paired with Evidently AI continuous data drift detection.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab_data:
            d_col1, d_col2 = st.columns([1.1, 1.4])
            dataset_stats = load_dataset_stats()
            with d_col1:
                st.markdown(f"""
                <div style="padding: 10px; line-height: 1.8; color: #E2E8F0; font-size: 0.88rem;">
                    <b>Dataset Profile & Quality Controls:</b><br>
                    • <b>Total Feature Records:</b> {dataset_stats['total_records']:,} curated evaluation windows<br>
                    • <b>Historical Horizon:</b> {dataset_stats['min_year']} – {dataset_stats['max_year']} ({dataset_stats['max_year'] - dataset_stats['min_year'] + 1}-year longitudinal depth)<br>
                    • <b>Unique Players in History:</b> {dataset_stats['players_count']:,} tracked across all seasons<br>
                    • <b>Active Premier League Squad:</b> {len(catalog):,} match-verified players across {catalog['club_name'].nunique() if not catalog.empty else 20} clubs<br>
                    • <b>Average Valuation Cadence:</b> ~{dataset_stats['avg_days_between']} days between updates<br>
                    • <b>Feature Dimensions:</b> {dataset_stats['feature_count']} engineered predictors<br>
                    • <b>Pruning & Quality Controls:</b> Strict deduplication by (player_id, date), exclusion of zero-minute/retired records<br>
                    • <b>Target Formulation:</b> ln(1 + Market Value EUR) with expm1 inverse clipping
                </div>
                """, unsafe_allow_html=True)
            with d_col2:
                pos_vals = catalog.groupby("position")["latest_recorded_val_eur"].mean().reset_index()
                pos_vals["val_m"] = pos_vals["latest_recorded_val_eur"] / 1_000_000.0
                fig_pos = go.Figure(go.Bar(
                    x=pos_vals["position"],
                    y=pos_vals["val_m"],
                    marker={"color": ["#00f2fe", "#00FF87", "#3b82f6", "#a855f7"]},
                    text=pos_vals["val_m"].apply(lambda v: f"€{v:.1f}M"),
                    textposition="auto"
                ))
                fig_pos.update_layout(
                    title={"text": "Average Market Valuation by Position (Active PL)", "font": {"color": "#FFFFFF", "size": 13}},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis={"tickfont": {"color": "#94A3B8"}, "showgrid": False},
                    yaxis={"title": {"text": "Avg Valuation (€M)", "font": {"color": "#94A3B8"}}, "tickfont": {"color": "#94A3B8"}, "showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
                    height=240,
                    margin={"l": 10, "r": 10, "t": 35, "b": 10}
                )
                st.plotly_chart(fig_pos, width="stretch")

        with tab_model:
            m_col1, m_col2 = st.columns([1.5, 1.1])
            tourn_metrics = load_tournament_metrics()
            champ_name = str(model_metrics.get("model_name", "LightGBM"))

            # Rank candidates by validation error
            sorted_models = sorted(
                tourn_metrics.items(),
                key=lambda x: x[1].get("val_mae", 99999999)
            )

            chart_names = []
            chart_maes = []
            chart_colors = []
            palette = ["#00FF87", "#00f2fe", "#38bdf8", "#818cf8", "#c084fc", "#f472b6"]

            for i, (m_name, m_data) in enumerate(sorted_models):
                display_label = f"{m_name} (Champion)" if m_name == champ_name else m_name
                mae_in_m = m_data.get("val_mae", 0.0) / 1_000_000.0
                chart_names.append(display_label)
                chart_maes.append(mae_in_m)
                chart_colors.append(palette[i % len(palette)])

            with m_col1:
                fig_tourn = go.Figure(go.Bar(
                    x=chart_maes,
                    y=chart_names,
                    orientation="h",
                    marker={"color": chart_colors},
                    text=[f"€{v:.3f}M" for v in chart_maes],
                    textposition="auto"
                ))
                fig_tourn.update_layout(
                    title={"text": "Tournament Arena: Validation MAE (Lower is Better)", "font": {"color": "#FFFFFF", "size": 13}},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis={"title": {"text": "Validation MAE (€ Millions)", "font": {"color": "#94A3B8"}}, "tickfont": {"color": "#94A3B8"}, "showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
                    yaxis={"tickfont": {"color": "#F1F5F9"}, "autorange": "reversed"},
                    height=240,
                    margin={"l": 10, "r": 10, "t": 35, "b": 10}
                )
                st.plotly_chart(fig_tourn, width="stretch")

            with m_col2:
                rmse_val = float(model_metrics.get("test_rmse", 5019762.0))
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px; font-size: 0.88rem; line-height: 1.7; color: #E2E8F0;">
                    <div style="color: #00FF87; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">Champion Model: {champ_name}</div>
                    • <b>Holdout Test MAE:</b> ±EUR {test_mae:,.0f}<br>
                    • <b>Holdout Test RMSE:</b> EUR {rmse_val:,.0f}<br>
                    • <b>Explained Variance (R²):</b> {test_r2 * 100:.2f}%<br>
                    • <b>Validation Split Horizon:</b> 12 Months Rolling<br>
                    • <b>Candidate Pool:</b> {len(tourn_metrics)} Algorithms Evaluated<br>
                    • <b>Model Registry:</b> DagsHub MLflow (pl-market-value-regressor)
                </div>
                """, unsafe_allow_html=True)


        with tab_stack:
            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 24px; padding-bottom: 8px;">
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00f2fe; font-weight: 700; font-size: 0.82rem;">DATA ENGINE</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">DuckDB & Apache Parquet</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Zero-copy vectorized SQL execution and columnar storage. DVC for dataset versioning.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00FF87; font-weight: 700; font-size: 0.82rem;">MACHINE LEARNING</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">LightGBM & Scikit-Learn</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Gradient boosted tree regressors with ColumnTransformer preprocessing pipelines.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00f2fe; font-weight: 700; font-size: 0.82rem;">EXPERIMENT TRACKING</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">MLflow & DagsHub Cloud</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Automated run tracking, artifact storage, and centralized remote Model Registry.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00FF87; font-weight: 700; font-size: 0.82rem;">INFERENCE SERVICE</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">FastAPI & Uvicorn ASGI</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Asynchronous REST API with Pydantic v2 validation and Prometheus instrumentation.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00f2fe; font-weight: 700; font-size: 0.82rem;">DRIFT OBSERVABILITY</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">Evidently AI & Prometheus</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Automated statistical drift detection (Wasserstein & PSI metrics) against baseline.</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                    <div style="color: #00FF87; font-weight: 700; font-size: 0.82rem;">CONTAINER & CI/CD</div>
                    <div style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">Docker & Streamlit UI</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">Multi-stage Docker containerization with automated testing and GitLab/GitHub CI/CD.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.subheader("Player Directory")

    # Filter controls
    fc1, fc2, fc3 = st.columns([2, 1.5, 1])

    player_name_list = catalog["player_name"].tolist()
    all_clubs = ["All Clubs"] + sorted([c for c in catalog["club_name"].dropna().unique().tolist() if c])
    all_positions = ["All Positions"] + sorted([p for p in catalog["position"].dropna().unique().tolist() if p])

    with fc1:
        search_selection = st.selectbox(
            "Search Player",
            options=["All Players"] + player_name_list,
            index=0,
            help="Type player name for instant autocomplete suggestions"
        )

    with fc2:
        selected_club = st.selectbox("Filter Club", options=all_clubs, index=0)

    with fc3:
        selected_pos = st.selectbox("Filter Position", options=all_positions, index=0)

    filtered_catalog = catalog.copy()
    if search_selection != "All Players":
        # Autocomplete selection handler
        matched = catalog[catalog["player_name"] == search_selection]
        if not matched.empty:
            st.session_state.selected_player_id = int(matched.iloc[0]["player_id"])
            st.rerun()

    if selected_club != "All Clubs":
        filtered_catalog = filtered_catalog[filtered_catalog["club_name"] == selected_club]
    if selected_pos != "All Positions":
        filtered_catalog = filtered_catalog[filtered_catalog["position"] == selected_pos]

    filtered_catalog = filtered_catalog.sort_values(by="player_name", ascending=True).reset_index(drop=True)

    st.caption(f"Showing {len(filtered_catalog):,} active Premier League players. Click any row to view profile.")

    # Directory table
    display_df = filtered_catalog[[
        "image_url", "player_name", "club_name", "position", "sub_position", "latest_age", "latest_recorded_val_eur"
    ]].rename(columns={
        "image_url": "Photo",
        "player_name": "Player",
        "club_name": "Club",
        "position": "Position",
        "sub_position": "Role",
        "latest_age": "Age",
        "latest_recorded_val_eur": "Market Value"
    })

    event = st.dataframe(
        display_df,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Photo": st.column_config.ImageColumn("Photo", width="small"),
            "Market Value": st.column_config.NumberColumn("Market Value", format="€%d"),
            "Age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
        hide_index=True,
        width="stretch",
        height=480
    )

    selection = getattr(event, "selection", None)
    if selection is not None:
        selected_rows = selection.get("rows", []) if isinstance(selection, dict) else getattr(selection, "rows", [])
        if selected_rows:
            clicked_idx = selected_rows[0]
            selected_pid = int(filtered_catalog.iloc[clicked_idx]["player_id"])
            st.session_state.selected_player_id = selected_pid
            st.rerun()


# Player Profile View
else:
    player_id = st.session_state.selected_player_id
    features = cached_features(player_id)
    val_df, history_df = cached_history(player_id)

    if not features:
        st.error("Player profile could not be loaded.")
        if st.button("← Back to Directory"):
            st.session_state.selected_player_id = None
            st.rerun()
        st.stop()

    back_col, _ = st.columns([2, 10])
    with back_col:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Directory"):
            st.session_state.selected_player_id = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Player Overview and Performance
    c_prof, c_form, c_val = st.columns([1.1, 1.4, 1.5])

    with c_prof:
        st.subheader("Player Profile")
        img_url = features.get("image_url", "")
        if not img_url or not str(img_url).startswith("http"):
            img_url = "https://via.placeholder.com/200x260?text=No+Photo"
        st.image(img_url, width=190)
        st.markdown(f"<h3 style='margin: 6px 0 2px 0; color: #FFFFFF;'>{features['player_name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #00f2fe; font-weight: 600; margin-bottom: 8px;'>{features['position']} • {features['sub_position']}</div>", unsafe_allow_html=True)
        st.write(f"**Club:** {features['club_name']}")
        st.write(f"**Country:** {features['country_of_citizenship']}")
        st.write(f"**Foot:** {str(features['dominant_foot']).capitalize()}")
        st.write(f"**Age:** {features['age_at_valuation']:.1f} yrs")

    # Active Match Statistics
    with c_form:
        st.subheader("Current Form")
        st.caption("Active season match statistics.")

        goals = int(features.get("window_goals", 0))
        assists = int(features.get("window_assists", 0))
        minutes = int(features.get("window_minutes", 0))
        apps = int(features.get("window_appearances", 0))
        yellows = int(features.get("window_yellow_cards", 0))
        reds = int(features.get("window_red_cards", 0))
        euro = int(features.get("window_european_minutes", 0))
        rate = float(features.get("goal_contributions_per_90", 0.0))

        cf1, cf2 = st.columns(2)
        cf1.metric("Goals", f"{goals}")
        cf2.metric("Assists", f"{assists}")

        cf3, cf4 = st.columns(2)
        cf3.metric("Minutes Played", f"{minutes:,} min")
        cf4.metric("Appearances", f"{apps}" if apps > 0 else "—")

        cf5, cf6 = st.columns(2)
        cf5.metric("G+A / 90 min", f"{rate:.2f}")
        cf6.metric("European Minutes", f"{euro:,} min")

        cf7, cf8 = st.columns(2)
        cf7.metric("Yellow Cards", f"{yellows}")
        cf8.metric("Red Cards", f"{reds}")

    # Valuation Estimates
    with c_val:
        st.subheader("Current Valuation")
        st.caption("AI projected valuation vs Transfermarkt target value.")

        target_val = float(features.get("target_market_value_eur", features.get("current_market_value_eur", 0.0)))
        prior_val = float(features.get("prior_market_value_eur", target_val))
        eval_days = int(features.get("days_between_valuations", 180))

        # Model valuation inference
        pred_eur = None
        try:
            req_payload = {
                "age_at_valuation": float(features["age_at_valuation"]),
                "position": str(features["position"]),
                "sub_position": str(features["sub_position"]),
                "dominant_foot": str(features["dominant_foot"]),
                "club_name": str(features["club_name"]),
                "last_known_value_eur": prior_val,
                "days_between_valuations": eval_days if eval_days > 0 else 180,
                "minutes_since_last_val": minutes,
                "european_minutes_played": euro,
                "goal_contributions_per_90": rate,
                "minutes_prior_window": int(features.get("minutes_prior_window", 1000)),
                "contrib_per_90_prior": float(features.get("contrib_per_90_prior", 0.0)),
                "yellow_cards_since_val": yellows
            }
            res = httpx.post(API_URL, json=req_payload, timeout=4.0)
            if res.status_code == 200:
                pred_eur = res.json()["predicted_market_value_eur"]
        except Exception:
            pred_eur = None

        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("Prior Value (Anchor)", f"EUR {prior_val:,.0f}")
        cv2.metric("Target Value", f"EUR {target_val:,.0f}")
        if pred_eur is not None:
            growth_eur = pred_eur - prior_val
            growth_pct = (growth_eur / prior_val) * 100.0 if prior_val > 0 else 0.0

            cv3.metric(
                "AI Projected Value",
                f"EUR {pred_eur:,.0f}",
                delta=f"{growth_eur:+,.0f} ({growth_pct:+.1f}%) vs Prior",
                delta_color="normal"
            )
        else:
            cv3.metric("AI Projected Value", "Connecting...")

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    # Valuation History and Trajectory
    ch_col, tb_col = st.columns([1.5, 1.2])

    with ch_col:
        st.subheader("Valuation Trajectory")
        st.caption("Historical Transfermarkt valuations over career.")

        if not val_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=val_df["valuation_date"],
                y=val_df["market_value_eur"],
                mode="lines+markers",
                line={"color": "#00f2fe", "width": 3, "shape": "spline"},
                marker={"size": 6, "color": "#00FF87", "line": {"color": "#38003C", "width": 1.5}},
                fill="tozeroy",
                fillcolor="rgba(0, 242, 254, 0.08)",
                name="Market Value",
                hovertemplate="<b>Date:</b> %{x}<br><b>Valuation:</b> EUR %{y:,.0f}<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis={"showgrid": True, "gridcolor": "rgba(255, 255, 255, 0.08)", "tickfont": {"color": "#94A3B8"}},
                yaxis={"showgrid": True, "gridcolor": "rgba(255, 255, 255, 0.08)", "tickfont": {"color": "#94A3B8"}, "title": {"text": "Value (EUR)", "font": {"color": "#94A3B8"}}},
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                height=340
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No recorded historical valuations.")

    with tb_col:
        st.subheader("Value History")
        st.caption("Valuation timeline and club affiliations.")

        if not history_df.empty:
            disp_history = history_df.copy()
            disp_history["valuation_date"] = disp_history["valuation_date"].astype(str)
            disp_history = disp_history.rename(columns={
                "valuation_date": "Date",
                "club_name": "Club",
                "market_value_eur": "Market Value",
                "val_change_eur": "Change",
                "val_change_pct": "Growth"
            })
            st.dataframe(
                disp_history,
                column_config={
                    "Market Value": st.column_config.NumberColumn("Market Value", format="€%d"),
                    "Change": st.column_config.NumberColumn("Change", format="€%d"),
                    "Growth": st.column_config.NumberColumn("Growth", format="%.1f%%"),
                },
                hide_index=True,
                width="stretch",
                height=340
            )
        else:
            st.info("No valuation history records available.")

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    # Scenario Simulator
    st.subheader("Value Simulator")
    st.caption("Project future transfer valuation by adjusting prospective performance metrics.")

    with st.form("simulator_form"):
        s1, s2, s3 = st.columns(3)

        with s1:
            p_age = float(features["age_at_valuation"])
            sim_age = st.slider("Player Age", min_value=16.0, max_value=max(42.0, p_age + 2.0), value=p_age, step=0.25)
            p_min = minutes if minutes > 0 else 1800
            sim_minutes = st.number_input("Minutes Played", min_value=0, max_value=max(6500, p_min + 500), value=p_min, step=50)
            p_euro = euro
            sim_euro = st.number_input("European Minutes", min_value=0, max_value=max(2500, p_euro + 500), value=p_euro, step=50)

        with s2:
            p_val = float(target_val if target_val > 0 else 5_000_000.0)
            sim_val = st.number_input("Last Known Value (€)", min_value=100_000.0, max_value=max(350_000_000.0, p_val * 1.5), value=p_val, step=500_000.0, help="Anchor valuation from which the AI projects scenario changes")
            sim_days = st.slider("Evaluation Window (Days)", min_value=30, max_value=365, value=180, step=15)
            club_list = sorted([c for c in catalog["club_name"].dropna().unique().tolist() if c])
            c_idx = club_list.index(features["club_name"]) if features["club_name"] in club_list else 0
            sim_club = st.selectbox("Transfer Club", options=club_list, index=c_idx)

        with s3:
            p_goals = goals if goals > 0 else 5
            sim_goals = st.number_input("Goals", min_value=0, max_value=max(70, p_goals + 10), value=p_goals, step=1)
            p_assists = assists if assists > 0 else 3
            sim_assists = st.number_input("Assists", min_value=0, max_value=max(45, p_assists + 10), value=p_assists, step=1)
            p_yellows = yellows
            sim_yellows = st.number_input("Yellow Cards", min_value=0, max_value=max(25, p_yellows + 5), value=p_yellows, step=1)

        sim_rate = round(((sim_goals + sim_assists) * 90.0) / sim_minutes, 2) if sim_minutes >= 90 else 0.0

        st.caption(f"Calculated Contribution: **{sim_rate:.2f} G+A / 90 min** ({sim_goals} goals + {sim_assists} assists across {sim_minutes:,} min)")

        sim_btn = st.form_submit_button("Simulate Valuation")

    if sim_btn:
        sim_payload = {
            "age_at_valuation": sim_age,
            "position": str(features["position"]),
            "sub_position": str(features["sub_position"]),
            "dominant_foot": str(features["dominant_foot"]),
            "club_name": str(sim_club),
            "last_known_value_eur": sim_val,
            "days_between_valuations": sim_days,
            "minutes_since_last_val": sim_minutes,
            "european_minutes_played": sim_euro,
            "goal_contributions_per_90": min(sim_rate, 4.0),
            "minutes_prior_window": int(features.get("minutes_prior_window", 1000)),
            "contrib_per_90_prior": float(features.get("contrib_per_90_prior", 0.0)),
            "yellow_cards_since_val": sim_yellows
        }

        try:
            sim_res = httpx.post(API_URL, json=sim_payload, timeout=5.0)
            if sim_res.status_code == 200:
                res_data = sim_res.json()
                sim_pred = res_data["predicted_market_value_eur"]
                sim_diff = sim_pred - sim_val
                sim_pct = (sim_diff / sim_val) * 100.0 if sim_val > 0 else 0.0

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Projected Value", f"EUR {sim_pred:,.0f}")
                rc2.metric("Projected Shift", f"EUR {sim_diff:+,.0f}", delta=f"{sim_pct:+.1f}%")
                rc3.metric("Valuation Interval", f"EUR {max(0, sim_pred - test_mae):,.0f} — {sim_pred + test_mae:,.0f}")
            else:
                st.error(f"Simulation Error ({sim_res.status_code}): {sim_res.text}")
        except httpx.ConnectError:
            st.error("FastAPI service connection error.")