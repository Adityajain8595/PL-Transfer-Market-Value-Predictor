import json
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.impute import SimpleImputer

from src.serve.dashboard_utils import get_player_features
from src.serve.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    PlayerFeatures,
    PlayerPredictionResponse,
    PredictionResponse,
    ScenarioSimulationRequest,
    ScenarioSimulationResponse,
)
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "champion_model.joblib"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"
model_pipeline = None

# Ensure pickled models from scikit-learn 1.7.x execute seamlessly on newer sklearn releases
_orig_simple_imputer_transform = SimpleImputer.transform


def _compat_simple_imputer_transform(self, X):
    if not hasattr(self, "_fill_dtype"):
        self._fill_dtype = self.statistics_.dtype if hasattr(self, "statistics_") else None
    return _orig_simple_imputer_transform(self, X)


SimpleImputer.transform = _compat_simple_imputer_transform


def load_champion_version() -> str:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                c_metrics = json.load(f)
                return str(c_metrics.get("model_version", f"{c_metrics.get('model_name', 'LightGBM')}-v6"))
        except (OSError, json.JSONDecodeError) as err:
            logger.warning(f"Could not parse champion metrics file: {err}")
    return "LightGBM-v6"


model_version_str = load_champion_version()


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
        logger.info(f"Loading champion model from {MODEL_PATH}...")
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model pipeline successfully loaded ({model_version_str}).")
    except Exception as err:
        logger.error("Failed loading model.")
        raise CustomException(err, sys) from err
    yield
    logger.info("Shutting down inference service...")

app = FastAPI(
    title="PL Market Value API",
    description="Real-time player market value inference & scenario simulation service.",
    version="1.0.0",
    lifespan=lifespan
)

# Prometheus instrumentator setup
Instrumentator().instrument(app).expose(app)

# Convert payload to dataframe
def payload_to_df(payload: PlayerFeatures) -> pd.DataFrame:
    data = payload.model_dump()
    data["log_last_known_value"] = float(np.round(np.log1p(data["last_known_value_eur"]), 4))
    del data["last_known_value_eur"]
    return pd.DataFrame([data])

# Health check probe
@app.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        model_loaded=(model_pipeline is not None),
        service="pl-market-value-inference"
    )

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_player(features: PlayerFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable.")
    try:
        df = payload_to_df(features)
        pred_log = float(model_pipeline.predict(df)[0])
        pred_eur = float(np.clip(np.expm1(pred_log), a_min=0, a_max=None))

        return PredictionResponse(
            predicted_market_value_eur=round(pred_eur, 2),
            log_market_value=round(pred_log, 4),
            model_version=model_version_str
        )
    except Exception as err:
        logger.error(f"Prediction error: {err!s}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {err!s}") from err

# Hydrated player prediction endpoint
@app.get("/predict/player/{player_id}", response_model=PlayerPredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_by_player_id(player_id: int):
    if model_pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable.")
    try:
        player_data = get_player_features(player_id)
        if not player_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Player ID {player_id} not found.")

        req = PlayerFeatures(
            age_at_valuation=float(player_data["age_at_valuation"]),
            position=str(player_data["position"]),
            sub_position=str(player_data["sub_position"]),
            dominant_foot=str(player_data["dominant_foot"]),
            club_name=str(player_data["club_name"]),
            last_known_value_eur=float(player_data["target_market_value_eur"]),
            days_between_valuations=int(player_data.get("days_between_valuations", 180)),
            minutes_since_last_val=int(player_data.get("minutes_since_last_val", 1000)),
            european_minutes_played=int(player_data.get("european_minutes_played", 0)),
            goal_contributions_per_90=float(player_data.get("goal_contributions_per_90", 0.0)),
            minutes_prior_window=int(player_data.get("minutes_prior_window", 1000)),
            contrib_per_90_prior=float(player_data.get("contrib_per_90_prior", 0.0)),
            yellow_cards_since_val=int(player_data.get("yellow_cards_since_val", 0))
        )
        df = payload_to_df(req)
        pred_log = float(model_pipeline.predict(df)[0])
        pred_eur = float(np.clip(np.expm1(pred_log), a_min=0, a_max=None))

        return PlayerPredictionResponse(
            player_id=player_id,
            player_name=str(player_data.get("player_name", "Unknown")),
            club_name=str(player_data.get("club_name", "Unknown")),
            last_known_value_eur=float(player_data["target_market_value_eur"]),
            predicted_market_value_eur=round(pred_eur, 2),
            log_market_value=round(pred_log, 4),
            model_version=model_version_str
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Player lookup inference error: {err!s}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {err!s}") from err

# Scenario simulator endpoint
@app.post("/predict/simulate", response_model=ScenarioSimulationResponse, status_code=status.HTTP_200_OK, tags=["Simulation"])
async def simulate_scenario(sim: ScenarioSimulationRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable.")
    try:
        player_data = get_player_features(sim.player_id)
        if not player_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Player ID {sim.player_id} not found.")

        current_val = float(player_data["target_market_value_eur"])
        target_club = sim.simulated_club or str(player_data["club_name"])
        goal_contrib = min(round(((sim.simulated_goals + sim.simulated_assists) * 90.0) / sim.simulated_minutes, 4), 4.0) if sim.simulated_minutes >= 90 else 0.0

        req = PlayerFeatures(
            age_at_valuation=float(player_data["age_at_valuation"]) + (sim.days_elapsed / 365.25),
            position=str(player_data["position"]),
            sub_position=str(player_data["sub_position"]),
            dominant_foot=str(player_data["dominant_foot"]),
            club_name=target_club,
            last_known_value_eur=current_val,
            days_between_valuations=sim.days_elapsed,
            minutes_since_last_val=sim.simulated_minutes,
            european_minutes_played=sim.simulated_european_minutes,
            goal_contributions_per_90=goal_contrib,
            minutes_prior_window=int(player_data.get("minutes_since_last_val", 1000)),
            contrib_per_90_prior=float(player_data.get("goal_contributions_per_90", 0.0)),
            yellow_cards_since_val=sim.simulated_yellow_cards
        )
        df = payload_to_df(req)
        pred_log = float(model_pipeline.predict(df)[0])
        pred_eur = float(np.clip(np.expm1(pred_log), a_min=0, a_max=None))

        val_change = pred_eur - current_val
        val_pct = (val_change / current_val) * 100.0 if current_val > 0 else 0.0

        return ScenarioSimulationResponse(
            player_id=sim.player_id,
            player_name=str(player_data.get("player_name", "Unknown")),
            current_market_value_eur=round(current_val, 2),
            predicted_market_value_eur=round(pred_eur, 2),
            value_change_eur=round(val_change, 2),
            value_change_pct=round(val_pct, 2),
            log_market_value=round(pred_log, 4)
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Scenario simulation error: {err!s}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Simulation error: {err!s}") from err

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_batch(batch: BatchPredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable.")
    try:
        dfs = [payload_to_df(player) for player in batch.players]
        batch_df = pd.concat(dfs, ignore_index=True)

        preds_log = model_pipeline.predict(batch_df)
        preds_eur = np.clip(np.expm1(preds_log), a_min=0, a_max=None)

        results = [
            PredictionResponse(
                predicted_market_value_eur=round(float(eur), 2),
                log_market_value=round(float(log_val), 4)
            )
            for eur, log_val in zip(preds_eur, preds_log)
        ]
        return BatchPredictionResponse(predictions=results)
    except Exception as err:
        logger.error(f"Batch prediction error: {err!s}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch error: {err!s}") from err