from pydantic import BaseModel, Field


# Single player raw input features
class PlayerFeatures(BaseModel):
    age_at_valuation: float = Field(..., ge=15.0, le=45.0, description="Age at valuation", json_schema_extra={"example": 24.5})
    position: str = Field(..., description="Playing position", json_schema_extra={"example": "Attack"})
    sub_position: str = Field(..., description="Specific sub position", json_schema_extra={"example": "Centre-Forward"})
    dominant_foot: str = Field(..., description="Dominant foot", json_schema_extra={"example": "right"})
    club_name: str = Field(..., description="Club name", json_schema_extra={"example": "Arsenal FC"})
    last_known_value_eur: float = Field(..., gt=0, description="Prior market valuation", json_schema_extra={"example": 45000000.0})
    days_between_valuations: int = Field(..., ge=1, le=1000, description="Elapsed evaluation days", json_schema_extra={"example": 180})
    minutes_since_last_val: int = Field(..., ge=0, description="Minutes in window", json_schema_extra={"example": 1850})
    european_minutes_played: int = Field(default=0, ge=0, description="European minutes", json_schema_extra={"example": 450})
    goal_contributions_per_90: float = Field(default=0.0, ge=0.0, description="Goal contributions rate", json_schema_extra={"example": 0.80})
    minutes_prior_window: int = Field(default=0, ge=0, description="Prior window minutes", json_schema_extra={"example": 1700})
    contrib_per_90_prior: float = Field(default=0.0, ge=0.0, description="Prior contributions rate", json_schema_extra={"example": 0.60})
    yellow_cards_since_val: int = Field(default=0, ge=0, description="Yellow cards", json_schema_extra={"example": 2})

# Prediction output response
class PredictionResponse(BaseModel):
    predicted_market_value_eur: float = Field(..., description="Estimated market valuation")
    log_market_value: float = Field(..., description="Log raw output")
    valuation_currency: str = Field(default="EUR")
    model_version: str = Field(default="LightGBM-v6")

# Single player hydrated response
class PlayerPredictionResponse(BaseModel):
    player_id: int
    player_name: str
    club_name: str
    last_known_value_eur: float
    predicted_market_value_eur: float
    log_market_value: float
    valuation_currency: str = Field(default="EUR")
    model_version: str = Field(default="LightGBM-v6")

# Scenario simulator request
class ScenarioSimulationRequest(BaseModel):
    player_id: int = Field(..., description="Target player identifier", json_schema_extra={"example": 50202})
    simulated_minutes: int = Field(..., ge=0, le=4500, description="Projected minutes played", json_schema_extra={"example": 2500})
    simulated_goals: int = Field(default=0, ge=0, description="Projected goals", json_schema_extra={"example": 15})
    simulated_assists: int = Field(default=0, ge=0, description="Projected assists", json_schema_extra={"example": 8})
    simulated_yellow_cards: int = Field(default=0, ge=0, description="Projected yellow cards", json_schema_extra={"example": 3})
    simulated_european_minutes: int = Field(default=0, ge=0, description="Projected European minutes", json_schema_extra={"example": 450})
    simulated_club: str | None = Field(default=None, description="Optional prospective club", json_schema_extra={"example": "Arsenal FC"})
    days_elapsed: int = Field(default=180, ge=30, le=365, description="Days to next valuation", json_schema_extra={"example": 180})

# Scenario simulator response
class ScenarioSimulationResponse(BaseModel):
    player_id: int
    player_name: str
    current_market_value_eur: float
    predicted_market_value_eur: float
    value_change_eur: float
    value_change_pct: float
    log_market_value: float
    valuation_currency: str = Field(default="EUR")

# Batch prediction request
class BatchPredictionRequest(BaseModel):
    players: list[PlayerFeatures]

# Batch prediction response
class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

# Health check response
class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    service: str