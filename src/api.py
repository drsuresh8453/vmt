"""
api.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

FastAPI REST endpoint for system integration.
Insurance systems, banks, fleet companies can call this API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict, get_model_info

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Annual Mileage Prediction API",
    description="""
    Predicts annual vehicle mileage for insurance premium pricing.
    Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

    ## Endpoints
    - **POST /predict** — Predict annual km for a vehicle
    - **GET /health**   — Check API and model health
    - **GET /model_info** — Current model version and metrics
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ─────────────────────────────────────────────────────────
class VehicleRecord(BaseModel):
    # Vehicle features
    vehicle_brand      : str   = Field(default="Maruti",        description="Vehicle brand name")
    vehicle_model      : str   = Field(default="Swift",         description="Vehicle model name")
    vehicle_segment    : str   = Field(default="Hatchback",     description="Hatchback/Sedan/SUV/MUV")
    brand_tier         : str   = Field(default="Budget",        description="Budget/Mid/Premium")
    manufacture_year   : int   = Field(default=2019,            description="Year of manufacture")
    engine_cc          : int   = Field(default=1200,            description="Engine displacement in cc")
    fuel_type          : str   = Field(default="Petrol",        description="Petrol/Diesel/CNG/Electric")
    transmission       : str   = Field(default="Manual",        description="Manual/Automatic")
    num_owners         : int   = Field(default=1,               description="Number of previous owners")
    vehicle_condition  : str   = Field(default="Good",          description="Poor/Average/Good/Excellent")
    is_premium_brand   : int   = Field(default=0,               description="1 if BMW/Audi/Mercedes")
    seating_capacity   : int   = Field(default=5,               description="Number of seats")
    color              : str   = Field(default="White",         description="Vehicle colour")
    selling_price      : float = Field(default=600000,          description="Current market value in INR")
    insurance_type     : str   = Field(default="Comprehensive", description="Third Party/Comprehensive/Zero Dep")
    num_accidents      : int   = Field(default=0,               description="Number of accidents on record")

    # Owner features
    occupation         : str   = Field(default="Salaried",      description="Salaried/Business/Student/Retired")
    owner_age          : int   = Field(default=32,              description="Age of primary driver")
    annual_income_lakh : float = Field(default=8.5,             description="Annual income in lakhs")
    gender             : str   = Field(default="Male",          description="Male/Female/Other")
    num_drivers        : int   = Field(default=1,               description="Number of people who drive")
    has_children       : int   = Field(default=1,               description="1 if has children")
    num_children       : int   = Field(default=1,               description="Number of children")
    driving_exp_years  : int   = Field(default=8,               description="Years of driving experience")
    education          : str   = Field(default="Graduate",      description="School/Graduate/PG")
    marital_status     : str   = Field(default="Married",       description="Single/Married")
    household_size     : int   = Field(default=3,               description="Number in household")
    num_vehicles_owned : int   = Field(default=1,               description="Total vehicles owned")

    # Location features
    city_tier          : str   = Field(default="Metro",         description="Metro/Tier2/Tier3")
    state              : str   = Field(default="Karnataka",     description="State of registration")
    home_to_office_km  : float = Field(default=18,              description="One-way commute distance in km")
    road_quality       : str   = Field(default="Good",          description="Poor/Average/Good")
    traffic_index      : str   = Field(default="High",          description="Low/Medium/High/Very High")
    has_metro_rail     : int   = Field(default=1,               description="1 if city has metro")
    region_type        : str   = Field(default="Urban",         description="Rural/Semi-urban/Urban")
    highway_access     : int   = Field(default=0,               description="1 if near highway")
    parking_type       : str   = Field(default="Society",       description="Garage/Society/Roadside")

    # Usage features
    weekend_trips      : str   = Field(default="Monthly",       description="Never/Monthly/Weekly")
    uses_for_business  : int   = Field(default=0,               description="1 if used for business")
    uses_for_tourism   : int   = Field(default=0,               description="1 if used for tourism")
    is_rideshare       : int   = Field(default=0,               description="1 if Ola/Uber driver")
    night_driving      : int   = Field(default=0,               description="1 if regular night driving")
    work_from_home     : str   = Field(default="Sometimes",     description="Never/Sometimes/Always")
    monthly_fuel_spend : float = Field(default=3500,            description="Monthly fuel spend in INR")
    toll_spend_monthly : float = Field(default=200,             description="Monthly toll spend in INR")
    daily_trips        : int   = Field(default=4,               description="Average trips per day")

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_brand": "Maruti", "vehicle_model": "Swift",
                "fuel_type": "Petrol", "occupation": "Salaried",
                "city_tier": "Metro", "home_to_office_km": 18,
                "is_rideshare": 0, "work_from_home": "Sometimes"
            }
        }


# ── Response Schema ────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_kms       : int
    predicted_kms_range : dict
    risk_category       : str
    premium_multiplier  : float
    estimated_premium   : int
    recommended_product : str
    shap_explanation    : dict
    model_info          : dict


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check API health and model availability."""
    try:
        info = get_model_info()
        return {
            "status"      : "healthy",
            "model_loaded": True,
            "model_name"  : info.get("model", "Unknown"),
            "model_mape"  : info.get("mape", "N/A"),
            "model_trained_on": info.get("trained_on", "N/A"),
            "api_version" : "1.0.0"
        }
    except Exception as e:
        return {
            "status"      : "degraded",
            "model_loaded": False,
            "error"       : str(e)
        }


@app.get("/model_info")
def model_info():
    """Get current model version and performance metrics."""
    try:
        info = get_model_info()
        return {
            "model_name"    : info.get("model", "Unknown"),
            "version"       : info.get("version", "v1"),
            "mape"          : info.get("mape", "N/A"),
            "mae"           : info.get("mae", "N/A"),
            "rmse"          : info.get("rmse", "N/A"),
            "r2"            : info.get("r2", "N/A"),
            "adj_r2"        : info.get("adj_r2", "N/A"),
            "cv_mae"        : info.get("cv_mae", "N/A"),
            "trained_on"    : info.get("trained_on", "N/A"),
            "n_train"       : info.get("n_train", "N/A"),
            "n_features"    : info.get("n_features", "N/A"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_mileage(record: VehicleRecord):
    """
    Predict annual vehicle mileage and insurance recommendation.

    Returns predicted km, risk category, premium estimate and SHAP explanation.
    """
    try:
        result = predict(record.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(records: list[VehicleRecord]):
    """Predict mileage for multiple vehicles at once."""
    try:
        results = []
        for record in records:
            result = predict(record.dict())
            results.append(result)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
