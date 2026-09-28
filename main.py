"""
FastAPI Backend for SIH Problem Statement SIH26081.
Smart Model Weighting & Weather Forecast Blending.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

try:
    from backend.forecast import (
        get_available_locations,
        get_available_parameters,
        generate_forecast_payload,
        load_dataset
    )
except ImportError:
    from forecast import (
        get_available_locations,
        get_available_parameters,
        generate_forecast_payload,
        load_dataset
    )

app = FastAPI(
    title="SIH26081 Weather Forecast Blending Engine",
    description="Adaptive weighting and blending system for multi-model weather forecasts",
    version="1.0.0"
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    location: str = "Varanasi"
    parameter: str = "Rainfall"
    use_ml: bool = False


@app.get("/")
def read_root():
    return {
        "project": "SIH26081 Multi-Model Weather Blending",
        "status": "online",
        "demo_mode": True,
        "description": "Smart adaptive weighting prototype combining Model A, Model B, and AI Model."
    }


@app.get("/locations")
def list_locations():
    """Returns supported locations with coordinates for the map."""
    return get_available_locations()


@app.get("/parameters")
def list_parameters():
    """Returns supported weather parameters and their display units."""
    return get_available_parameters()


@app.get("/forecast/{location}")
def get_forecast(
    location: str,
    parameter: str = "Rainfall",
    use_ml: bool = False
):
    """
    Returns model predictions, adaptive weights, final forecast, and explainability.
    """
    try:
        return generate_forecast_payload(location=location, parameter=parameter, use_ml_weights=use_ml)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
def post_forecast(req: ForecastRequest):
    """
    POST endpoint to generate a blended forecast.
    """
    try:
        return generate_forecast_payload(
            location=req.location,
            parameter=req.parameter,
            use_ml_weights=req.use_ml
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical-data")
def get_historical_data(
    location: Optional[str] = None,
    parameter: Optional[str] = None
):
    """Returns historical records for inspection and charting."""
    df = load_dataset()
    if location:
        df = df[df["location"] == location]
    if parameter:
        df = df[df["parameter"] == parameter]
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
