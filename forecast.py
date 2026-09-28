"""
Forecast Data Service for SIH Problem Statement SIH26081.
Manages location metadata, built-in datasets, and forecast retrieval.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
try:
    from backend.weighting import (
        calculate_model_errors,
        compute_adaptive_weights,
        train_simple_ml_model,
        blend_forecast,
        evaluate_confidence,
        generate_explanation
    )
except ImportError:
    from weighting import (
        calculate_model_errors,
        compute_adaptive_weights,
        train_simple_ml_model,
        blend_forecast,
        evaluate_confidence,
        generate_explanation
    )

CSV_PATH = os.path.join(os.path.dirname(__file__), "weather_data.csv")

LOCATIONS = [
    {
        "id": "Varanasi",
        "name": "Varanasi",
        "state": "Uttar Pradesh",
        "lat": 25.3176,
        "lon": 82.9739,
        "description": "Historic city along the Ganges with subtropical humid climate"
    },
    {
        "id": "Delhi",
        "name": "Delhi",
        "state": "National Capital Territory",
        "lat": 28.6139,
        "lon": 77.2090,
        "description": "Northern inland plains with seasonal monsoon variations"
    },
    {
        "id": "Mumbai",
        "name": "Mumbai",
        "state": "Maharashtra",
        "lat": 19.0760,
        "lon": 72.8777,
        "description": "Western coastal metropolis with heavy maritime rainfall patterns"
    },
    {
        "id": "Lucknow",
        "name": "Lucknow",
        "state": "Uttar Pradesh",
        "lat": 26.8467,
        "lon": 80.9462,
        "description": "Central Indo-Gangetic plains with warm monsoonal climate"
    }
]

PARAMETERS = {
    "Rainfall": {"unit": "mm", "icon": "CloudRain"},
    "Temperature": {"unit": "°C", "icon": "Thermometer"},
    "Wind Speed": {"unit": "km/h", "icon": "Wind"}
}

# Raw Tomorrow Predictions matching user specification (Section 4 & Section 8)
TOMORROW_PREDICTIONS = {
    "Varanasi": {
        "Rainfall": {"model_a": 30.0, "model_b": 38.0, "ai_model": 34.0},
        "Temperature": {"model_a": 37.0, "model_b": 32.0, "ai_model": 34.5},
        "Wind Speed": {"model_a": 19.0, "model_b": 13.0, "ai_model": 16.0}
    },
    "Delhi": {
        "Rainfall": {"model_a": 20.0, "model_b": 35.0, "ai_model": 28.0},
        "Temperature": {"model_a": 39.0, "model_b": 34.0, "ai_model": 36.5},
        "Wind Speed": {"model_a": 25.0, "model_b": 16.0, "ai_model": 20.0}
    },
    "Mumbai": {
        "Rainfall": {"model_a": 42.0, "model_b": 35.0, "ai_model": 39.0},
        "Temperature": {"model_a": 32.0, "model_b": 28.5, "ai_model": 30.0},
        "Wind Speed": {"model_a": 28.0, "model_b": 19.5, "ai_model": 24.0}
    },
    "Lucknow": {
        "Rainfall": {"model_a": 18.0, "model_b": 25.0, "ai_model": 22.0},
        "Temperature": {"model_a": 36.0, "model_b": 31.0, "ai_model": 33.5},
        "Wind Speed": {"model_a": 17.5, "model_b": 11.5, "ai_model": 14.5}
    }
}


def load_dataset() -> pd.DataFrame:
    """Loads historical CSV dataset."""
    return pd.read_csv(CSV_PATH)


def get_available_locations() -> List[Dict[str, Any]]:
    """Returns list of supported demo locations."""
    return LOCATIONS


def get_available_parameters() -> Dict[str, Any]:
    """Returns list of supported weather parameters and units."""
    return PARAMETERS


def generate_forecast_payload(
    location: str = "Varanasi",
    parameter: str = "Rainfall",
    use_ml_weights: bool = False
) -> Dict[str, Any]:
    """
    Generates a full forecast payload including raw predictions,
    adaptive weights, final forecast, past performance, and explanations.
    """
    if location not in TOMORROW_PREDICTIONS:
        location = "Varanasi"
    if parameter not in PARAMETERS:
        parameter = "Rainfall"
        
    df = load_dataset()
    filtered = df[(df["location"] == location) & (df["parameter"] == parameter)]
    
    # 1. Historical error stats
    error_stats = calculate_model_errors(filtered)
    
    # 2. Weights calculation
    if use_ml_weights:
        weights, _ = train_simple_ml_model(filtered)
        weight_method = "Simple ML (Linear Regression)"
    else:
        weights = compute_adaptive_weights(error_stats)
        weight_method = "Adaptive Error Weighting (Inverse Error)"
        
    # 3. Model Predictions for Tomorrow
    predictions = TOMORROW_PREDICTIONS[location][parameter]
    
    # 4. Final Blended Forecast
    final_forecast = blend_forecast(predictions, weights)
    
    # 5. Confidence
    confidence = evaluate_confidence(predictions, error_stats, parameter)
    
    unit = PARAMETERS[parameter]["unit"]
    
    # 6. Explainability
    explanation = generate_explanation(
        predictions=predictions,
        weights=weights,
        error_stats=error_stats,
        parameter=parameter,
        unit=unit,
        confidence=confidence
    )
    
    # 7. Recent historical data for charts & tables
    history_records = filtered.tail(5).to_dict(orient="records")
    
    return {
        "location": location,
        "parameter": parameter,
        "unit": unit,
        "forecast_date": "Tomorrow",
        "predictions": predictions,
        "weights": weights,
        "weights_percent": {k: int(round(v * 100)) for k, v in weights.items()},
        "final_forecast": final_forecast,
        "confidence": confidence,
        "weight_method": weight_method,
        "error_stats": error_stats,
        "explanation": explanation,
        "history": history_records,
        "formula": {
            "equation": f"Final Forecast = ({predictions['model_a']} × {weights['model_a']}) + ({predictions['model_b']} × {weights['model_b']}) + ({predictions['ai_model']} × {weights['ai_model']})",
            "evaluated": f"= {round(predictions['model_a']*weights['model_a'], 2)} + {round(predictions['model_b']*weights['model_b'], 2)} + {round(predictions['ai_model']*weights['ai_model'], 2)} = {final_forecast} {unit}"
        }
    }
