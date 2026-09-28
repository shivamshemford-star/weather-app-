"""
Adaptive Weighting Engine for SIH Problem Statement SIH26081.

Core Concept:
1. Calculate previous error for each weather model: Error = |Predicted - Actual|
2. Convert error into importance: Performance Score = 1 / Error
3. Give higher weight to better model: Weight = Score / Total Scores
4. Blend predictions: Final Forecast = (Model A * Weight A) + (Model B * Weight B) + (AI Model * Weight AI)
"""

import math
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_model_errors(history_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates historical error metrics (MAE, RMSE) for each forecasting model.
    """
    models = ["model_a", "model_b", "ai_model"]
    stats = {}
    
    for m in models:
        errors = (history_df[m] - history_df["actual"]).abs()
        mae = float(errors.mean())
        rmse = float(np.sqrt(((history_df[m] - history_df["actual"]) ** 2).mean()))
        
        # Simple student-friendly accuracy representation
        mean_actual = max(float(history_df["actual"].mean()), 1.0)
        accuracy_pct = max(10.0, min(99.0, 100.0 - (mae / mean_actual * 100.0)))
        
        stats[m] = {
            "mae": round(mae, 1),
            "rmse": round(rmse, 1),
            "accuracy_pct": round(accuracy_pct, 1)
        }
    
    return stats


def compute_adaptive_weights(error_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Computes normalized adaptive weights based on inverse Mean Absolute Error.
    Better past performance (lower error) -> Higher weight.
    Rounds to clean student-friendly multiples of 5% (e.g., 20%, 30%, 50%)
    while strictly ensuring sum = 1.0 (100%).
    """
    scores = {}
    for model, metrics in error_stats.items():
        mae = max(metrics["mae"], 0.1)
        scores[model] = 1.0 / mae
        
    total_score = sum(scores.values())
    raw_weights = {m: scores[m] / total_score for m in scores}
    
    # Round to nearest 0.05 (5%) for ultra-clean presentation
    rounded = {m: round(round(w / 0.05) * 0.05, 2) for m, w in raw_weights.items()}
    diff = round(1.0 - sum(rounded.values()), 2)
    
    if diff != 0:
        # Reconcile on best model
        best_model = max(scores, key=scores.get)
        rounded[best_model] = round(rounded[best_model] + diff, 2)
        
    return rounded


def train_simple_ml_model(history_df: pd.DataFrame) -> Tuple[Dict[str, float], Any]:
    """
    Optional Simple ML Model (Linear Regression) as specified in Section 6.
    Learns: [model_a, model_b, ai_model] -> actual
    """
    X = history_df[["model_a", "model_b", "ai_model"]].values
    y = history_df["actual"].values
    
    reg = LinearRegression(positive=True)  # positive coefficients for intuitive weights
    reg.fit(X, y)
    
    coefs = reg.coef_
    total_coef = sum(coefs)
    if total_coef > 0:
        norm_weights = {
            "model_a": round(float(coefs[0] / total_coef), 2),
            "model_b": round(float(coefs[1] / total_coef), 2),
            "ai_model": round(float(coefs[2] / total_coef), 2)
        }
        # ensure sum = 1.0
        diff = round(1.0 - sum(norm_weights.values()), 2)
        norm_weights["ai_model"] = round(norm_weights["ai_model"] + diff, 2)
    else:
        norm_weights = {"model_a": 0.33, "model_b": 0.33, "ai_model": 0.34}
        
    return norm_weights, reg


def blend_forecast(predictions: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculates final blended forecast:
    Final Forecast = (Model A * Weight A) + (Model B * Weight B) + (AI Model * Weight AI)
    """
    final = (
        predictions["model_a"] * weights["model_a"] +
        predictions["model_b"] * weights["model_b"] +
        predictions["ai_model"] * weights["ai_model"]
    )
    return round(final, 1)


def evaluate_confidence(predictions: Dict[str, float], error_stats: Dict[str, Dict[str, float]], parameter: str) -> str:
    """
    Simple 3-tier confidence system:
    - High: Models agree closely and historical error is low.
    - Medium: Models differ moderately.
    - Low: High disagreement or high historical error.
    """
    vals = [predictions["model_a"], predictions["model_b"], predictions["ai_model"]]
    spread = max(vals) - min(vals)
    avg_mae = np.mean([error_stats[m]["mae"] for m in error_stats])
    
    # Scale relative to parameter magnitude
    if parameter == "Rainfall":
        threshold_low = 8.0
        threshold_med = 18.0
    elif parameter == "Temperature":
        threshold_low = 3.0
        threshold_med = 6.0
    else:  # Wind Speed
        threshold_low = 4.0
        threshold_med = 9.0
        
    if spread <= threshold_low and avg_mae <= threshold_low:
        return "High"
    elif spread <= threshold_med and avg_mae <= threshold_med:
        return "Medium"
    else:
        return "Low"


def generate_explanation(
    predictions: Dict[str, float],
    weights: Dict[str, float],
    error_stats: Dict[str, Dict[str, float]],
    parameter: str,
    unit: str,
    confidence: str
) -> Dict[str, Any]:
    """
    Generates student-friendly 'Why This Forecast?' explanation.
    """
    # Sort models by error (lowest error first)
    sorted_models = sorted(error_stats.items(), key=lambda x: x[1]["mae"])
    best_key, best_stats = sorted_models[0]
    mid_key, mid_stats = sorted_models[1]
    worst_key, worst_stats = sorted_models[2]
    
    labels = {
        "model_a": "Model A (Physics-Based)",
        "model_b": "Model B (Global Model)",
        "ai_model": "AI Model (Machine Learning)"
    }
    
    points = [
        f"**{labels[best_key]}** had the lowest recent error ({best_stats['mae']} {unit}), so the system assigned it the highest weight ({int(weights[best_key]*100)}%).",
        f"**{labels[mid_key]}** performed moderately well with an average error of {mid_stats['mae']} {unit}, receiving a {int(weights[mid_key]*100)}% weight.",
        f"**{labels[worst_key]}** had higher recent deviations ({worst_stats['mae']} {unit}), so its influence was reduced to {int(weights[worst_key]*100)}%.",
        f"The model predictions span from {min(predictions.values())} {unit} to {max(predictions.values())} {unit}. Blending them produces an optimal compromise with **{confidence} confidence**."
    ]
    
    summary = (
        f"Rather than blindly trusting one model, the adaptive weighting algorithm dynamically rewarded "
        f"{labels[best_key]} with {int(weights[best_key]*100)}% influence due to its superior track record, "
        f"yielding a balanced final forecast."
    )
    
    return {
        "summary": summary,
        "key_reasons": points,
        "best_model": labels[best_key],
        "best_model_key": best_key
    }
