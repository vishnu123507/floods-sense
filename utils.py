"""
utils.py
FloodSense Pro — Shared Utility Functions
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_pct",
    "River_Level_m",
    "Soil_Moisture_pct",
]
MODEL_PATH  = "flood_model.pkl"
SCALER_PATH = "scaler.pkl"


# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model_and_scaler():
    """
    Loads the pre-trained RandomForest model and StandardScaler.
    If artefacts are missing, auto-trains them first.
    Returns (model, scaler).
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        _auto_train()

    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return model, scaler


def _auto_train():
    """Silently runs the full training pipeline to generate artefacts."""
    from model_training import run
    run()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_flood(
    model,
    scaler,
    rainfall     : float,
    temperature  : float,
    humidity     : float,
    river_level  : float,
    soil_moisture: float,
) -> dict:
    """
    Runs one prediction and returns a rich result dict.
    """
    raw  = pd.DataFrame(
        [[rainfall, temperature, humidity, river_level, soil_moisture]],
        columns=FEATURES,
    )
    scaled = scaler.transform(raw)
    prob   = float(model.predict_proba(scaled)[0][1])
    label  = int(model.predict(scaled)[0])

    risk_level, color, emoji = classify_risk(prob)
    explanation = build_explanation(prob, rainfall, river_level, humidity, soil_moisture)

    return {
        "probability"  : round(prob * 100, 1),
        "label"        : label,
        "risk_level"   : risk_level,
        "color"        : color,
        "emoji"        : emoji,
        "explanation"  : explanation,
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs"       : {
            "rainfall"     : rainfall,
            "temperature"  : temperature,
            "humidity"     : humidity,
            "river_level"  : river_level,
            "soil_moisture": soil_moisture,
        },
    }


# ── Risk classification ───────────────────────────────────────────────────────
def classify_risk(prob: float) -> tuple[str, str, str]:
    """Returns (label, hex_color, emoji) for a given probability 0–1."""
    if prob < 0.35:
        return "Low Risk",    "#22c55e", "🟢"
    elif prob < 0.65:
        return "Medium Risk", "#f59e0b", "🟡"
    else:
        return "High Risk",   "#ef4444", "🔴"


# ── Natural-language explanation ──────────────────────────────────────────────
def build_explanation(prob, rainfall, river_level, humidity, soil_moisture) -> str:
    parts = []

    if rainfall > 150:
        parts.append("extremely heavy rainfall")
    elif rainfall > 80:
        parts.append("significant rainfall")
    elif rainfall > 30:
        parts.append("moderate rainfall")
    else:
        parts.append("low rainfall")

    if river_level > 10:
        parts.append("critically high river levels")
    elif river_level > 6:
        parts.append("elevated river levels")

    if soil_moisture > 75:
        parts.append("saturated soil (low absorption capacity)")
    elif soil_moisture > 50:
        parts.append("moist soil")

    if humidity > 85:
        parts.append("very high atmospheric humidity")

    if prob >= 0.65:
        prefix = "⚠️ Warning: "
        suffix = " are contributing to a HIGH flood probability. Immediate monitoring is advised."
    elif prob >= 0.35:
        prefix = "📋 Note: "
        suffix = " suggest a MODERATE flood risk. Continue monitoring conditions."
    else:
        prefix = "✅ "
        suffix = " indicate LOW flood risk under current conditions."

    combined = ", ".join(parts) if parts else "current environmental conditions"
    combined = combined[0].upper() + combined[1:]
    return prefix + combined + suffix


# ── History helpers ───────────────────────────────────────────────────────────
def add_to_history(session_state, result: dict, max_items: int = 10):
    """Appends a prediction result to session_state.history (capped at max_items)."""
    if "history" not in session_state:
        session_state.history = []
    session_state.history.insert(0, result)
    session_state.history = session_state.history[:max_items]


def history_to_dataframe(history: list) -> Optional[pd.DataFrame]:
    if not history:
        return None
    rows = []
    for h in history:
        rows.append({
            "Time"          : h["timestamp"],
            "Rainfall (mm)" : h["inputs"]["rainfall"],
            "River Lvl (m)" : h["inputs"]["river_level"],
            "Humidity (%)"  : h["inputs"]["humidity"],
            "Probability %" : h["probability"],
            "Risk"          : h["risk_level"],
        })
    return pd.DataFrame(rows)
