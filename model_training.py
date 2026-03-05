"""
model_training.py
FloodSense Pro — ML Training Pipeline
Generates synthetic dataset, trains Random Forest, saves artefacts.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_pct",
    "River_Level_m",
    "Soil_Moisture_pct",
]
DATA_PATH  = os.path.join("data", "flood_dataset.csv")
MODEL_PATH = "flood_model.pkl"
SCALER_PATH = "scaler.pkl"


# ── 1. Dataset Generation ─────────────────────────────────────────────────────
def generate_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic environmental dataset.
    Flood label is derived from a weighted combination of features
    plus Gaussian noise to mimic real-world uncertainty.
    """
    rng = np.random.default_rng(seed)

    rainfall      = rng.uniform(0,   300, n)
    temperature   = rng.uniform(10,   45, n)
    humidity      = rng.uniform(20,  100, n)
    river_level   = rng.uniform(0,    15, n)
    soil_moisture = rng.uniform(0,   100, n)

    # Weighted flood score
    score = (
        0.35 * (rainfall      / 300) +
        0.30 * (river_level   /  15) +
        0.20 * (soil_moisture / 100) +
        0.10 * (humidity      / 100) +
        0.05 * (temperature   /  45)
    ) + rng.normal(0, 0.045, n)

    flood_risk = (score > 0.52).astype(int)

    df = pd.DataFrame({
        "Rainfall_mm"       : rainfall.round(2),
        "Temperature_C"     : temperature.round(2),
        "Humidity_pct"      : humidity.round(2),
        "River_Level_m"     : river_level.round(2),
        "Soil_Moisture_pct" : soil_moisture.round(2),
        "Flood_Risk"        : flood_risk,
    })

    # Inject ~3 % missing values for realism
    for col in ["Rainfall_mm", "Humidity_pct", "Soil_Moisture_pct"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print(f"  Raw shape      : {df.shape}")
    print(f"  Missing values :\n{df.isnull().sum().to_string()}\n")

    # Fill missing with median
    df = df.fillna(df.median(numeric_only=True))

    X = df[FEATURES].copy()
    y = df["Flood_Risk"].copy()

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"  Flood prevalence : {y.mean()*100:.1f} %\n")
    return X_train, X_test, y_train, y_test, scaler, df


# ── 3. Training ───────────────────────────────────────────────────────────────
def train(X_train, y_train) -> RandomForestClassifier:
    print("  Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.\n")
    return clf


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : accuracy_score (y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, zero_division=0),
        "recall"    : recall_score   (y_test, y_pred, zero_division=0),
        "f1"        : f1_score       (y_test, y_pred, zero_division=0),
        "cm"        : confusion_matrix(y_test, y_pred),
    }

    print("  ── Evaluation Metrics ──")
    for k, v in metrics.items():
        if k != "cm":
            print(f"  {k.capitalize():12s}: {v*100:.2f} %")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Flood','Flood'])}")
    return metrics


# ── 5. Plots ──────────────────────────────────────────────────────────────────
def save_plots(clf, metrics: dict):
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Flood","Flood"],
                yticklabels=["No Flood","Flood"], ax=ax)
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join("data", "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1a73e8","#34a853","#fbbc04","#ea4335","#9334ea"]
    ax.bar(range(len(FEATURES)), imp[idx],
           color=[colors[i] for i in range(len(FEATURES))],
           edgecolor="white")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in idx], rotation=25, ha="right")
    ax.set_title("Feature Importances", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join("data", "feature_importance.png"), dpi=150)
    plt.close()
    print("  Plots saved to data/")


# ── 6. Save artefacts ─────────────────────────────────────────────────────────
def save_artefacts(clf, scaler, df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(clf,    f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"  Model  → {MODEL_PATH}")
    print(f"  Scaler → {SCALER_PATH}")
    print(f"  Data   → {DATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
def run():
    print("=" * 50)
    print("  FloodSense Pro — Model Training Pipeline")
    print("=" * 50 + "\n")

    # Create output directory first (works on Windows + Linux + macOS)
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()
    X_train, X_test, y_train, y_test, scaler, df_clean = preprocess(df)
    clf     = train(X_train, y_train)
    metrics = evaluate(clf, X_test, y_test)
    save_plots(clf, metrics)
    save_artefacts(clf, scaler, df_clean)

    print("\n✅  All done. Run:  streamlit run app.py")
    return metrics


if __name__ == "__main__":
    run()
