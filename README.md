# 🌊 FloodSense Pro — Real-Time AI Flood Prediction System

A production-grade machine-learning web application that combines a trained **Random Forest** model with live **OpenWeatherMap** data to predict flood risk in real time.

---

## 📁 Project Structure

```
floodsense_pro/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── model_training.py       # Dataset generation + ML training pipeline
├── weather_api.py          # OpenWeatherMap API integration
├── utils.py                # Shared helpers: prediction, risk logic, history
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── data/                   # Auto-generated on first run
    ├── flood_dataset.csv
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## ⚙️ Setup Instructions

### 1 — Clone / download the project

```bash
# If using git:
git clone <your-repo-url>
cd floodsense_pro
```

### 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Pre-train the model

```bash
python model_training.py
```


> The app **auto-trains on first launch** if the model is missing. This step is optional.

### 5 — Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 OpenWeatherMap API Setup

1. Visit **https://openweathermap.org/api** and create a free account
2. Navigate to **API keys** in your account dashboard
3. Copy your API key
4. In the app **sidebar**, paste the key into the **API Key** field
5. Enter a city name (e.g. `Mumbai`, `London`, `Bangkok`)
6. Click **☁️ Fetch Live Weather** — rainfall, temperature, and humidity auto-fill

> Free tier supports 60 calls/minute and 1 million calls/month — more than enough.

---

## 🎯 Key Features

| Feature | Details |
|---|---|
| **Prediction** | Triggered ONLY by the "Estimate Flood Risk" button — sliders do NOT auto-update |
| **Dark / Light Mode** | Toggle in sidebar instantly switches theme |
| **Live Weather** | Fetches real rainfall, temp, humidity; auto-fills sliders |
| **Gauge Chart** | Plotly animated gauge showing probability 0–100% |
| **Risk Levels** | 🟢 Low (<35%) · 🟡 Medium (35–65%) · 🔴 High (>65%) |
| **Explanation** | Plain-English interpretation of the prediction |
| **History** | Last 10 predictions stored in session with rainfall vs probability chart |
| **Data Insights** | Box plots, scatter plots, feature importance visualisations |
| **Error Handling** | Friendly messages for invalid API key, city not found, network errors |

---

## 🌊 Risk Level Guide

| Probability | Level | Action |
|---|---|---|
| 0 – 34% | 🟢 Low Risk | Normal monitoring |
| 35 – 64% | 🟡 Medium Risk | Increased vigilance |
| 65 – 100% | 🔴 High Risk | Immediate action advised |

---

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 300 estimators
- **Training data**: 3 000 synthetic samples
- **Features**: Rainfall, Temperature, Humidity, River Water Level, Soil Moisture
- **Train/Test split**: 80% / 20% stratified
- **Target**: Binary flood risk (0 = No Flood, 1 = Flood)
- **Expected accuracy**: ~87%

---

*Built as a college AI/ML project. Dataset is synthetic; replace `generate_dataset()` in `model_training.py` with real sensor data for production use.*
