"""
weather_api.py
FloodSense Pro — OpenWeatherMap API Integration

Usage:
    from weather_api import fetch_weather, WeatherData

Free API key: https://openweathermap.org/api  (sign up → Current Weather Data)
"""

import requests
from dataclasses import dataclass, field
from typing import Optional


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class WeatherData:
    city          : str
    country       : str
    temperature   : float          # °C
    humidity      : float          # %
    rainfall_1h   : float          # mm (last 1 hour; 0 if dry)
    description   : str
    icon_code     : str
    wind_speed    : float          # m/s
    feels_like    : float          # °C
    success       : bool  = True
    error_message : str   = ""

    # Computed helpers
    @property
    def icon_url(self) -> str:
        return f"https://openweathermap.org/img/wn/{self.icon_code}@2x.png"

    @property
    def rainfall_display(self) -> str:
        return f"{self.rainfall_1h:.1f} mm/h"


@dataclass
class WeatherError:
    success       : bool  = False
    error_message : str   = ""
    city          : str   = ""
    country       : str   = ""
    temperature   : float = 0.0
    humidity      : float = 0.0
    rainfall_1h   : float = 0.0
    description   : str   = ""
    icon_code     : str   = "01d"
    wind_speed    : float = 0.0
    feels_like    : float = 0.0

    @property
    def icon_url(self) -> str:
        return f"https://openweathermap.org/img/wn/{self.icon_code}@2x.png"


# ── Main fetch function ───────────────────────────────────────────────────────
def fetch_weather(city: str, api_key: str) -> WeatherData | WeatherError:
    """
    Fetches current weather for *city* using the OpenWeatherMap
    Current Weather Data API (free tier).

    Returns WeatherData on success, WeatherError on failure.
    """
    if not api_key or api_key.strip() == "" or api_key == "YOUR_API_KEY_HERE":
        return WeatherError(error_message="⚠️ No API key provided. Enter your key in the sidebar.")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q"     : city.strip(),
        "appid" : api_key.strip(),
        "units" : "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
    except requests.exceptions.ConnectionError:
        return WeatherError(error_message="🌐 Network error. Check your internet connection.")
    except requests.exceptions.Timeout:
        return WeatherError(error_message="⏱️ Request timed out. Try again.")

    if resp.status_code == 401:
        return WeatherError(error_message="🔑 Invalid API key. Check your OpenWeatherMap key.")
    if resp.status_code == 404:
        return WeatherError(error_message=f"🏙️ City '{city}' not found. Try a different name.")
    if resp.status_code == 429:
        return WeatherError(error_message="🚦 API rate limit reached. Wait a moment and retry.")
    if resp.status_code != 200:
        return WeatherError(error_message=f"❌ API error {resp.status_code}: {resp.text[:120]}")

    try:
        d = resp.json()
        rainfall = 0.0
        if "rain" in d:
            rainfall = d["rain"].get("1h", d["rain"].get("3h", 0.0))

        return WeatherData(
            city        = d["name"],
            country     = d["sys"]["country"],
            temperature = round(d["main"]["temp"],       1),
            humidity    = round(d["main"]["humidity"],   1),
            rainfall_1h = round(rainfall,                2),
            description = d["weather"][0]["description"].title(),
            icon_code   = d["weather"][0]["icon"],
            wind_speed  = round(d["wind"]["speed"],      1),
            feels_like  = round(d["main"]["feels_like"], 1),
        )
    except (KeyError, ValueError) as exc:
        return WeatherError(error_message=f"⚠️ Could not parse API response: {exc}")


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    key  = sys.argv[1] if len(sys.argv) > 1 else "YOUR_API_KEY_HERE"
    city = sys.argv[2] if len(sys.argv) > 2 else "London"
    w = fetch_weather(city, key)
    if w.success:
        print(f"{w.city}, {w.country}  |  {w.temperature}°C  |  "
              f"Humidity {w.humidity}%  |  Rain {w.rainfall_1h} mm/h  |  {w.description}")
    else:
        print("Error:", w.error_message)
