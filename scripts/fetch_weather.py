#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

SF = ZoneInfo("America/Los_Angeles")

API_KEY = os.environ.get("VISUAL_CROSSING_API_KEY", "").strip()
LOCATION = os.environ.get("WX_LOCATION", "san francisco").strip()
UNIT = os.environ.get("WX_UNIT", "us").strip().lower()  # 'us' | 'metric'
OUTDIR = os.environ.get("CRIME_DATA_DIR", "crime_prediction_data")

today = datetime.now(SF).date()
tomorrow = today + timedelta(days=1)
week_end = tomorrow + timedelta(days=6)

def rowpick(d: dict) -> dict:
    # Visual Crossing veya Open-Meteo normalize
    return {
        "date": d.get("datetime"),
        "tempmax": d.get("tempmax"),
        "tempmin": d.get("tempmin"),
        "precip": d.get("precip", 0),
        "precipprob": d.get("precipprob"),
        "windspeed": d.get("windspeed"),
        "humidity": d.get("humidity"),
        "description": d.get("description"),
        "icon": d.get("icon"),
        "unit_group": UNIT,
    }

def write_outputs(days: list) -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    # yarin.csv
    t_key = tomorrow.strftime("%Y-%m-%d")
    td = next((d for d in days if d.get("datetime") == t_key), None)
    df_t = pd.DataFrame([rowpick(td)]) if td else pd.DataFrame([{"date": t_key, "unit_group": UNIT}])
    df_t.to_csv(os.path.join(OUTDIR, "yarin.csv"), index=False)
    # week.csv (yarından 7 gün)
    rows = []
    for d in days:
        try:
            dd = datetime.strptime(d.get("datetime", ""), "%Y-%m-%d").date()
        except Exception:
            continue
        if tomorrow <= dd <= week_end:
            rows.append(rowpick(d))
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "week.csv"), index=False)
    return len(rows)

def vc_request_range() -> dict:
    base = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    loc = quote_plus(LOCATION)
    url = f"{base}/{loc}/{tomorrow.isoformat()}/{week_end.isoformat()}?include=days&unitGroup={UNIT}&key={API_KEY}&contentType=json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def vc_request_generic() -> dict:
    base = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    loc = quote_plus(LOCATION)
    url = f"{base}/{loc}?unitGroup={UNIT}&key={API_KEY}&contentType=json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def open_meteo_fallback() -> list:
    # Anahtarsız, günlük veri
    lat, lon = 37.7749, -122.4194
    temp_unit = "fahrenheit" if UNIT == "us" else "celsius"
    wind_unit = "mph" if UNIT == "us" else "kmh"
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,windspeed_10m_max"
        "&forecast_days=8"
        "&timezone=America%2FLos_Angeles"
        f"&temperature_unit={temp_unit}&windspeed_unit={wind_unit}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json().get("daily", {})
    days = []
    for i, dt_str in enumerate(d.get("time", []) or []):
        days.append({
            "datetime": dt_str,
            "tempmax": (d.get("temperature_2m_max") or [None])[i],
            "tempmin": (d.get("temperature_2m_min") or [None])[i],
            "precip": (d.get("precipitation_sum") or [0])[i],
            "precipprob": (d.get("precipitation_probability_max") or [None])[i],  # %
            "windspeed": (d.get("windspeed_10m_max") or [None])[i],               # mph / kmh
            "humidity": None,
            "description": None,
            "icon": None,
        })
    return days

def placeholder_write():
    os.makedirs(OUTDIR, exist_ok=True)
    # Yarın
    pd.DataFrame([{"date": tomorrow.isoformat(), "unit_group": UNIT}]).to_csv(
        os.path.join(OUTDIR, "yarin.csv"), index=False
    )
    # Hafta
    rows = [{"date": (tomorrow + timedelta(days=i)).isoformat(), "unit_group": UNIT} for i in range(7)]
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "week.csv"), index=False)

def main():
    wrote = 0
    try:
        if API_KEY:
            # 1) Visual Crossing explicit range
            data = vc_request_range()
            days = data.get("days", [])
            wrote = write_outputs(days)
            # 2) Gerekirse generic çağrıya düş
            if wrote < 2:
                data = vc_request_generic()
                days = data.get("days", [])
                wrote = write_outputs(days)
        else:
            # Key yok → doğrudan Open-Meteo
            days = open_meteo_fallback()
            wrote = write_outputs(days)
    except Exception as e:
        # Open-Meteo dene; o da olmazsa placeholder
        try:
            print(f"⚠️ VC başarısız: {e} → Open-Meteo fallback deneniyor.")
            days = open_meteo_fallback()
            wrote = write_outputs(days)
        except Exception as e2:
            print(f"⚠️ Open-Meteo da başarısız: {e2} → placeholder yazılıyor.")
            placeholder_write()
            wrote = 0

    # Özet log
    try:
        yarin = pd.read_csv(os.path.join(OUTDIR, "yarin.csv"), nrows=3)
        week = pd.read_csv(os.path.join(OUTDIR, "week.csv"), nrows=3)
        print("✅ yarin.csv ve week.csv yazıldı (week satır sayısı görünüm):", len(pd.read_csv(os.path.join(OUTDIR, "week.csv"))))
        print("---- yarin.csv (head) ----")
        print(yarin.to_string(index=False))
        print("---- week.csv (head) ----")
        print(week.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
