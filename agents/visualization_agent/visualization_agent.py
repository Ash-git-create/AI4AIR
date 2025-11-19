# visualization.py — FINAL VERSION (Berlin + Heidelberg fully supported)

import json
from pathlib import Path
from datetime import timedelta
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROCESSING_FILE = Path("/home/ai4air/proccesing/processing_results.json")
HIST_FILE = Path("/home/ai4air/proccesing/harmonized_readings.json")
OUTPUT_FILE = Path("/home/ai4air/proccesing/visualization_output.json")

# -------------------------------------------------------------------
# AQI Classification
# -------------------------------------------------------------------
GOOD_LIMITS = {
    "pm2_5_ug_per_m3": 12.0,
    "pm10_ug_per_m3": 54.0,
    "no2_ug_per_m3": 100.0,
    "o3_ug_per_m3": 100.0,
    "so2_ug_per_m3": 40.0,
}

ICONS = {
    "Good": "🟢",
    "Moderate": "🟡",
    "Unhealthy": "🔴",
}


def classify_level(pollutant: str, value):
    """Return (level, icon) for pollutant value."""
    if value is None:
        return "Good", ICONS["Good"]

    v = float(value)
    limit = GOOD_LIMITS.get(pollutant, 50.0)

    if v <= limit:
        return "Good", ICONS["Good"]
    elif v <= limit * 2:
        return "Moderate", ICONS["Moderate"]
    else:
        return "Unhealthy", ICONS["Unhealthy"]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------
# PICK LATEST VALID PROCESSING RUN
# -------------------------------------------------------------------
def get_latest_processing_run(city: str) -> dict:
    """
    Must return processing run that contains:
      - forecast_readings for this city
      - model_performance
    """
    data = load_json(PROCESSING_FILE)

    def matches_city(run):
        return (
            isinstance(run, dict)
            and "forecast_readings" in run
            and any(r.get("location") == city for r in run["forecast_readings"])
        )

    # CASE 1 — dict file
    if isinstance(data, dict):
        if matches_city(data):
            return data
        # nested list under "runs"
        if "runs" in data and isinstance(data["runs"], list):
            for r in reversed(data["runs"]):
                if matches_city(r):
                    return r

    # CASE 2 — list file
    if isinstance(data, list):
        for r in reversed(data):
            if matches_city(r):
                return r

    # No matching city found
    raise ValueError(
        f"processing_results.json does not contain forecast data for city '{city}'."
    )


# -------------------------------------------------------------------
# MAIN CALCULATION PER CITY
# -------------------------------------------------------------------
def compute_city_all(city: str) -> dict:
    print(f"[Visualization Agent] Computing pollutant outputs for {city}")

    # Load historical harmonized
    hist_data = load_json(HIST_FILE)
    hist_df = pd.DataFrame(hist_data["harmonized_readings"])
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

    # Filter historical to city
    hist_df = hist_df[hist_df["location"] == city].copy()

    # Load processing forecast
    proc = get_latest_processing_run(city)
    forecast_list = proc["forecast_readings"]
    model_perf = proc.get("model_performance", {})

    forecast_df = pd.DataFrame(forecast_list)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

    # Filter forecast to city
    forecast_df = forecast_df[forecast_df["location"] == city].copy()

    pollutants = [
        "pm2_5_ug_per_m3",
        "pm10_ug_per_m3",
        "no2_ug_per_m3",
        "o3_ug_per_m3",
        "so2_ug_per_m3",
    ]

    output = {
        "city": city,
        "model_performance": model_perf,
        "data": {},
    }

    for pol in pollutants:
        output["data"][pol] = compute_single_pollutant(city, pol, hist_df, forecast_df)

    return output


def compute_single_pollutant(city, pollutant, hist_df, forecast_df):
    """Return dict { today, next_5_days, trend } for pollutant."""

    # TODAY value
    if not forecast_df.empty:
        mid = forecast_df.iloc[len(forecast_df) // 2]
        today_val = mid[pollutant]
    elif not hist_df.empty:
        today_val = hist_df.iloc[-1][pollutant]
    else:
        today_val = None

    level, icon = classify_level(pollutant, today_val)

    # NEXT 5 DAYS
    next_5 = []
    if not forecast_df.empty:
        daily = (
            forecast_df.assign(date=forecast_df["timestamp"].dt.date)
            .groupby("date")[pollutant]
            .mean()
            .reset_index()
            .head(5)
        )

        for i, row in daily.iterrows():
            val = float(row[pollutant])
            lvl, ico = classify_level(pollutant, val)
            next_5.append(
                {
                    "date": row["date"].isoformat(),
                    "label": "Today" if i == 0 else row["date"].strftime("%a"),
                    "value": round(val, 2),
                    "level": lvl,
                    "icon": ico,
                }
            )

    # TREND: historical last 4 days + all forecast
    trend = []

    if not hist_df.empty:
        hist_window = hist_df[hist_df["timestamp"] >= hist_df["timestamp"].max() - timedelta(days=4)]
        for _, r in hist_window.iterrows():
            trend.append(
                {
                    "timestamp": r["timestamp"].isoformat(),
                    pollutant: float(r[pollutant]),
                    "type": "historical",
                }
            )

    if not forecast_df.empty:
        for _, r in forecast_df.iterrows():
            trend.append(
                {
                    "timestamp": r["timestamp"].isoformat(),
                    pollutant: float(r[pollutant]),
                    "type": "forecast",
                }
            )

    trend = sorted(trend, key=lambda x: x["timestamp"])

    return {
        "today": {"value": today_val, "level": level, "icon": icon},
        "next_5_days": next_5,
        "trend": trend,
    }


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------
def run_visualization_agent(city: str) -> dict:
    print(f"[Visualization Agent] Running for {city}")

    result = compute_city_all(city)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[Visualization Agent] Saved → {OUTPUT_FILE}")
    return result

# -------------------------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    # Allow dynamic city selection from command line
    if len(sys.argv) > 1:
        city = sys.argv[1]
    else:
        city = "Berlin"  # default fallback

    print(f"[Visualization Agent] CLI invoked → city={city}")

    out = run_visualization_agent(city)
    print(json.dumps(out, indent=2))