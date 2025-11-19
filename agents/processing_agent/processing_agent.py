# processing.py  (FINAL VERSION — FULL FILE)

import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Load OpenAI Key
# ============================================================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_KEY)

# ============================================================
# File paths
# ============================================================
HARMONIZED_FILE = Path("/home/ai4air/proccesing/harmonized_readings.json")
OUTPUT_FILE = Path("/home/ai4air/proccesing/processing_results.json")


# ============================================================
# Remove ```json fences from model output
# ============================================================
def extract_json_from_markdown(output: str) -> str:
    output = output.strip()

    if output.startswith("```"):
        first_newline = output.find("\n")
        if first_newline != -1:
            output = output[first_newline + 1:]
    if output.endswith("```"):
        output = output[:-3]

    return output.strip()


# ============================================================
# Load harmonized data from file
# ============================================================
def load_harmonized_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Harmonized file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["harmonized_readings"]


# ============================================================
# Group readings by city/interval
# ============================================================
def group_by_city_interval(readings):
    groups = {}

    for r in readings:
        city = r["location"]
        ts = datetime.fromisoformat(r["timestamp"].replace("Z", ""))
        hour = ts.strftime("%H")

        if city not in groups:
            groups[city] = {"00": [], "08": [], "12": [], "16": []}

        if hour in groups[city]:
            groups[city][hour].append(r)

    return groups


# ============================================================
# Build prompt for OpenAI forecast
# ============================================================
def create_prompt(grouped):
    def block(city, interval, rows):
        txt = f"\n### {city} — {interval}\n"
        for r in rows:
            txt += (
                f"- {r['timestamp']}: "
                f"pm2_5={r['pm2_5_ug_per_m3']}, "
                f"pm10={r['pm10_ug_per_m3']}, "
                f"no2={r['no2_ug_per_m3']}, "
                f"o3={r['o3_ug_per_m3']}, "
                f"so2={r['so2_ug_per_m3']}\n"
            )
        return txt

    prompt = "You are an air quality forecasting model.\n"
    prompt += "Below is historical data grouped by city and time interval.\n"

    for city, intervals in grouped.items():
        for interval in ["00", "08", "12", "16"]:
            prompt += f"\n===== {city} @ {interval} =====\n"
            prompt += block(city, interval, intervals[interval])

    prompt += """
Now forecast the next 5 days (4 intervals per day).
Return ONLY JSON in this format:

{
  "forecast_readings": [
    {
      "location": "...",
      "timestamp": "YYYY-MM-DDTHH:00:00Z",
      "pm2_5_ug_per_m3": ...,
      "pm10_ug_per_m3": ...,
      "no2_ug_per_m3": ...,
      "o3_ug_per_m3": ...,
      "so2_ug_per_m3": ...
    }
  ]
}

NO explanation. JSON only.
"""

    return prompt


# ============================================================
# Ask OpenAI for forecast
# ============================================================
def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    print("========== RAW MODEL OUTPUT ==========")
    print(raw)
    print("======================================")

    cleaned = extract_json_from_markdown(raw)
    return json.loads(cleaned)


# ============================================================
# Calculate MAE & RMSE
# ============================================================
def calculate_model_performance(history, forecast):
    import math

    fields = [
        "pm2_5_ug_per_m3",
        "pm10_ug_per_m3",
        "no2_ug_per_m3",
        "o3_ug_per_m3",
        "so2_ug_per_m3",
    ]

    hist_sorted = sorted(history, key=lambda x: x["timestamp"])
    last_actual = hist_sorted[-7:]

    preds = forecast["forecast_readings"][:7]

    n = min(len(last_actual), len(preds))
    if n == 0:
        return {"MAE": None, "RMSE": None, "samples_compared": 0}

    mae_sum, rmse_sum, count = 0, 0, 0

    for i in range(n):
        actual = last_actual[i]
        predicted = preds[i]

        for f in fields:
            if f in actual and f in predicted:
                diff = abs(predicted[f] - actual[f])
                mae_sum += diff
                rmse_sum += diff ** 2
                count += 1

    mae = mae_sum / count
    rmse = (rmse_sum / count) ** 0.5

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "samples_compared": count,
    }


# ============================================================
# Save final output in CORRECT STRUCTURE
# ============================================================
def save_output(data):
    final = {
        "forecast_readings": data.get("forecast_readings", []),
        "model_performance": data.get("model_performance", {})
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(f"Saved forecast run → {OUTPUT_FILE}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_processing_pipeline(harmonized_path: Path):
    print(f"Loading harmonized readings from: {harmonized_path}")

    readings = load_harmonized_data(harmonized_path)
    grouped = group_by_city_interval(readings)
    prompt = create_prompt(grouped)
    forecast = ask_openai(prompt)

    # Add model performance
    performance = calculate_model_performance(readings, forecast)
    forecast["model_performance"] = performance

    save_output(forecast)

    return forecast


# ============================================================
# Wrapper for MAIN AGENT
# ============================================================
def run_processing_agent():
    print("[Processing Agent] Triggered by MAIN AGENT...")

    forecast = run_processing_pipeline(HARMONIZED_FILE)

    return forecast


# ============================================================
# CLI ENTRY
# ============================================================
if __name__ == "__main__":
    print("Running processing agent directly…")
    run_processing_agent()