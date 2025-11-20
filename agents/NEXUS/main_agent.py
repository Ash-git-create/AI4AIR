# backend.py

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ===========================================================
# PATH FIX — USE THE REAL FOLDER NAME: proccesing
# ===========================================================

BASE_DIR = Path("/home/ai4air")
PROCESSING_DIR = BASE_DIR / "proccesing"   # <-- EXACT spelling from your server

# Make Python see the folder
sys.path.append(str(BASE_DIR))
sys.path.append(str(PROCESSING_DIR))


# ===========================================================
# IMPORTS — MATCH THE REAL FOLDER NAME
# ===========================================================

from proccesing.processing_agent import run_processing_agent
from proccesing.visualization import run_visualization_agent


# JSON files
PROC_RESULTS = PROCESSING_DIR / "processing_results.json"
VIS_OUTPUT  = PROCESSING_DIR / "visualization_output.json"
HISTORICAL  = PROCESSING_DIR / "harmonized_readings.json"


# ===========================================================
# FASTAPI APP
# ===========================================================

app = FastAPI(title="AI4AIR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================
# ROOT
# ===========================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "AI4AIR Backend is running",
        "routes": [
            "/run_pipeline/{city}",
            "/get_processing_data",
            "/get_visualization_data",
        ],
    }


# ===========================================================
# RUN PIPELINE
# ===========================================================

@app.get("/run_pipeline/{city}")
def run_pipeline(city: str) -> Dict[str, Any]:

    print("\n==============================================")
    print(f" 🚀 RUNNING FULL PIPELINE FOR: {city}")
    print("==============================================\n")

    # STEP 1 — Processing
    print("🔹 Step 1 — Running Processing Agent…")
    processing_output = run_processing_agent()
    print("✅ Processing Agent Finished.\n")

    # STEP 2 — Visualization
    print("🔹 Step 2 — Running Visualization Agent…")
    visualization_output = run_visualization_agent(city)
    print("✅ Visualization Agent Finished.\n")

    return {
        "status": "ok",
        "message": f"Pipeline completed for {city}",
        "city": city,
        "processing": {
            "output_path": str(PROC_RESULTS),
            "meta": processing_output
        },
        "visualization": {
            "output_path": str(VIS_OUTPUT),
            "data": visualization_output
        }
    }


# ===========================================================
# SAFE JSON UTIL
# ===========================================================

def safe_read_json(path: Path) -> Any:
    if not path.exists():
        return {"error": f"{path.name} not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"failed to read {path.name}: {e}"}


# ===========================================================
# GET PROCESSING DATA
# ===========================================================

@app.get("/get_processing_data")
def get_processing_data():
    return safe_read_json(PROC_RESULTS)


# ===========================================================
# GET VIS DATA
# ===========================================================

@app.get("/get_visualization_data")
def get_visualization_data():
    return safe_read_json(VIS_OUTPUT)


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    print("\n==============================================")
    print(" 🌐 AI4AIR Backend API Server Starting...")
    print("==============================================\n")
    uvicorn.run(app, host="0.0.0.0", port=5001)
