# AI4AIR – AI Agent System for Air Quality Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Multi--Agent--System-lightgrey)

**AI4AIR** collects, processes and visualises air quality data for a chosen region, and uses an LLM to forecast pollutant levels from recent observations.

> **Team project**, built by three contributors for a university module. Commits were made from shared machines, so git authorship does not track who wrote what.

## Project Overview

AI4AIR is composed of four agents:

| Agent                      | Role                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| Main-Agent / NEXUS         | Coordinates all other agents, manages task scheduling, and facilitates communication between agents. |
| Homogen-Agent / HOMOGEN    | Ingests, cleans, aligns, and harmonizes air quality and weather data from various sources.           |
| Processing-Agent / AIRCAST | Applies machine learning models to predict future air quality metrics (e.g., PM2.5, AQI).            |
| Visualization-Agent / DASH | Visualizes current and predicted data via interactive dashboards.                                    |

## Key Objectives

- Harmonize air quality data from heterogeneous sources
- Predict future pollutant levels (PM2.5, NO₂, AQI) using machine learning
- Provide accessible and interactive visual dashboards for public and institutional use
- Support modular agent-based development for future scalability

## Tech Stack

- Language: Python 3.10+
- Forecasting: OpenAI API (gpt-4o-mini), prompted over recent observations
- Visualization: Streamlit, Plotly
- Data ingestion: Requests, Pandas, Sensor.Community API, Copernicus CAMS via `cdsapi`
- Storage: MySQL via SQLAlchemy
- Service layer: FastAPI

Statistical and deep-learning forecasters (scikit-learn, XGBoost, LSTM) were scoped but not implemented. Forecasting is LLM-prompted.

## Project Structure

```
AI4AIR/
│
├── agents/
│   ├── NEXUS/
│   │   └── main_agent.py
│   ├── HOMOGEN/
│   │   └── harmonizer.py
│   ├── AIRCAST/
│   │   └── processing_agent.py
│   ├── DASH/
│   │   └── visualization_agent.py
│   └── requirements.txt
│
├── dashboard/
│   └── app.py
│   └── requirements.txt
├── data/
├── README.md
└── .env.example
          
```

## Example Workflow

0. NEXUS initializes the pipeline, schedules agent tasks, and manages communication between HOMOGEN, PROCESSOR, and VISIOS.
1. HOMOGEN connects to the Sensor.Community and Copernicus CAMS APIs, fetches recent air quality and weather data, and stores harmonised output in MySQL.
2. AIRCAST loads this harmonized data, applies trained machine learning models, and forecasts future pollutant levels and AQI.
3. DASH reads both current and predicted data, and displays it via a modern dashboard for user interaction.

## Sample Use Case

- City: Berlin
- Goal: Predict PM2.5 levels for the next 24 hours
- Outcome: Real-time dashboard shows air quality trends, alerts high-risk periods, and helps citizens plan activities.

## Known limitations

- The forecast is produced by prompting gpt-4o-mini. There is no trained model and no accuracy evaluation.
- `dashboard/app.py` reads a hardcoded backend host. Set `BACKEND_URL` before running it elsewhere.
- Throughput has not been measured.

## License

MIT. See [`LICENSE`](LICENSE).

## Contributing

Contributions, bug reports, and feature suggestions are welcome.
Please open an issue or submit a pull request.

## Contact

For questions, feedback, or collaboration requests:

- GitHub: [Poojashrees3](https://github.com/Poojashrees3), [PrajwalUnaik](https://github.com/PrajwalUnaik), [Ash-git-create](https://github.com/Ash-git-create)
