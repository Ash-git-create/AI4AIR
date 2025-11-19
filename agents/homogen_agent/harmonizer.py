# -*- coding: utf-8 -*-
"""
Air Quality Data Harmonizer Agent - V6.0 (Multi-Sensor Aggregation)

- Added time splits (00:00, 06:00, 12:00, 18:00)
- Implemented multi-sensor aggregation with ±2 hour windows
- Uses DB data instead of API for harmonization
- Added backfill support
- Processes only yesterday's date
"""

import requests
import cdsapi
import xarray as xr
import pandas as pd
import json
import os
import time
import zipfile
import random
import openai
import mysql.connector
import logging
import numpy as np
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional

# SQLAlchemy imports for DB
from sqlalchemy import create_engine, Column, String, Date, JSON, DateTime, Integer, Float, UniqueConstraint, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert 
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.mysql import insert as mysql_insert

# --- Load Environment Variables ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set the key in your .env file.")

# ======================================================================
# === DATABASE CONFIGURATION WITH SSL/TLS ===
# ======================================================================
DB_DRIVER = "mysql+pymysql"
DB_USER = "******"
DB_PASSWORD = "******"  # Your password
DB_HOST = "******"
DB_PORT = 3306
DB_NAME = "air_quality_db"
# ======================================================================

# URL.create() automatically handles password encoding
DATABASE_URL = URL.create(
    drivername="mysql+mysqlconnector",
    username=DB_USER,
    password=DB_PASSWORD,  # This gets automatically encoded
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

engine = create_engine(
    DATABASE_URL,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={
        'connect_timeout': 30,
        'charset': 'utf8mb4',
        'auth_plugin': 'mysql_native_password'
    }
)

session_factory = sessionmaker(bind=engine)
ScopedSession = scoped_session(session_factory)
Base = declarative_base()

# --- Setup Logging ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
log_filename = LOG_DIR / f"harmonizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger("HarmonizerAgent")

# --- Database ORM Models ---

class HarmonizationStatus(Base):
    __tablename__ = "harmonization_status"
    city = Column(String(100), primary_key=True)
    last_updated = Column(Date, nullable=False)

class RawSensorData(Base):
    __tablename__ = "raw_sensor_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(100), index=True)
    fetch_timestamp = Column(DateTime, default=datetime.utcnow)
    target_date = Column(Date, index=True)
    raw_data = Column(JSON)

class RawCamsData(Base):
    __tablename__ = "raw_cams_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fetch_timestamp = Column(DateTime, default=datetime.utcnow)
    target_date = Column(Date, index=True)
    raw_data = Column(JSON)

class HarmonizedReading(Base):
    __tablename__ = "harmonized_readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    location = Column(String(100), index=True)
    timestamp = Column(DateTime, index=True)
    
    pollutant_pm25 = Column(Float, nullable=True)
    pollutant_pm10 = Column(Float, nullable=True)
    pollutant_no2 = Column(Float, nullable=True)
    pollutant_o3 = Column(Float, nullable=True)
    pollutant_so2 = Column(Float, nullable=True)
    
    other_data = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint('location', 'timestamp', name='_location_timestamp_uc'),
    )


# --- Database Management Class ---

class DatabaseManager:
    
    def __init__(self, session_scoped):
        self.Session = session_scoped

    def create_tables(self):
        logger.info("Connecting to database and creating tables (if not present)...")
        logger.info(f"Database: {DB_NAME} on {DB_HOST}:{DB_PORT}")
        try:
            # Test connection first
            with engine.connect() as conn:
                logger.info("✅ Database connection successful!")
            
            Base.metadata.create_all(engine)
            logger.info("✅ Tables are ready.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to connect or create tables: {e}")
            logger.error("Please check:")
            logger.error("1. Your Cloud SQL instance is RUNNING")
            logger.error("2. DB_HOST is the correct PRIVATE IP (not public IP)")
            logger.error("3. DB_PORT is 3306 (default MySQL port)")
            logger.error("4. The VM and Cloud SQL are in same VPC/region")
            logger.error("5. Cloud SQL has authorized the VM's network")
            raise

    def get_last_update_date(self, city: str) -> Optional[date]:
        session = self.Session()
        try:
            status = session.query(HarmonizationStatus).filter_by(city=city).first()
            result = status.last_updated if status else None
            logger.info(f"Retrieved last update date for {city}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting last update date for {city}: {e}")
            return None
        finally:
            self.Session.remove()

    def set_last_update_date(self, city: str, update_date: date):
        session = self.Session()
        try:
            # Use MySQL-specific upsert
            stmt = mysql_insert(HarmonizationStatus).values(
                city=city, last_updated=update_date
            )
            upsert_stmt = stmt.on_duplicate_key_update(
                last_updated=stmt.inserted.last_updated
            )
            session.execute(upsert_stmt)
            session.commit()
            
            logger.info(f"DB state updated for {city}: {update_date.isoformat()}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error setting last update date for {city}: {e}")
        finally:
            self.Session.remove()

    def save_raw_sensor_data(self, city: str, target_date: date, data: list):
        session = self.Session()
        try:
            record = RawSensorData(city=city, target_date=target_date, raw_data=data)
            session.add(record)
            session.commit()
            logger.info(f"Saved raw Sensor.Community data to DB for {city} {target_date} - {len(data)} records")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving raw sensor data: {e}")
        finally:
            self.Session.remove()

    def save_raw_cams_data(self, target_date: date, df: pd.DataFrame):
        session = self.Session()
        try:
            data_json = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')
            record = RawCamsData(target_date=target_date, raw_data=data_json)
            session.add(record)
            session.commit()
            logger.info(f"Saved raw CAMS data to DB for {target_date} - {len(data_json)} records")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving raw CAMS data: {e}")
        finally:
            self.Session.remove()

    def append_harmonized_data(self, readings: List[Dict[str, Any]]):
        session = self.Session()
        records_to_insert = []
        for r in readings:
            try:
                ts_str = r.get("timestamp")
                ts_obj = datetime.fromisoformat(ts_str) if ts_str else None
                
                if not r.get("location") or not ts_obj:
                    logger.warning(f"Skipping record with missing location/timestamp: {r}")
                    continue
                
                pollutants = r.get("pollutants", {})
                
                records_to_insert.append({
                    "location": r.get("location"),
                    "timestamp": ts_obj,
                    "pollutant_pm25": pollutants.get("PM2.5"),
                    "pollutant_pm10": pollutants.get("PM10"),
                    "pollutant_no2": pollutants.get("NO2"),
                    "pollutant_o3": pollutants.get("O3"),
                    "pollutant_so2": pollutants.get("SO2"),
                    "other_data": r
                })
            except Exception as e:
                logger.error(f"Error processing record {r}: {e}")
        
        if not records_to_insert:
            logger.warning("No valid new records to append.")
            self.Session.remove()
            return
            
        try:
            # Use MySQL-specific upsert
            stmt = mysql_insert(HarmonizedReading).values(records_to_insert)
            
            # Use "on_duplicate_key_update" with a no-op to skip duplicates
            upsert_stmt = stmt.on_duplicate_key_update(
                id=HarmonizedReading.id  # No-op, just keeps the existing ID
            )
            result = session.execute(upsert_stmt)
            session.commit()
            logger.info(f"Successfully appended/skipped {len(records_to_insert)} harmonized reading(s) to DB. Rows affected: {result.rowcount}")
        except IntegrityError:
            session.rollback()
            logger.warning("Integrity error (likely duplicates), rolling back.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error appending harmonized data: {e}")
        finally:
            self.Session.remove()

    def get_harmonized_readings(self, city: str, limit: int = 100) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            readings = session.query(HarmonizedReading).filter(
                HarmonizedReading.location.ilike(f"%{city}%")
            ).order_by(
                HarmonizedReading.timestamp.desc()
            ).limit(limit).all()
            
            # Convert ORM objects to dictionaries
            result = [
                {
                    "location": r.location,
                    "timestamp": r.timestamp.isoformat(),
                    "pollutant_pm25": r.pollutant_pm25,
                    "pollutant_pm10": r.pollutant_pm10,
                    "pollutant_no2": r.pollutant_no2,
                    "pollutant_o3": r.pollutant_o3,
                    "pollutant_so2": r.pollutant_so2,
                    "other_data": r.other_data
                } for r in readings
            ]
            logger.info(f"Retrieved {len(result)} harmonized readings for {city}")
            return result
        except Exception as e:
            logger.error(f"Error getting harmonized readings for {city}: {e}")
            return []
        finally:
            self.Session.remove()

    def get_sensor_data_for_time_window(self, city: str, target_datetime: datetime) -> List[Dict[str, Any]]:
        """Get all sensor data within ±2 hours of target datetime - FIXED VERSION"""
        session = self.Session()
        try:
            window_start = target_datetime - timedelta(hours=3)
            window_end = target_datetime + timedelta(hours=3)
            
            # Query ALL raw sensor data for the city (not filtered by date)
            records = session.query(RawSensorData).filter(
                RawSensorData.city == city
            ).all()
            
            # Extract and filter records within the time window
            result = []
            for record in records:
                for sensor_record in record.raw_data:
                    try:
                        # Handle timestamp format variations
                        ts = sensor_record.get("timestamp")
                        if not ts:
                            continue
                        
                        # Normalize timestamp format
                        ts = ts.replace('Z', '+00:00')
                        if ' ' in ts:  # Handle "YYYY-MM-DD HH:MM:SS" format
                            ts = ts.replace(' ', 'T')
                        
                        record_timestamp = datetime.fromisoformat(ts)
                        
                        # Check if within time window (regardless of date)
                        if window_start <= record_timestamp <= window_end:
                            result.append(sensor_record)
                    except (ValueError, AttributeError, KeyError) as e:
                        logger.debug(f"Skipping record due to timestamp parsing error: {e}")
                        continue
            
            logger.info(f"Found {len(result)} sensor records for {city} in window {window_start} to {window_end}")
            return result
        except Exception as e:
            logger.error(f"Error getting sensor data for time window: {e}")
            return []
        finally:
            self.Session.remove()

    def get_cams_data_for_time(self, target_date: date, target_time: str) -> Dict[str, Any]:
        """Get CAMS data for specific date and time (00, 06, 12, 18)"""
        session = self.Session()
        try:
            # Query raw CAMS data for the target date
            records = session.query(RawCamsData).filter(
                RawCamsData.target_date == target_date
            ).all()
            
            # Find the record matching the target time
            for record in records:
                for cams_record in record.raw_data:
                    try:
                        # CAMS time format might vary, try to match the time
                        time_str = str(cams_record.get('time', ''))
                        if target_time in time_str:
                            return cams_record
                    except (ValueError, AttributeError):
                        continue
            
            logger.warning(f"No CAMS data found for {target_date} at time {target_time}")
            return {}
        except Exception as e:
            logger.error(f"Error getting CAMS data for time: {e}")
            return {}
        finally:
            self.Session.remove()

    def get_earliest_sensor_date(self) -> Optional[date]:
        """Get the earliest date with sensor data for backfill"""
        session = self.Session()
        try:
            result = session.query(RawSensorData.target_date).order_by(
                RawSensorData.target_date.asc()
            ).first()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting earliest sensor date: {e}")
            return None
        finally:
            self.Session.remove()

    def prune_raw_data(self, before_date: date):
        """Delete raw sensor and CAMS data older than cutoff date"""
        session = self.Session()
        try:
            # Delete raw sensor data
            sensor_deleted = session.query(RawSensorData).filter(
                RawSensorData.target_date < before_date
            ).delete()
            
            # Delete raw CAMS data
            cams_deleted = session.query(RawCamsData).filter(
                RawCamsData.target_date < before_date
            ).delete()
            
            session.commit()
            logger.info(f"Pruned raw data: {sensor_deleted} sensor records and {cams_deleted} CAMS records older than {before_date}")
            return {"sensor_records_deleted": sensor_deleted, "cams_records_deleted": cams_deleted}
        except Exception as e:
            session.rollback()
            logger.error(f"Error pruning raw data: {e}")
            return {"error": str(e)}
        finally:
            self.Session.remove()


# === Agent Class for LangChain ===

class HarmoniserAgent:
    
    CITY_URLS = {
        "Heidelberg": "https://data.sensor.community/airrohr/v1/filter/area=49.3988,8.6724,5",
        "Berlin": "https://data.sensor.community/airrohr/v1/filter/area=52.5200,13.4050,10",
    }
    
    # Define CAMS area for cities. Could be more dynamic.
    CAMS_AREAS = {
        "Heidelberg": [53, 8, 49, 14], # N, W, S, E
        "Berlin": [53, 13, 52, 14]
    }
    
    # Time splits for harmonization
    TIME_SPLITS = ["00:00", "06:00", "12:00", "18:00"]
    
    def __init__(self, db_manager: DatabaseManager, openai_api_key: str):
        self.db_manager = db_manager
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.cds_client = cdsapi.Client()

    def _fetch_sensor_community_data(self, city: str, target_date: date) -> list:
        """Fetches and saves raw Sensor.Community data."""
        url = self.CITY_URLS.get(city)
        if not url:
            logger.warning(f"Unknown city '{city}'. Skipping.")
            return []
        try:
            logger.info(f"Fetching Sensor.Community data for {city} from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Save raw data to DB
            self.db_manager.save_raw_sensor_data(city, target_date, data)

            target_date_str = target_date.strftime('%Y-%m-%d')
            #filtered = [r for r in data if r.get("timestamp", "").startswith(target_date_str)]

            filtered = []
            for r in data:
                ts = r.get("timestamp")
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts_dt.date() == target_date:
                        filtered.append(r)
                except:
                    continue   
            
            logger.info(f"Fetched {len(data)} total records, {len(filtered)} match target date {target_date_str}")
            
            # Log sample of the data
            if filtered:
                sample_data = filtered[:3]  # Log first 3 records as sample
                logger.info(f"Sample Sensor.Community data for {city}: {json.dumps(sample_data, indent=2)}")
            
            return filtered
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Sensor.Community data for {city}: {e}")
            return []

    def _fetch_and_extract_cams_data(self, city: str, target_date: date) -> pd.DataFrame:
        """Fetches, saves, and extracts CAMS data - USING WORKING V5.7 LOGIC"""
        location_area = self.CAMS_AREAS.get(city)
        if not location_area:
            logger.warning(f"No CAMS area defined for {city}. Skipping CAMS.")
            return pd.DataFrame()
            
        logger.info(f"Requesting CAMS data for {city} on {target_date.isoformat()}")
        request_date_str = target_date.strftime('%Y-%m-%d')
        
        # Create a temporary file path for the download
        zip_file_path = None
        try:
            zip_file_path = f"/tmp/cams_data_{city}_{target_date.isoformat()}.zip"
            logger.info(f"Starting CAMS API request for {city} - area: {location_area}")
            
            # --- USING WORKING V5.7 CAMS API CALL ---
            # 1. Define the request payload
            request_payload = {
                "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um", "sulphur_dioxide"],
                "model": "ensemble", "level": "0", "date": request_date_str,
                "type": "analysis", "time": ["00:00", "06:00", "12:00", "18:00"],
                "leadtime_hour": "0", "format": "netcdf_zip", "area": location_area,
            }
            
            # 2. Queue the request (2-argument method, like in v4.1)
            result = self.cds_client.retrieve(
                "cams-europe-air-quality-forecasts", 
                request_payload
            )
            
            # 3. Download the queued result to the target file
            result.download(zip_file_path)
            # --- END WORKING V5.7 CAMS API CALL ---
            
            logger.info(f"CAMS data downloaded to {zip_file_path}")

            with zipfile.ZipFile(zip_file_path, 'r') as z:
                nc_filename = next((n for n in z.namelist() if n.endswith('.nc')), None)
                if not nc_filename:
                    raise FileNotFoundError("No .nc file found in downloaded zip.")
                
                logger.info(f"Extracting NetCDF file: {nc_filename}")
                with z.open(nc_filename) as nc_file:
                    try:
                        with xr.open_dataset(nc_file, decode_timedelta=False) as ds:
                            df = ds.to_dataframe().reset_index()
                        logger.info(f"CAMS data extracted successfully. Shape: {df.shape}")
                    except Exception as xr_e:
                        logger.error(f"Warning: xarray failed to open {nc_filename}. {xr_e}. Returning empty DataFrame.")
                        df = pd.DataFrame()

            # Save raw data (as JSON) to DB
            if not df.empty:
                self.db_manager.save_raw_cams_data(target_date, df)
                # Log sample of CAMS data
                sample_df = df.head(3)
                logger.info(f"Sample CAMS data for {city}: {sample_df.to_dict(orient='records')}")
            else:
                logger.warning("CAMS data extraction resulted in empty DataFrame")
            
            os.remove(zip_file_path)
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch or process CAMS data for {city}: {e}")
            # Clean up zip file if it exists and failed mid-process
            if zip_file_path and os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            return pd.DataFrame()

    def _aggregate_sensor_data(self, sensor_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple sensor records using robust statistics"""
        if not sensor_records:
            return {
                "pollutants": {},
                "statistics": {"sensor_count": 0},
                "raw_samples": []
            }
        
        # Extract pollutant values
        pollutants = {
            "PM2.5": [],
            "PM10": [],
            "NO2": [],
            "O3": [],
            "SO2": []
        }
        
        value_mapping = {
            "P1": "PM10",
            "P2": "PM2.5", 
            "no2": "NO2",
            "o3": "O3",
            "so2": "SO2"
        }
        
        for record in sensor_records:
            sensordatavalues = record.get("sensordatavalues", [])
            for value_data in sensordatavalues:
                value_type = value_data.get("value_type")
                if value_type in value_mapping:
                    pollutant_name = value_mapping[value_type]
                    try:
                        value = float(value_data.get("value", 0))
                        if value > 0:  # Filter out zero/negative values
                            pollutants[pollutant_name].append(value)
                    except (ValueError, TypeError):
                        continue
        
        # Apply robust aggregation
        aggregated_pollutants = {}
        statistics = {}
        
        # PM2.5 → median
        if pollutants["PM2.5"]:
            aggregated_pollutants["PM2.5"] = np.median(pollutants["PM2.5"])
            statistics["PM2.5"] = {
                "min": np.min(pollutants["PM2.5"]),
                "max": np.max(pollutants["PM2.5"]),
                "std": np.std(pollutants["PM2.5"]),
                "count": len(pollutants["PM2.5"])
            }
        
        # PM10 → median  
        if pollutants["PM10"]:
            aggregated_pollutants["PM10"] = np.median(pollutants["PM10"])
            statistics["PM10"] = {
                "min": np.min(pollutants["PM10"]),
                "max": np.max(pollutants["PM10"]),
                "std": np.std(pollutants["PM10"]),
                "count": len(pollutants["PM10"])
            }
        
        # NO2 → mean
        if pollutants["NO2"]:
            aggregated_pollutants["NO2"] = np.mean(pollutants["NO2"])
            statistics["NO2"] = {
                "min": np.min(pollutants["NO2"]),
                "max": np.max(pollutants["NO2"]),
                "std": np.std(pollutants["NO2"]),
                "count": len(pollutants["NO2"])
            }
        
        # O3 → mean
        if pollutants["O3"]:
            aggregated_pollutants["O3"] = np.mean(pollutants["O3"])
            statistics["O3"] = {
                "min": np.min(pollutants["O3"]),
                "max": np.max(pollutants["O3"]),
                "std": np.std(pollutants["O3"]),
                "count": len(pollutants["O3"])
            }
        
        # SO2 → mean
        if pollutants["SO2"]:
            aggregated_pollutants["SO2"] = np.mean(pollutants["SO2"])
            statistics["SO2"] = {
                "min": np.min(pollutants["SO2"]),
                "max": np.max(pollutants["SO2"]),
                "std": np.std(pollutants["SO2"]),
                "count": len(pollutants["SO2"])
            }
        
        statistics["sensor_count"] = len(sensor_records)
        
        # Sample raw records for context (max 20)
        raw_samples = sensor_records[:20]
        
        return {
            "pollutants": aggregated_pollutants,
            "statistics": statistics,
            "raw_samples": raw_samples
        }

    def _prepare_data_for_api(self, aggregated_sensor_data: Dict[str, Any], cams_data: Dict[str, Any], 
                            target_date: date, target_time: str) -> str:
        """Prepare data for OpenAI API call for a specific time split"""
        
        prepared_data = {
            "harmonization_request_date": target_date.isoformat(),
            "harmonization_request_time": target_time,
            "aggregated_sensor_data": aggregated_sensor_data,
            "cams_data": cams_data
        }
        
        data_json = json.dumps(prepared_data, indent=2)
        
        logger.info(f"Prepared data for AI harmonization at {target_time} - "
                   f"Sensor count: {aggregated_sensor_data.get('statistics', {}).get('sensor_count', 0)}, "
                   f"CAMS fields: {len(cams_data)}")
        
        return data_json

    def _call_openai_api_for_harmonization(self, data_json: str, target_date: date, target_time: str, max_retries: int = 3) -> str:
        """Calls the OpenAI API with retry logic for a specific time split."""
        system_prompt = """
        You are an expert Air Quality Data Harmonizer.
        I will provide you aggregated Sensor.Community readings + CAMS model outputs for a specific time.
        Your task is to reconcile them and output a single, clean JSON object.
        The output format MUST be:
        {
          "harmonized_readings": [
            {
              "location": "CityName",
              "timestamp": "YYYY-MM-DDTHH:MM:SS",
              "pollutants": {
                "PM2.5": <value_float_or_null>,
                "PM10": <value_float_or_null>,
                "NO2": <value_float_or_null>,
                "O3": <value_float_or__null>,
                "SO2": <value_float_or_null>
              },
              "confidence": "High/Medium/Low",
              "source_analysis": "Brief note on how you derived the values."
            }
          ]
        }
        Generate exactly one harmonized reading. Use the provided date and time for the timestamp.
        """

        user_prompt = f"""
        Input Data for {target_date.isoformat()} at {target_time}:
        {data_json}

        Please generate exactly one harmonized reading for this specific time.
        Use {target_date.isoformat()}T{target_time}:00 for the timestamp.
        """

        logger.info(f"Sending data to OpenAI for harmonization (date={target_date.isoformat()}, time={target_time})...")
        logger.info(f"OpenAI request payload size: {len(data_json)} characters")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"OpenAI API call attempt {attempt + 1}/{max_retries}")
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    timeout=180
                )
                response_content = completion.choices[0].message.content.strip()
                logger.info(f"OpenAI API call successful. Response length: {len(response_content)} characters")
                logger.debug(f"OpenAI raw response: {response_content}")
                return response_content
            except openai.RateLimitError:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"❌ OpenAI error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
        return ""

    def _process_time_split(self, city: str, harmonization_date: date, time_split: str) -> Optional[Dict[str, Any]]:
        """Process one time split and return harmonized reading"""
        try:
            # Create target datetime for this split
            target_datetime = datetime.combine(harmonization_date, datetime.strptime(time_split, "%H:%M").time())
            
            # Get sensor data for time window (±2 hours) - USING FIXED VERSION
            sensor_records = self.db_manager.get_sensor_data_for_time_window(city, target_datetime)
            
            # Aggregate sensor data
            aggregated_sensor_data = self._aggregate_sensor_data(sensor_records)
            
            # Get CAMS data for exact time
            cams_time_code = time_split.replace(":", "")  # Convert "06:00" to "0600"
            cams_data = self.db_manager.get_cams_data_for_time(harmonization_date, cams_time_code)
            
            # Prepare and call OpenAI
            data_json = self._prepare_data_for_api(aggregated_sensor_data, cams_data, harmonization_date, time_split)
            harmonized_str = self._call_openai_api_for_harmonization(data_json, harmonization_date, time_split)
            
            if harmonized_str:
                try:
                    harmonized_json = json.loads(harmonized_str)
                    readings = harmonized_json.get("harmonized_readings", [])
                    if readings:
                        # Ensure timestamp is set correctly
                        for reading in readings:
                            reading["timestamp"] = f"{harmonization_date.isoformat()}T{time_split}:00"
                        return readings[0]  # Return first reading
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error for {time_split}: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing time split {time_split}: {e}")
            return None

    def _update_city_data(self, city: str):
        """The main orchestration for a single city with new autonomous daily mode - FIXED VERSION"""
        logger.info(f"\n--- Starting autonomous daily update for {city} ---")
        
        # Determine harmonization date (yesterday)
        today = date.today()
        harmonization_date = today - timedelta(days=1)

        # FIX: CAMS data has 2-day latency, so fetch data from 1 days ago
        cams_fetch_date = today - timedelta(days=1)
        
        # FIXED: Fetch data for the correct dates
        # Fetch Sensor.Community data for the harmonization date (yesterday)
        logger.info(f"Fetching Sensor.Community data for {city} for current date {today}")
        sensor_data = self._fetch_sensor_community_data(city, today)
        
        # Fetch CAMS data for the harmonization date (yesterday) - USING WORKING V5.7 LOGIC
        logger.info(f"Fetching CAMS data for {city} for CAMS date {cams_fetch_date} (2-day latency)")
        cams_data = self._fetch_and_extract_cams_data(city, cams_fetch_date)

        # Check last update status
        last_update = self.db_manager.get_last_update_date(city)
        
        if last_update is None:
            # First run - harmonize yesterday only
            logger.info(f"First run for {city} - harmonizing yesterday only: {harmonization_date}")
        elif last_update < harmonization_date:
            # Need update - harmonize yesterday only
            logger.info(f"Data needs update for {city} - harmonizing yesterday: {harmonization_date}")
        elif last_update == harmonization_date:
            # Already up to date - do nothing
            logger.info(f"Data for {city} already harmonized for {harmonization_date} - skipping")
            return
        else:
            # Should not happen, but handle gracefully
            logger.warning(f"Unexpected last_update_date {last_update} for {city} - harmonizing yesterday: {harmonization_date}")
        
        # Process all time splits for yesterday only
        harmonized_readings = []
        for time_split in self.TIME_SPLITS:
            logger.info(f"Processing time split: {time_split}")
            reading = self._process_time_split(city, harmonization_date, time_split)
            if reading:
                harmonized_readings.append(reading)
                logger.info(f"✅ Successfully harmonized {time_split} for {city}")
            else:
                logger.warning(f"⚠️ Failed to harmonize {time_split} for {city}")
            
            # Be polite to APIs
            time.sleep(2)
        
        # Save all harmonized readings and update status
        if harmonized_readings:
            self.db_manager.append_harmonized_data(harmonized_readings)
            self.db_manager.set_last_update_date(city, harmonization_date)
            logger.info(f"✅ Saved {len(harmonized_readings)} harmonized readings for {city} {harmonization_date}")
        else:
            logger.warning(f"⚠️ No harmonized readings generated for {city} {harmonization_date}")

    # === MANUAL CONTROL MODE METHODS ===

    def harmonize_date(self, city: str, target_date: date):
        """Manual control: Harmonize specific date only"""
        logger.info(f"🟧 MANUAL MODE: Harmonizing {city} for specific date {target_date}")
        
        # Process all time splits for the specified date
        harmonized_readings = []
        for time_split in self.TIME_SPLITS:
            logger.info(f"Processing time split: {time_split}")
            reading = self._process_time_split(city, target_date, time_split)
            if reading:
                harmonized_readings.append(reading)
                logger.info(f"✅ Successfully harmonized {time_split} for {city} {target_date}")
            else:
                logger.warning(f"⚠️ Failed to harmonize {time_split} for {city} {target_date}")
            
            time.sleep(2)
        
        # Save harmonized readings (don't update last_update_date in manual mode)
        if harmonized_readings:
            self.db_manager.append_harmonized_data(harmonized_readings)
            logger.info(f"✅ Saved {len(harmonized_readings)} harmonized readings for {city} {target_date}")
            return {"status": "success", "readings_count": len(harmonized_readings)}
        else:
            logger.warning(f"⚠️ No harmonized readings generated for {city} {target_date}")
            return {"status": "failed", "readings_count": 0}

    def harmonize_range(self, city: str, start_date: date, end_date: date):
        """Manual control: Harmonize date range (inclusive)"""
        logger.info(f"🟧 MANUAL MODE: Harmonizing {city} for date range {start_date} to {end_date}")
        
        current_date = start_date
        total_readings = 0
        
        while current_date <= end_date:
            logger.info(f"Processing {city} for {current_date}")
            result = self.harmonize_date(city, current_date)
            if result.get("status") == "success":
                total_readings += result.get("readings_count", 0)
            
            current_date += timedelta(days=1)
            
            # Be polite between days
            if current_date <= end_date:
                time.sleep(5)
        
        logger.info(f"🟧 MANUAL MODE COMPLETE: {total_readings} total readings for {city} from {start_date} to {end_date}")
        return {"status": "completed", "total_readings": total_readings}

    def fetch_cams_for_date(self, city: str, target_date: date):
        """Manual control: Fetch CAMS raw data for specific date"""
        logger.info(f"🟧 MANUAL MODE: Fetching CAMS data for {city} on {target_date}")
        
        result = self._fetch_and_extract_cams_data(city, target_date)
        
        if not result.empty:
            logger.info(f"✅ Successfully fetched CAMS data for {city} {target_date}")
            return {"status": "success", "records_count": len(result)}
        else:
            logger.warning(f"⚠️ Failed to fetch CAMS data for {city} {target_date}")
            return {"status": "failed", "records_count": 0}

    def prune_raw_data(self, before_date: date):
        """Manual control: Delete raw data older than cutoff date"""
        logger.info(f"🟧 MANUAL MODE: Pruning raw data older than {before_date}")
        
        result = self.db_manager.prune_raw_data(before_date)
        
        if "error" not in result:
            logger.info(f"✅ Successfully pruned raw data older than {before_date}")
            return {"status": "success", **result}
        else:
            logger.error(f"❌ Failed to prune raw data: {result.get('error')}")
            return {"status": "failed", "error": result.get("error")}

    def run(self, payload: dict) -> dict:
        """
        Main entry point for the agent (LangChain compatible).
        
        Payload is optional. If not provided, it runs for default cities.
        Example payload: {"cities": ["Heidelberg", "Berlin"]}
        """
        logger.info("🟢 HarmoniserAgent received payload:")
        logger.info(json.dumps(payload, indent=2))
        
        cities_to_run = payload.get("cities", ["Heidelberg", "Berlin"])
        harmonization_date = date.today() - timedelta(days=1)
        
        all_results = []
        
        for city in cities_to_run:
            if city in self.CITY_URLS:
                self._update_city_data(city)
                
                # Get today's harmonized readings for this city
                city_readings = []
                for time_split in self.TIME_SPLITS:
                    # Construct timestamp for querying
                    timestamp_str = f"{harmonization_date.isoformat()}T{time_split}:00"
                    try:
                        timestamp_obj = datetime.fromisoformat(timestamp_str)
                        # Query for this specific reading
                        session = ScopedSession()
                        reading = session.query(HarmonizedReading).filter(
                            HarmonizedReading.location.ilike(f"%{city}%"),
                            HarmonizedReading.timestamp == timestamp_obj
                        ).first()
                        
                        if reading:
                            city_readings.append({
                                "location": reading.location,
                                "timestamp": reading.timestamp.isoformat(),
                                "pollutants": {
                                    "PM2.5": reading.pollutant_pm25,
                                    "PM10": reading.pollutant_pm10,
                                    "NO2": reading.pollutant_no2,
                                    "O3": reading.pollutant_o3,
                                    "SO2": reading.pollutant_so2
                                },
                                "other_data": reading.other_data
                            })
                    except Exception as e:
                        logger.error(f"Error retrieving reading for {city} {timestamp_str}: {e}")
                    finally:
                        ScopedSession.remove()
                
                all_results.extend(city_readings)
                
                # Pause between cities to respect rate limits
                if city != cities_to_run[-1]:
                    logger.info("Pausing for 20 seconds to avoid OpenAI rate limiting...")
                    time.sleep(20)
            else:
                logger.warning(f"Skipping unknown city: {city}")

        return {
            "status": "success",
            "harmonization_date": harmonization_date.isoformat(),
            "results": all_results
        }


# --- Main Run ---
if __name__ == "__main__":
    print("--- Harmonizer Agent V6.0 (Multi-Sensor Aggregation) ---")
    print(f"Connecting to database: {DB_DRIVER} on {DB_HOST}:{DB_PORT}")
    print(f"Log file: {log_filename}")
    
    # We also need a CDS API key. Users must create a .cdsapirc file.
    cds_api_rc_path = Path.home() / ".cdsapirc"
    if not cds_api_rc_path.exists():
        logger.warning("="*50)
        logger.warning("WARNING: '.cdsapirc' file not found in home directory.")
        logger.warning("Please move your .cdsapirc file to /home/ai4air/")
        logger.warning("You can get a key from: https://cds.climate.copernicus.eu/api-how-to")
        logger.warning("Example content:")
        logger.warning("url: https://cds.climate.copernicus.eu/api/v2")
        logger.warning("key: 12345:abcdef-1234-5678-abcd-abcdef123456")
        logger.warning("="*50)
    
    try:
        # 1. Initialize the Database Manager
        db_manager = DatabaseManager(ScopedSession)
        
        # 2. Create tables if they don't exist (idempotent)
        db_manager.create_tables()

        # 3. Initialize the Agent
        harmoniser_agent = HarmoniserAgent(
            db_manager=db_manager,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 4. Run the agent
        logger.info("Starting main harmonization process...")
        run_results = harmoniser_agent.run({})
        
        logger.info("\n--- Agent Run Complete ---")
        logger.info(f"Final results: {json.dumps(run_results, default=str)}")

        print("\n--- Agent Run Complete ---")
        print(json.dumps(run_results, indent=2, default=str))
        print(f"\nDetailed logs available at: {log_filename}")

    except Exception as e:
        logger.error(f"\n--- A critical error occurred ---")
        logger.error(f"Error: {e}", exc_info=True)
        if "OperationalError" in str(e) or "Connection refused" in str(e):
            logger.error(">>> CRITICAL: Could not connect to the database.")
            logger.error(">>> Please check your DB_USER, DB_PASSWORD, DB_HOST, and DB_PORT settings.")
        
        print(f"\n--- A critical error occurred ---")
        print(f"Error: {e}")
        print(f"Check detailed logs at: {log_filename}")
    finally:
        # Clean up the session context
        ScopedSession.remove()
        logger.info("Application shutdown complete.")
