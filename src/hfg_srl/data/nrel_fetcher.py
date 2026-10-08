"""NREL NSRDB data downloader — fetch solar irradiance data from NREL's NSRDB API.

NREL's National Solar Radiation Database (NSRDB) provides hourly/solar-resolution
solar radiation data for the United States and many other locations.

Usage (from command line):
    python -m hfg_srl.data.nrel_fetcher \
        --api_key YOUR_API_KEY \
        --lat 39.7392 --lon -104.9903 \
        --year 2020 \
        --output data/raw/nsrdb_denver_2020.csv

You can get a free API key from: https://developer.nrel.gov/signup/

If you don't have an API key, you can use --mode synthetic to generate
synthetic data with NREL-like statistical properties.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- NSRDB API endpoints ---
NSRDB_BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

# Column mapping from NSRDB raw output to our schema
NSRDB_COLUMN_MAP = {
    "GHI": "ghi_wm2",       # Global Horizontal Irradiance (W/m²)
    "DNI": "dni_wm2",       # Direct Normal Irradiance
    "DHI": "dhi_wm2",       # Diffuse Horizontal Irradiance
    "Temperaure": "temp_c", # Ambient temperature (°C)
    "Wind Speed": "wind_speed_ms",
    "Pressure": "pressure_mbar",
}


def fetch_nsrdb_data(
    api_key: str,
    lat: float,
    lon: float,
    year: int = 2020,
    interval: str = "60",  # 30 or 60 minutes
    attributes: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fetch solar irradiance data from NREL NSRDB API.

    Args:
        api_key: NREL developer API key.
        lat: Latitude of the location.
        lon: Longitude of the location.
        year: Year of data to fetch.
        interval: Time interval in minutes ('30' or '60').
        attributes: List of NSRDB attributes to request.
            Defaults to GHI, DNI, DHI, Temperature, Wind Speed.
        output_path: If set, save the raw CSV to this path.

    Returns:
        DataFrame with datetime index and solar/weather columns.
    """
    import urllib.request
    import urllib.parse

    if attributes is None:
        attributes = ["ghi", "dni", "dhi", "temperature", "wind_speed"]

    params = {
        "api_key": api_key,
        "lat": lat,
        "lon": lon,
        "year": year,
        "interval": interval,
        "attributes": ",".join(attributes),
        "email": "example@email.com",
        "mailing_list": "false",
    }

    url = f"{NSRDB_BASE_URL}?{urllib.parse.urlencode(params)}"
    logger.info(f"Fetching NSRDB data from {url[:100]}...")

    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            raw_data = response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NSRDB data: {e}") from e

    # NSRDB CSV has a metadata header — skip to column headers
    lines = raw_data.split("\n")
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Year"):
            header_idx = i
            break

    df = pd.read_csv(
        pd.io.common.StringIO("\n".join(lines[header_idx:]))
    )

    # Build datetime index
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
            columns={
                "Year": "year", "Month": "month", "Day": "day",
                "Hour": "hour", "Minute": "minute",
            }
        )
    )
    df = df.set_index("timestamp")
    df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"])

    # Clean column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Saved NSRDB data to {output_path} ({len(df)} rows)")

    return df


def generate_synthetic_nsrdb_like(
    lat: float = 39.74,
    lon: float = -104.99,
    year: int = 2020,
    interval_minutes: int = 60,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate synthetic solar data with NREL-like statistical properties.

    Use this as a fallback when you don't have an NREL API key.
    The data follows realistic diurnal + seasonal patterns for the given latitude.

    Args:
        lat: Latitude (affects seasonal solar intensity).
        lon: Longitude (affects time offset, minor effect).
        year: Year for the timestamp index.
        interval_minutes: Time step in minutes.
        seed: Random seed.
        output_path: Optional path to save the generated CSV.

    Returns:
        DataFrame with datetime index and solar/weather columns.
    """
    rng = np.random.default_rng(seed)

    steps_per_day = 24 * 60 // interval_minutes
    days = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    total_steps = steps_per_day * days

    # Generate timestamps
    timestamps = pd.date_range(
        start=f"{year}-01-01",
        periods=total_steps,
        freq=f"{interval_minutes}min",
    )

    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour + timestamps.minute / 60.0

    # --- Solar declination and hour angle ---
    declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    hour_angle = np.radians((hour_of_day - 12.0) * 15.0)

    # Cosine of solar zenith angle
    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

    # GHI: clear-sky model with cloud cover noise
    clear_sky_ghi = 1000.0 * cos_zenith ** 1.2  # ~ 1000 W/m² at zenith
    # Cloud cover: random reduction, autocorrelated
    cloud = np.ones(total_steps)
    cloud_noise = rng.normal(0, 0.15, total_steps)
    # Simple autocorrelation
    for i in range(1, total_steps):
        cloud[i] = 0.9 * cloud[i-1] + 0.1 * (1.0 - abs(cloud_noise[i]))
    cloud = np.clip(cloud, 0.1, 1.0)

    ghi = clear_sky_ghi * cloud
    ghi = np.clip(ghi, 0.0, 1200.0)

    # DNI and DHI (approximate from GHI)
    dni = np.where(
        cos_zenith > 0.01,
        ghi / cos_zenith * 0.75,
        0.0,
    )
    dni = np.clip(dni, 0.0, 1000.0)
    dhi = np.clip(ghi - dni * cos_zenith * 0.9, 0.0, None)

    # Temperature: diurnal + seasonal
    temp_seasonal = 15.0 + 12.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    temp_diurnal = 5.0 * np.sin(2 * np.pi * (hour_of_day - 14.0) / 24.0)
    temperature = temp_seasonal + temp_diurnal + rng.normal(0, 1.5, total_steps)

    # Wind speed: Weibull-like distribution
    wind_speed = rng.weibull(2.0, total_steps) * 3.5
    wind_speed = np.clip(wind_speed, 0.0, 25.0)

    df = pd.DataFrame(
        {
            "ghi_wm2": ghi,
            "dni_wm2": dni,
            "dhi_wm2": dhi,
            "temp_c": temperature,
            "wind_speed_ms": wind_speed,
        },
        index=timestamps,
    )
    df.index.name = "timestamp"

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Saved synthetic NSRDB-like data to {output_path} ({len(df)} rows)")

    return df


def ghi_to_pv_output(
    ghi: pd.Series | np.ndarray,
    pv_capacity_kw: float,
    temp_c: Optional[pd.Series | np.ndarray] = None,
    derate_factor: float = 0.82,
    temp_coeff: float = -0.004,  # per °C
    ref_temp_c: float = 25.0,
) -> np.ndarray:
    """Convert GHI (W/m²) to PV output (kW).

    Simple model: P = GHI/1000 * P_max * derate * (1 + temp_coeff * (T - T_ref))

    Args:
        ghi: Global Horizontal Irradiance in W/m².
        pv_capacity_kw: Nameplate PV capacity in kW.
        temp_c: Ambient temperature in °C (optional, for temperature correction).
        derate_factor: System derate factor (inverter, wiring, soiling, etc.).
        temp_coeff: Temperature coefficient of power (per °C).
        ref_temp_c: Reference temperature (°C).

    Returns:
        PV output in kW.
    """
    ghi_arr = np.asarray(ghi, dtype=np.float64)
    output = ghi_arr / 1000.0 * pv_capacity_kw * derate_factor

    if temp_c is not None:
        temp_arr = np.asarray(temp_c, dtype=np.float64)
        output *= (1.0 + temp_coeff * (temp_arr - ref_temp_c))

    return np.clip(output, 0.0, pv_capacity_kw)


def main():
    parser = argparse.ArgumentParser(description="NREL NSRDB data fetcher")
    parser.add_argument("--api_key", type=str, default=None,
                        help="NREL developer API key (if not set, generates synthetic data)")
    parser.add_argument("--lat", type=float, default=39.7392,
                        help="Latitude (default: Denver)")
    parser.add_argument("--lon", type=float, default=-104.9903,
                        help="Longitude (default: Denver)")
    parser.add_argument("--year", type=int, default=2020,
                        help="Year of data")
    parser.add_argument("--interval", type=str, default="60",
                        choices=["30", "60"],
                        help="Time interval in minutes")
    parser.add_argument("--output", type=str, default="data/raw/nsrdb_data.csv",
                        help="Output CSV path")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "api", "synthetic"],
                        help="Data fetch mode: api (requires key), synthetic, or auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    use_api = (args.mode == "api") or (args.mode == "auto" and args.api_key is not None)

    if use_api and args.api_key:
        df = fetch_nsrdb_data(
            api_key=args.api_key,
            lat=args.lat,
            lon=args.lon,
            year=args.year,
            interval=args.interval,
            output_path=args.output,
        )
    else:
        if args.mode == "api":
            logger.warning("API mode requested but no api_key provided. Falling back to synthetic.")
        logger.info("Generating synthetic NSRDB-like data...")
        df = generate_synthetic_nsrdb_like(
            lat=args.lat,
            lon=args.lon,
            year=args.year,
            interval_minutes=int(args.interval),
            seed=args.seed,
            output_path=args.output,
        )

    print(f"Done. Shape: {df.shape}")
    print(df.describe())


if __name__ == "__main__":
    main()
