from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skyfield.api import EarthSatellite, load, wgs84

# ------------------------- Project configuration -------------------------
BASE_DIR = Path(__file__).resolve().parent
TLE_FILE = BASE_DIR / "tle.txt"

SATELLITE_NAME = "METOP-B"
NORAD_ID = 38771

LAT_DEG = 30.143333
LON_DEG = 31.718333
ALT_M = 208

START_UTC = datetime(2026, 10, 17, 0, 0, 0, tzinfo=timezone.utc)
END_UTC = datetime(2026, 10, 18, 0, 0, 0, tzinfo=timezone.utc)

MIN_ELEVATION_DEG = 10.0
SAMPLE_INTERVAL_SECONDS = 1  # Change only if the project team agrees.

CSV_FILE = BASE_DIR / "tracking_results.csv"
AZ_PLOT_FILE = BASE_DIR / "azimuth_vs_time.png"
EL_PLOT_FILE = BASE_DIR / "elevation_vs_time.png"
SUMMARY_FILE = BASE_DIR / "tracking_run_summary.txt"


def read_tle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.name}. Put tle.txt in the same folder as this script."
        )

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)

    if line1 is None or line2 is None:
        raise ValueError("tle.txt must contain one TLE line 1 and one TLE line 2.")

    try:
        id1 = int(line1[2:7])
        id2 = int(line2[2:7])
    except ValueError as exc:
        raise ValueError("Could not parse NORAD catalog number from the TLE.") from exc

    if id1 != NORAD_ID or id2 != NORAD_ID:
        raise ValueError(
            f"TLE NORAD ID mismatch: expected {NORAD_ID}, got line 1={id1}, line 2={id2}."
        )
    return line1, line2


def main():
    if END_UTC <= START_UTC:
        raise ValueError("END_UTC must be later than START_UTC.")
    if SAMPLE_INTERVAL_SECONDS <= 0:
        raise ValueError("SAMPLE_INTERVAL_SECONDS must be positive.")

    line1, line2 = read_tle(TLE_FILE)

    # Skyfield uses UTC-aware datetimes for these conversions.
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, SATELLITE_NAME, ts)
    station = wgs84.latlon(LAT_DEG, LON_DEG, elevation_m=ALT_M)

    total_seconds = int((END_UTC - START_UTC).total_seconds())
    offsets = np.arange(0, total_seconds + 1, SAMPLE_INTERVAL_SECONDS, dtype=int)
    datetimes = [START_UTC + timedelta(seconds=int(s)) for s in offsets]
    times = ts.from_datetimes(datetimes)

    # Satellite geocentric positions, converted to WGS84 sub-satellite coordinates.
    geocentric = satellite.at(times)
    subpoint = wgs84.subpoint(geocentric)
    sat_lat = np.asarray(subpoint.latitude.degrees)
    sat_lon = np.asarray(subpoint.longitude.degrees)
    sat_alt_km = np.asarray(subpoint.elevation.km)

    # Direction from the ground station to the satellite.
    topocentric = (satellite - station).at(times)
    alt, az, distance = topocentric.altaz()
    elevation_deg = np.asarray(alt.degrees)
    azimuth_deg = np.asarray(az.degrees)
    range_km = np.asarray(distance.km)

    df = pd.DataFrame({
        "timestamp_utc": [dt.isoformat().replace("+00:00", "Z") for dt in datetimes],
        "satellite_latitude_deg": sat_lat,
        "satellite_longitude_deg": sat_lon,
        "satellite_altitude_km": sat_alt_km,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "slant_range_km": range_km,
        "above_min_elevation": elevation_deg >= MIN_ELEVATION_DEG,
    })

    # Basic data-quality checks.
    numeric_cols = [
        "satellite_latitude_deg", "satellite_longitude_deg",
        "satellite_altitude_km", "azimuth_deg", "elevation_deg", "slant_range_km"
    ]
    nonfinite_counts = {
        col: int((~np.isfinite(df[col].to_numpy())).sum()) for col in numeric_cols
    }
    if any(count > 0 for count in nonfinite_counts.values()):
        raise ValueError(f"Non-finite numeric values detected: {nonfinite_counts}")

    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("UTC timestamps are not increasing.")
    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps detected.")
    if not df["satellite_latitude_deg"].between(-90, 90).all():
        raise ValueError("Satellite latitude outside [-90, 90] degrees.")
    if not df["satellite_longitude_deg"].between(-180, 180).all():
        raise ValueError("Satellite longitude outside [-180, 180] degrees.")
    if not df["azimuth_deg"].between(0, 360).all():
        raise ValueError("Azimuth outside [0, 360] degrees.")
    if not df["elevation_deg"].between(-90, 90).all():
        raise ValueError("Elevation outside [-90, 90] degrees.")

    # Azimuth is circular: a transition such as 359° -> 0° is normal.
    az_step = np.abs(np.diff(azimuth_deg))
    az_wrap_count = int((az_step > 180).sum())

    df.to_csv(CSV_FILE, index=False, float_format="%.6f")

    # Use UTC labels and readable date ticks.
    plot_df = df.copy()
    plot_df["time"] = pd.to_datetime(plot_df["timestamp_utc"], utc=True)

    plt.figure(figsize=(13, 5))
    plt.plot(plot_df["time"], plot_df["azimuth_deg"], linewidth=0.8)
    plt.title("METOP-B Azimuth vs Time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Azimuth (degrees)")
    plt.ylim(0, 360)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(AZ_PLOT_FILE, dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(plot_df["time"], plot_df["elevation_deg"], linewidth=0.8)
    plt.axhline(
        MIN_ELEVATION_DEG, linestyle="--", linewidth=1.0,
        label=f"Minimum elevation = {MIN_ELEVATION_DEG:.1f}°"
    )
    plt.title("METOP-B Elevation vs Time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Elevation (degrees)")
    plt.ylim(-90, 90)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EL_PLOT_FILE, dpi=160)
    plt.close()

    epoch_dt = satellite.epoch.utc_datetime().astimezone(timezone.utc)
    above_count = int(df["above_min_elevation"].sum())
    summary = [
        "MEMBER 2 — TRACKING SIMULATION RUN SUMMARY",
        f"Satellite: {SATELLITE_NAME}",
        f"NORAD ID: {NORAD_ID}",
        f"TLE epoch UTC: {epoch_dt.isoformat()}",
        f"Ground station latitude: {LAT_DEG:.6f} deg",
        f"Ground station longitude: {LON_DEG:.6f} deg",
        f"Ground station altitude: {ALT_M} m",
        f"Observation start UTC: {START_UTC.isoformat()}",
        f"Observation end UTC: {END_UTC.isoformat()}",
        f"Minimum elevation mask: {MIN_ELEVATION_DEG:.1f} deg",
        f"Sampling interval: {SAMPLE_INTERVAL_SECONDS} second(s)",
        f"Number of samples: {len(df)}",
        f"Samples at or above minimum elevation: {above_count}",
        f"Azimuth wraparound transitions (>180 deg between samples): {az_wrap_count}",
        f"Non-finite value counts: {nonfinite_counts}",
        "",
        "Outputs:",
        f"- {CSV_FILE.name}",
        f"- {AZ_PLOT_FILE.name}",
        f"- {EL_PLOT_FILE.name}",
        "",
        "Notes:",
        "- Azimuth wraparound near 0/360 degrees is expected for a circular angle.",
        "- satellite_altitude_km is altitude above the WGS84 ellipsoid; slant_range_km is station-to-satellite distance.",
        "- The above_min_elevation column is a mask for Member 3; this CSV includes all timestamps.",
        "- Confirm the TLE source/freshness and station coordinates before final submission.",
    ]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("\nCompleted successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
