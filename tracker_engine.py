from datetime import timedelta
import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

_ts = load.timescale()


def parse_tle(text, norad_id=None):
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    line1 = next((l for l in lines if l.startswith("1 ")), None)
    line2 = next((l for l in lines if l.startswith("2 ")), None)
    if line1 is None or line2 is None:
        raise ValueError("TLE must contain one line 1 and one line 2.")
    id1, id2 = int(line1[2:7]), int(line2[2:7])
    if id1 != id2:
        raise ValueError(f"NORAD ID mismatch inside TLE: {id1} vs {id2}.")
    if norad_id is not None and id1 != norad_id:
        raise ValueError(f"TLE NORAD ID mismatch: expected {norad_id}, got {id1}.")
    return line1, line2


def compute_track(line1, line2, name, lat, lon, alt_m,
                  start_utc, end_utc, step_s=1, min_el=10.0):
    # Same math as tracking_simulation.py, wrapped in a function.
    if end_utc <= start_utc:
        raise ValueError("End time must be later than start time.")
    satellite = EarthSatellite(line1, line2, name, _ts)
    station = wgs84.latlon(lat, lon, elevation_m=alt_m)

    total = int((end_utc - start_utc).total_seconds())
    offsets = np.arange(0, total + 1, step_s, dtype=int)
    datetimes = [start_utc + timedelta(seconds=int(s)) for s in offsets]
    times = _ts.from_datetimes(datetimes)

    subpoint = wgs84.subpoint(satellite.at(times))
    alt, az, dist = (satellite - station).at(times).altaz()

    df = pd.DataFrame({
        "timestamp_utc": [d.isoformat().replace("+00:00", "Z") for d in datetimes],
        "satellite_latitude_deg": np.asarray(subpoint.latitude.degrees),
        "satellite_longitude_deg": np.asarray(subpoint.longitude.degrees),
        "satellite_altitude_km": np.asarray(subpoint.elevation.km),
        "azimuth_deg": np.asarray(az.degrees),
        "elevation_deg": np.asarray(alt.degrees),
        "slant_range_km": np.asarray(dist.km),
    })
    df["above_min_elevation"] = df["elevation_deg"] >= min_el
    return df, satellite.epoch.utc_datetime()


def find_passes(df, min_el=10.0):
    # A pass = consecutive samples at or above min_el.
    # AOS/LOS here mean the min_el crossing, with the sampling step as resolution.
    mask = (df["elevation_deg"] >= min_el).to_numpy()
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask) - 1)

    rows = []
    for n, (s, e) in enumerate(zip(starts, ends), start=1):
        seg = df.iloc[s:e + 1]
        peak = seg["elevation_deg"].idxmax()
        t0 = pd.to_datetime(df["timestamp_utc"].iloc[s], utc=True)
        t1 = pd.to_datetime(df["timestamp_utc"].iloc[e], utc=True)
        rows.append({
            "pass_no": n,
            "aos_utc": df["timestamp_utc"].iloc[s],
            "los_utc": df["timestamp_utc"].iloc[e],
            "max_el_deg": round(float(df["elevation_deg"].loc[peak]), 2),
            "max_el_time_utc": df["timestamp_utc"].loc[peak],
            "duration_s": int((t1 - t0).total_seconds()),
            "aos_az_deg": round(float(df["azimuth_deg"].iloc[s]), 1),
            "los_az_deg": round(float(df["azimuth_deg"].iloc[e]), 1),
            "partial": bool(s == 0 or e == len(mask) - 1),
        })
    return pd.DataFrame(rows)
