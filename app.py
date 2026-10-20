from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from tracker_engine import parse_tle, compute_track, find_passes

st.set_page_config(page_title="Satellite Tracker", layout="wide")
st.title("Ground Station Tracking")

TLE_FILE = Path(__file__).resolve().parent / "tle.txt"
R_EARTH_KM = 6371.0


@st.cache_data(show_spinner=False)
def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text


@st.cache_data(show_spinner="Computing track...")
def run(line1, line2, name, lat, lon, alt, start_iso, hours, min_el):
    start = datetime.fromisoformat(start_iso)
    end = start + timedelta(hours=hours)
    df, epoch = compute_track(line1, line2, name, lat, lon, alt, start, end, 1, min_el)
    return df, epoch, find_passes(df, min_el)


def split_wrap(lat, lon):
    # Insert gaps where the line crosses the +-180 deg longitude edge.
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    jump = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    return np.insert(lat, jump, np.nan), np.insert(lon, jump, np.nan)


def footprint(lat0, lon0, alt_km, min_el_deg, n=120):
    # Circle on Earth from which the satellite is above min_el_deg.
    el = np.radians(min_el_deg)
    lam = np.arccos(R_EARTH_KM * np.cos(el) / (R_EARTH_KM + alt_km)) - el
    la1, lo1 = np.radians(lat0), np.radians(lon0)
    brg = np.linspace(0, 2 * np.pi, n)
    la2 = np.arcsin(np.sin(la1) * np.cos(lam) + np.cos(la1) * np.sin(lam) * np.cos(brg))
    lo2 = lo1 + np.arctan2(np.sin(brg) * np.sin(lam) * np.cos(la1),
                           np.cos(lam) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 180) % 360 - 180


# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.header("Setup")
    norad = st.number_input("NORAD ID", value=38771, step=1)
    name = st.text_input("Satellite name", "METOP-B")
    source = st.radio("TLE source", ["tle.txt file", "CelesTrak (latest)", "Paste TLE"])
    pasted = ""
    if source == "Paste TLE":
        pasted = st.text_area("TLE (2 lines)", height=120)
    st.subheader("Ground station")
    lat = st.number_input("Latitude (deg)", value=30.143333, format="%.6f")
    lon = st.number_input("Longitude (deg)", value=31.718333, format="%.6f")
    alt = st.number_input("Altitude (m)", value=208.0)
    min_el = st.number_input("Min elevation (deg)", value=10.0)
    mode = st.radio("Mode", ["Planning (fixed day)", "Now (live clock)"])
    st.subheader("Time window (UTC)")
    start_date = st.date_input("Start date", value=datetime(2026, 10, 17).date())
    hours = st.slider("Window (hours)", 1, 48, 24)

try:
    if source == "tle.txt file":
        text = TLE_FILE.read_text(encoding="utf-8")
    elif source == "CelesTrak (latest)":
        text = fetch_tle(int(norad))
    else:
        text = pasted
    line1, line2 = parse_tle(text, int(norad))
except Exception as exc:
    st.error(f"TLE problem: {exc}")
    st.stop()

if mode.startswith("Now"):
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    start = now_utc.replace(minute=0, second=0)
    hours = 25
else:
    now_utc = None
    start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
df, epoch, passes = run(line1, line2, name, lat, lon, alt, start.isoformat(), hours, min_el)
st.caption(f"{name} (NORAD {int(norad)}) | TLE epoch UTC: {epoch.isoformat()} | "
           f"window: {hours} h from {start.date()} | all times UTC")

@st.fragment(run_every=1)
def live_now_panel(df, passes, start):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    i = int((now - start).total_seconds())
    if i < 0 or i >= len(df):
        st.warning("Live window ended - refresh the page.")
        return
    r = df.iloc[i]
    az = r["azimuth_deg"]
    el = r["elevation_deg"]
    rng = r["slant_range_km"]
    vis = "VISIBLE" if r["above_min_elevation"] else "below the elevation mask"
    st.write("**LIVE (UTC) " + now.strftime("%H:%M:%S") + "** - " + vis)
    c1, c2, c3 = st.columns(3)
    c1.metric("Azimuth", f"{az:.1f}°")
    c2.metric("Elevation", f"{el:.1f}°")
    c3.metric("Distance", f"{rng:.0f} km")
    if len(passes):
        nxt = passes[pd.to_datetime(passes["aos_utc"], utc=True) > now]
        if len(nxt):
            p = nxt.iloc[0]
            mx = p["max_el_deg"]
            s = int((pd.to_datetime(p["aos_utc"], utc=True) - now).total_seconds())
            st.write(f"Next pass in {s // 3600} h {s % 3600 // 60} min {s % 60} s - max elevation {mx} deg")

tab_live, tab_pass, tab_export = st.tabs(["Live Tracking", "Passes", "Export / Handover"])

# ---------------------------- Live Tracking ----------------------------
with tab_live:
    if now_utc is not None:
        live_now_panel(df, passes, start)
    default_idx = 0
    if len(passes):
        hit = df.index[df["timestamp_utc"] == passes.iloc[0]["max_el_time_utc"]]
        if len(hit):
            default_idx = (int(hit[0]) // 10) * 10
    if now_utc is not None:
        default_idx = min(int((now_utc - start).total_seconds()) // 10 * 10, len(df) - 1)
    idx = st.slider("Time in window (sample index, 1 s each)", 0, len(df) - 1,
                    default_idx, step=10)
    row = df.iloc[idx]
    st.write(f"**Selected time (UTC): {row['timestamp_utc']}**  -  "
             + ("visible" if row["above_min_elevation"] else "below the elevation mask"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Azimuth", f"{row['azimuth_deg']:.1f}°")
    c2.metric("Elevation", f"{row['elevation_deg']:.1f}°")
    c3.metric("Altitude", f"{row['satellite_altitude_km']:.0f} km")
    c4.metric("Distance", f"{row['slant_range_km']:.0f} km")
    st.code(f"AZ{row['azimuth_deg']:.1f},EL{row['elevation_deg']:.1f}", language="text")
    st.caption("Preview of the serial packet for the MCU (format to be agreed with the MCU member).")

    full_track = st.checkbox("Show the full window track (all orbits)", value=False)
    if full_track:
        past, future = df.iloc[::10], df.iloc[0:0]
    else:
        past = df.iloc[max(0, idx - 3000): idx + 1: 10]
        future = df.iloc[idx: idx + 3001: 10]
    plat, plon = split_wrap(past["satellite_latitude_deg"], past["satellite_longitude_deg"])
    tlat, tlon = split_wrap(future["satellite_latitude_deg"], future["satellite_longitude_deg"])
    flat, flon = footprint(row["satellite_latitude_deg"], row["satellite_longitude_deg"],
                           row["satellite_altitude_km"], min_el)
    flat, flon = split_wrap(flat, flon)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=plat, lon=plon, mode="lines",
                                name="Track (past)" if not full_track else "Track (all)",
                                line=dict(width=2, color="#8aa4bd" if not full_track else "#1f77b4",
                                          dash="dot" if not full_track else "solid")))
    fig.add_trace(go.Scattergeo(lat=tlat, lon=tlon, mode="lines", name="Track (next ~50 min)",
                                line=dict(width=3, color="#1f77b4")))
    fig.add_trace(go.Scattergeo(lat=flat, lon=flon, mode="lines", name="Footprint",
                                line=dict(width=2, color="#d62728")))
    fig.add_trace(go.Scattergeo(lat=[lat], lon=[lon], mode="markers+text", name="Station",
                                text=["Station"], textposition="bottom center",
                                marker=dict(size=9, color="black")))
    fig.add_trace(go.Scattergeo(lat=[row["satellite_latitude_deg"]],
                                lon=[row["satellite_longitude_deg"]],
                                mode="markers+text", name=name, text=[name],
                                textposition="top center",
                                marker=dict(size=11, color="#ff7f0e")))
    fig.update_geos(projection_type="natural earth", showcountries=True,
                    showland=True, landcolor="#f0efe6", showocean=True,
                    oceancolor="#dbe9f6")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Passes --------------------------------
with tab_pass:
    if passes.empty:
        st.info("No passes above the elevation mask in this window.")
    else:
        st.subheader("Pass table")
        st.dataframe(passes, use_container_width=True, hide_index=True)
        st.caption("AOS/LOS here = crossing of the min-elevation mask (sampling resolution 1 s). "
                   "partial = the time window cuts the pass.")
        n = st.selectbox("Select pass", list(passes["pass_no"]))
        p = passes[passes["pass_no"] == n].iloc[0]
        seg = df[(df["timestamp_utc"] >= p["aos_utc"]) & (df["timestamp_utc"] <= p["los_utc"])]

        m1, m2, m3 = st.columns(3)
        m1.metric("Max elevation", f"{p['max_el_deg']:.2f}°")
        m2.metric("Duration", f"{p['duration_s'] // 60} min {p['duration_s'] % 60} s")
        m3.metric("Az AOS to LOS", f"{p['aos_az_deg']:.0f}° to {p['los_az_deg']:.0f}°")

        left, right = st.columns(2)
        fe = go.Figure(go.Scatter(x=seg["timestamp_utc"], y=seg["elevation_deg"], mode="lines"))
        fe.update_layout(title="Elevation vs time (UTC)", height=380,
                         xaxis_title="Time (UTC)", yaxis_title="Elevation (deg)")
        left.plotly_chart(fe, use_container_width=True)

        fs = go.Figure(go.Scatterpolar(theta=seg["azimuth_deg"], r=90 - seg["elevation_deg"],
                                       mode="lines"))
        fs.update_layout(title="Sky plot (Az / El)", height=380, showlegend=False,
                         polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                                    radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                                                    ticktext=["90°", "60°", "30°", "0°"])))
        right.plotly_chart(fs, use_container_width=True)


# --------------------------- Export / Handover ---------------------------
def handover_table(df, passes):
    rows = []
    for _, p in passes.iterrows():
        seg = df[(df["timestamp_utc"] >= p["aos_utc"]) & (df["timestamp_utc"] <= p["los_utc"])]
        if len(seg) < 2:
            continue
        az = np.degrees(np.unwrap(np.radians(seg["azimuth_deg"].to_numpy())))
        el = seg["elevation_deg"].to_numpy()
        az_rate = np.abs(np.diff(az))  # deg per 1 s sample = deg/s
        el_rate = np.abs(np.diff(el))
        rows.append({
            "pass_no": p["pass_no"],
            "aos_utc": p["aos_utc"],
            "los_utc": p["los_utc"],
            "aos_az_deg": p["aos_az_deg"],
            "los_az_deg": p["los_az_deg"],
            "az_sweep_deg": round(float(az[-1] - az[0]), 1),
            "el_min_deg": round(float(el.min()), 1),
            "el_max_deg": round(float(el.max()), 1),
            "max_az_rate_deg_s": round(float(az_rate.max()), 2),
            "max_el_rate_deg_s": round(float(el_rate.max()), 2),
        })
    return pd.DataFrame(rows)


with tab_export:
    st.subheader("Files for the Task 1 checklist")
    d1, d2, d3 = st.columns(3)
    d1.download_button("tracking_results.csv", df.to_csv(index=False, float_format="%.6f"),
                       "tracking_results.csv", "text/csv")
    d2.download_button("pass_prediction.csv", passes.to_csv(index=False),
                       "pass_prediction.csv", "text/csv")
    d3.download_button("tle.txt (TLE used)", f"{line1}\n{line2}\n", "tle.txt", "text/plain")

    st.subheader("Handover to Task 2 (Mechanical Design)")
    if passes.empty:
        st.info("No passes in this window, nothing to hand over.")
    else:
        ho = handover_table(df, passes)
        st.dataframe(ho, use_container_width=True, hide_index=True)
        st.caption(f"Satellite {name} (NORAD {int(norad)}), station {lat:.6f}, {lon:.6f}, {alt:.0f} m, "
                   f"elevation mask {min_el:.0f} deg. az_sweep_deg is the total azimuth travelled "
                   "(sign = direction, no 0/360 jump). Rates are measured over 1 s steps, so a very "
                   "high pass shows a very high azimuth rate near the zenith.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Elevation range", f"{ho['el_min_deg'].min():.0f}° to {ho['el_max_deg'].max():.0f}°")
        k2.metric("Max azimuth rate", f"{ho['max_az_rate_deg_s'].max():.2f} °/s")
        k3.metric("Max elevation rate", f"{ho['max_el_rate_deg_s'].max():.2f} °/s")
        st.download_button("handover_task2.csv", ho.to_csv(index=False),
                           "handover_task2.csv", "text/csv")
