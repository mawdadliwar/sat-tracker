from datetime import datetime, timezone
from pathlib import Path
from tracker_engine import parse_tle, compute_track, find_passes

l1, l2 = parse_tle(Path("tle.txt").read_text(), 38771)
df, epoch = compute_track(
    l1, l2, "METOP-B", 30.143333, 31.718333, 208,
    datetime(2026, 10, 17, tzinfo=timezone.utc),
    datetime(2026, 10, 18, tzinfo=timezone.utc),
)
print("samples:", len(df), "| above 10 deg:", int(df["above_min_elevation"].sum()))
print(find_passes(df).to_string(index=False))
