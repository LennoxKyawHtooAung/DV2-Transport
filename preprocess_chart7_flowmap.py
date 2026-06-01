import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "rapidrail_2026_daily.csv"

OUTPUT_LINES = DATA_DIR / "rapidkl_flowmap_lines.csv"
OUTPUT_NODES = DATA_DIR / "rapidkl_flowmap_nodes.csv"
OUTPUT_LABELS = DATA_DIR / "rapidkl_flowmap_labels.csv"
OUTPUT_MISSING = DATA_DIR / "rapidkl_flowmap_missing_stations.csv"

CHUNK_SIZE = 300000
TOP_CORRIDORS = 8


# =========================
# Manual coordinate lookup
# =========================
STATION_INFO = {
    "KLCC": {
        "line": "Kelana Jaya Line",
        "lon": 101.7131,
        "lat": 3.1597
    },
    "Ampang Park": {
        "line": "Kelana Jaya Line",
        "lon": 101.7190,
        "lat": 3.1595
    },
    "Bukit Bintang": {
        "line": "Kajang / Monorail",
        "lon": 101.7118,
        "lat": 3.1460
    },
    "Tun Razak Exchange": {
        "line": "Kajang Line",
        "lon": 101.7185,
        "lat": 3.1424
    },
    "Maluri": {
        "line": "Kajang Line",
        "lon": 101.7275,
        "lat": 3.1233
    },
    "KL Sentral": {
        "line": "Kelana Jaya Line",
        "lon": 101.6869,
        "lat": 3.1340
    },
    "Bank Rakyat Bangsar": {
        "line": "Kelana Jaya Line",
        "lon": 101.6788,
        "lat": 3.1278
    },
    "Wangsa Maju": {
        "line": "Kelana Jaya Line",
        "lon": 101.7376,
        "lat": 3.2058
    },
    "Abdullah Hukum": {
        "line": "Kelana Jaya Line",
        "lon": 101.6735,
        "lat": 3.1186
    },
    "Pasar Seni": {
        "line": "Kelana Jaya Line",
        "lon": 101.6950,
        "lat": 3.1427
    },
    "Masjid Jamek": {
        "line": "Kelana Jaya Line",
        "lon": 101.6967,
        "lat": 3.1495
    },
    "Muzium Negara": {
        "line": "Kajang Line",
        "lon": 101.6866,
        "lat": 3.1384
    },
    "Cochrane": {
        "line": "Kajang Line",
        "lon": 101.7259,
        "lat": 3.1342
    },
    "Bandar Utama": {
        "line": "Kajang Line",
        "lon": 101.6167,
        "lat": 3.1462
    },
    "Bukit Nanas": {
        "line": "Monorail",
        "lon": 101.7048,
        "lat": 3.1566
    }
}


ALIASES = {
    "Wangs Maju": "Wangsa Maju",
    "Bank Rakyat-Bangsar": "Bank Rakyat Bangsar",
    "Tun Razak Exchange MRT": "Tun Razak Exchange",
    "TRX": "Tun Razak Exchange",
    "KL Sentral LRT": "KL Sentral",
    "KL Sentral MRT": "KL Sentral",
    "Bukit Bintang MRT": "Bukit Bintang",
    "Bukit Bintang Monorail": "Bukit Bintang",
    "KLCC LRT": "KLCC"
}


LINE_LOOKUP = {
    "KJ": "Kelana Jaya Line",
    "KG": "Kajang Line",
    "MR": "Monorail",
    "AG": "Ampang/Sri Petaling Line",
    "SP": "Ampang/Sri Petaling Line",
    "PY": "Putrajaya Line",
    "PYL": "Putrajaya Line"
}


def station_label(raw_name):
    """
    Converts values like 'KJ10: KLCC' into 'KLCC'.
    """
    if pd.isna(raw_name):
        return None

    text = str(raw_name).strip()

    if ":" in text:
        text = text.split(":", 1)[1].strip()

    text = " ".join(text.split())

    return ALIASES.get(text, text)


def line_from_raw(raw_name):
    """
    Extracts route code from values like 'KJ10: KLCC'.
    """
    if pd.isna(raw_name):
        return "Unknown"

    text = str(raw_name).strip()

    if ":" in text:
        station_code = text.split(":", 1)[0].strip()
    else:
        station_code = text

    match = re.match(r"([A-Z]+)", station_code)

    if not match:
        return "Unknown"

    code = match.group(1)

    return LINE_LOOKUP.get(code, "Unknown")


def get_lon(station):
    return STATION_INFO.get(station, {}).get("lon", np.nan)


def get_lat(station):
    return STATION_INFO.get(station, {}).get("lat", np.nan)


def get_station_line(station):
    return STATION_INFO.get(station, {}).get("line", "Unknown")


print("Reading RapidKL OD file in chunks...")

summary_parts = []

usecols = ["date", "origin", "destination", "ridership"]

for i, chunk in enumerate(pd.read_csv(INPUT_FILE, usecols=usecols, chunksize=CHUNK_SIZE)):
    print(f"Processing chunk {i + 1}...")

    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk["ridership"] = pd.to_numeric(chunk["ridership"], errors="coerce").fillna(0)

    chunk = chunk[
        (chunk["date"] >= "2026-01-01") &
        (chunk["date"] < "2026-06-01") &
        (chunk["ridership"] > 0) &
        (chunk["origin"] != "A0: All Stations") &
        (chunk["destination"] != "A0: All Stations") &
        (chunk["origin"] != chunk["destination"])
    ].copy()

    if chunk.empty:
        continue

    chunk["origin_label"] = chunk["origin"].apply(station_label)
    chunk["destination_label"] = chunk["destination"].apply(station_label)

    chunk["origin_line"] = chunk["origin"].apply(line_from_raw)
    chunk["destination_line"] = chunk["destination"].apply(line_from_raw)

    chunk = chunk[
        chunk["origin_label"].notna() &
        chunk["destination_label"].notna()
    ].copy()

    # Merge reverse directions into one corridor using vectorised operations.
    chunk["station_a"] = np.where(
        chunk["origin_label"] <= chunk["destination_label"],
        chunk["origin_label"],
        chunk["destination_label"]
    )

    chunk["station_b"] = np.where(
        chunk["origin_label"] <= chunk["destination_label"],
        chunk["destination_label"],
        chunk["origin_label"]
    )

    chunk["route_group"] = np.where(
        chunk["origin_line"] == chunk["destination_line"],
        chunk["origin_line"],
        "Inter-line"
    )

    grouped = chunk.groupby(
        ["station_a", "station_b", "route_group"],
        as_index=False
    )["ridership"].sum()

    summary_parts.append(grouped)


if not summary_parts:
    raise ValueError("No valid flow rows were produced. Check the raw RapidKL file.")

print("Combining chunk summaries...")

corridors = pd.concat(summary_parts, ignore_index=True)

corridors = corridors.groupby(
    ["station_a", "station_b", "route_group"],
    as_index=False
)["ridership"].sum()

corridors = corridors.sort_values("ridership", ascending=False).head(TOP_CORRIDORS).copy()

corridors["ridership_k"] = corridors["ridership"] / 1000
corridors["corridor"] = corridors["station_a"] + " ↔ " + corridors["station_b"]

corridors["origin_lon"] = corridors["station_a"].apply(get_lon)
corridors["origin_lat"] = corridors["station_a"].apply(get_lat)
corridors["dest_lon"] = corridors["station_b"].apply(get_lon)
corridors["dest_lat"] = corridors["station_b"].apply(get_lat)

missing_stations = sorted(
    set(corridors.loc[corridors["origin_lon"].isna(), "station_a"]).union(
        set(corridors.loc[corridors["dest_lon"].isna(), "station_b"])
    )
)

if missing_stations:
    pd.DataFrame({
        "station_label": missing_stations,
        "longitude": "",
        "latitude": ""
    }).to_csv(OUTPUT_MISSING, index=False)

    print()
    print("Missing coordinates for:")
    for station in missing_stations:
        print(f"- {station}")

    print()
    print(f"Created missing-coordinate template: {OUTPUT_MISSING}")
    print("Add these stations to STATION_INFO, then rerun.")
    raise SystemExit


corridors["mid_lon"] = (corridors["origin_lon"] + corridors["dest_lon"]) / 2
corridors["mid_lat"] = (corridors["origin_lat"] + corridors["dest_lat"]) / 2

corridors.to_csv(OUTPUT_LINES, index=False)


# =========================
# Station node file
# =========================
stations_used = sorted(
    set(corridors["station_a"]).union(set(corridors["station_b"]))
)

nodes = pd.DataFrame({
    "station": stations_used
})

nodes["station_line"] = nodes["station"].apply(get_station_line)
nodes["lon"] = nodes["station"].apply(get_lon)
nodes["lat"] = nodes["station"].apply(get_lat)

nodes.to_csv(OUTPUT_NODES, index=False)


# =========================
# Annotation label file
# =========================
labels = pd.DataFrame([
    {
        "lon": 101.681,
        "lat": 3.193,
        "label": "Top flows concentrate around central KL interchanges"
    },
    {
        "lon": 101.727,
        "lat": 3.176,
        "label": "Line thickness highlights repeated high-demand corridors"
    }
])

labels.to_csv(OUTPUT_LABELS, index=False)


print()
print("Done.")
print(f"Created: {OUTPUT_LINES}")
print(f"Created: {OUTPUT_NODES}")
print(f"Created: {OUTPUT_LABELS}")

print()
print("Top corridors:")
print(corridors[["corridor", "route_group", "ridership_k"]])