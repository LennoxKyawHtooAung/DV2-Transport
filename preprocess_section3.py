import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "rapidrail_2026_daily.csv"

OUTPUT_MATRIX = DATA_DIR / "rapidkl_od_matrix_top15.csv"
OUTPUT_TOP_FLOWS = DATA_DIR / "rapidkl_top_flows.csv"

TOP_STATIONS = 15
TOP_FLOWS = 20
CHUNK_SIZE = 500000


def station_code(station_name):
    return str(station_name).split(":", 1)[0].strip()


def station_label(station_name):
    station_name = str(station_name)
    if ":" in station_name:
        return station_name.split(":", 1)[1].strip()
    return station_name.strip()


def line_code(station_name):
    code = station_code(station_name)
    match = re.match(r"([A-Z]+)", code)
    return match.group(1) if match else code


def line_name(code):
    lookup = {
        "AG": "Ampang",
        "SP": "Sri Petaling",
        "KJ": "Kelana Jaya",
        "KG": "MRT Kajang",
        "MR": "Monorail",
        "PYL": "MRT Putrajaya",
        "PY": "MRT Putrajaya"
    }
    return lookup.get(code, code)


def clean_chunk(chunk):
    chunk = chunk[
        (chunk["origin"] != "A0: All Stations") &
        (chunk["destination"] != "A0: All Stations") &
        (chunk["origin"] != chunk["destination"])
    ].copy()

    chunk["month"] = chunk["date"].astype(str).str.slice(0, 7)

    chunk["origin_label"] = chunk["origin"].apply(station_label)
    chunk["destination_label"] = chunk["destination"].apply(station_label)

    chunk["origin_line_code"] = chunk["origin"].apply(line_code)
    chunk["destination_line_code"] = chunk["destination"].apply(line_code)

    chunk["origin_line"] = chunk["origin_line_code"].apply(line_name)
    chunk["destination_line"] = chunk["destination_line_code"].apply(line_name)

    chunk["flow_type"] = chunk.apply(
        lambda row: "Same line" if row["origin_line_code"] == row["destination_line_code"] else "Inter-line",
        axis=1
    )

    chunk["od_pair_short"] = chunk["origin_label"] + " → " + chunk["destination_label"]

    return chunk


print("Pass 1: finding top stations...")

station_totals_parts = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk = chunk[
        (chunk["origin"] != "A0: All Stations") &
        (chunk["destination"] != "A0: All Stations") &
        (chunk["origin"] != chunk["destination"])
    ].copy()

    origin_totals = chunk.groupby("origin", as_index=False)["ridership"].sum()
    origin_totals = origin_totals.rename(columns={"origin": "station"})

    destination_totals = chunk.groupby("destination", as_index=False)["ridership"].sum()
    destination_totals = destination_totals.rename(columns={"destination": "station"})

    station_totals_parts.append(origin_totals)
    station_totals_parts.append(destination_totals)

station_totals = pd.concat(station_totals_parts, ignore_index=True)
station_totals = station_totals.groupby("station", as_index=False)["ridership"].sum()
station_totals = station_totals.sort_values("ridership", ascending=False).head(TOP_STATIONS)

top_stations = station_totals["station"].tolist()
station_rank = {station: rank + 1 for rank, station in enumerate(top_stations)}

print("Top stations:")
for station in top_stations:
    print(f"- {station}")

print()
print("Pass 2: building OD matrix and top flow summaries...")

matrix_parts = []
flow_parts = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk = clean_chunk(chunk)

    matrix_chunk = chunk[
        chunk["origin"].isin(top_stations) &
        chunk["destination"].isin(top_stations)
    ].copy()

    if not matrix_chunk.empty:
        matrix_grouped = matrix_chunk.groupby(
            [
                "month",
                "origin",
                "destination",
                "origin_label",
                "destination_label",
                "origin_line_code",
                "destination_line_code",
                "origin_line",
                "destination_line",
                "flow_type"
            ],
            as_index=False
        )["ridership"].sum()

        matrix_parts.append(matrix_grouped)

    flow_grouped = chunk.groupby(
        [
            "origin",
            "destination",
            "origin_label",
            "destination_label",
            "origin_line_code",
            "destination_line_code",
            "origin_line",
            "destination_line",
            "flow_type",
            "od_pair_short"
        ],
        as_index=False
    )["ridership"].sum()

    flow_parts.append(flow_grouped)

print("Finalising OD matrix...")

matrix = pd.concat(matrix_parts, ignore_index=True)
matrix = matrix.groupby(
    [
        "month",
        "origin",
        "destination",
        "origin_label",
        "destination_label",
        "origin_line_code",
        "destination_line_code",
        "origin_line",
        "destination_line",
        "flow_type"
    ],
    as_index=False
)["ridership"].sum()

matrix["origin_rank"] = matrix["origin"].map(station_rank)
matrix["destination_rank"] = matrix["destination"].map(station_rank)
matrix["ridership_k"] = matrix["ridership"] / 1000

matrix.to_csv(OUTPUT_MATRIX, index=False)

print("Finalising top flows...")

flows = pd.concat(flow_parts, ignore_index=True)
flows = flows.groupby(
    [
        "origin",
        "destination",
        "origin_label",
        "destination_label",
        "origin_line_code",
        "destination_line_code",
        "origin_line",
        "destination_line",
        "flow_type",
        "od_pair_short"
    ],
    as_index=False
)["ridership"].sum()

flows = flows.sort_values("ridership", ascending=False).head(TOP_FLOWS).copy()
flows["rank"] = range(1, len(flows) + 1)
flows["ridership_k"] = flows["ridership"] / 1000
flows["ridership_label"] = flows["ridership_k"].round(0).astype(int).astype(str) + "K"

flows.to_csv(OUTPUT_TOP_FLOWS, index=False)

print()
print("Done.")
print(f"Created: {OUTPUT_MATRIX}")
print(f"Created: {OUTPUT_TOP_FLOWS}")
print()
print("Top flow preview:")
print(flows[["rank", "od_pair_short", "ridership_label", "flow_type"]].head(10))