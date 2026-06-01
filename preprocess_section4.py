import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

FARE_FILE = DATA_DIR / "Fare_rapidkl.csv"
TIME_FILE = DATA_DIR / "Time_rapidkl.csv"
ROUTE_FILE = DATA_DIR / "Route_rapidkl.csv"

OUTPUT_FILE = DATA_DIR / "fare_time_route_long.csv"


def read_csv_safe(path, dtype=None):
    return pd.read_csv(
        path,
        dtype=dtype,
        engine="python",
        encoding="utf-8-sig"
    )


def clean_station_name(name):
    return str(name).strip()


def count_transfers(route_text):
    route_text = str(route_text)

    if route_text.lower() == "nan" or route_text.strip() == "":
        return None

    return route_text.count(">>")


def transfer_group(transfer_count):
    if transfer_count == 0:
        return "Direct journey"
    else:
        return "1+ transfer"


print("Reading Fare, Time and Route matrices...")

fare = read_csv_safe(FARE_FILE)
time = read_csv_safe(TIME_FILE)
route = read_csv_safe(ROUTE_FILE, dtype=str)

print("Files loaded successfully.")
print(f"Fare shape: {fare.shape}")
print(f"Time shape: {time.shape}")
print(f"Route shape: {route.shape}")

origin_col = fare.columns[0]
stations = list(fare.columns[1:])

records = []

print("Reshaping matrices into long format...")

for row_index, origin in enumerate(fare[origin_col]):
    origin = clean_station_name(origin)

    for col_index, destination in enumerate(stations):
        destination = clean_station_name(destination)

        # Keep only one direction for each station pair.
        if row_index >= col_index:
            continue

        fare_value = fare.iloc[row_index, col_index + 1]
        time_value = time.iloc[row_index, col_index + 1]
        route_value = route.iloc[row_index, col_index + 1]

        if pd.isna(fare_value) or pd.isna(time_value):
            continue

        if origin == destination:
            continue

        try:
            fare_float = float(fare_value)
            time_float = float(time_value)
        except ValueError:
            continue

        transfers = count_transfers(route_value)

        if transfers is None:
            continue

        records.append({
            "origin": origin,
            "destination": destination,
            "od_pair": f"{origin} → {destination}",
            "fare_rm": fare_float,
            "time_min": time_float,
            "route": str(route_value),
            "transfer_count": transfers,
            "transfer_group": transfer_group(transfers)
        })

df = pd.DataFrame(records)

df = df[
    (df["fare_rm"] > 0) &
    (df["time_min"] > 0)
].copy()

df["fare_label"] = "RM " + df["fare_rm"].map(lambda x: f"{x:.2f}")
df["time_label"] = df["time_min"].map(lambda x: f"{x:.0f} min")

df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Created: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print()
print("Journey type counts:")
print(df["transfer_group"].value_counts())