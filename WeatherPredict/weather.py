"""
weather.py
----------
Fetch daily weather data from the Visual Crossing API for a set of spatial
nodes and save the results to a single combined CSV file.

Usage:
    Set the VC_API_KEY environment variable (or hard-code for development),
    ensure nodes.csv exists in the same directory, then run:

        python weather.py

Output:
    weather_all_nodes.csv — Merged daily records for all nodes.
"""

import os
import time

import pandas as pd
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY    = os.getenv("VC_API_KEY", "Y4NHLNSCH7YTG8ZMDRWAVGJFF")
START      = "2025-09-01"
END        = "2026-01-31"
NODES_CSV  = Path(__file__).parent / "nodes.csv"
OUT_PATH   = Path(__file__).parent / "weather_all_nodes.csv"

# Seconds to sleep between API calls (avoids rate-limiting)
REQUEST_DELAY = 0.25

API_PARAMS = {
    "include":   "days",
    "unitGroup": "metric",
    "lang":      "vi",
    "timezone":  "Asia/Ho_Chi_Minh",
    "key":       API_KEY,
}

BASE_URL = (
    "https://weather.visualcrossing.com"
    "/VisualCrossingWebServices/rest/services/timeline"
)

# ---------------------------------------------------------------------------
# HTTP session (connection pooling)
# ---------------------------------------------------------------------------
session = requests.Session()


def fetch_range(lat: float, lon: float, start: str, end: str) -> dict:
    """
    Fetch daily weather records for a lat/lon point over a date range.

    Args:
        lat:   Latitude of the node.
        lon:   Longitude of the node.
        start: Start date string, format 'YYYY-MM-DD'.
        end:   End date string,   format 'YYYY-MM-DD'.

    Returns:
        Parsed JSON response dict from the Visual Crossing API.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    url = f"{BASE_URL}/{lat},{lon}/{start}/{end}"
    response = session.get(url, params=API_PARAMS, timeout=35)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    nodes_df = pd.read_csv(NODES_CSV)
    required_cols = {"node", "lat", "lon"}
    if not required_cols.issubset(nodes_df.columns):
        raise ValueError(
            f"nodes.csv must contain columns {required_cols}, "
            f"got: {list(nodes_df.columns)}"
        )

    print(f"Nodes: {len(nodes_df)} | Range: {START} → {END}")

    rows_all = []

    for i, row in nodes_df.iterrows():
        node = str(row["node"])
        lat  = float(row["lat"])
        lon  = float(row["lon"])

        print(f"  [{i + 1}/{len(nodes_df)}] node={node} ({lat}, {lon}) …", end=" ", flush=True)

        data = fetch_range(lat, lon, START, END)

        common_meta = {
            "resolvedAddress": data.get("resolvedAddress"),
            "tz":              data.get("timezone"),
            "tzOffset":        data.get("tzoffset"),
            "queryCost":       data.get("queryCost"),
        }

        for day in data.get("days", []):
            record = dict(day)
            record.pop("hours", None)

            # Flatten list fields to comma-separated strings
            for field in ("preciptype", "stations"):
                if isinstance(record.get(field), list):
                    record[field] = ",".join(record[field])

            record.update(common_meta)
            record.update({
                "node": node,
                "lat":  lat,
                "lon":  lon,
                "date": record.get("datetime"),
            })
            rows_all.append(record)

        print(f"OK ({len(data.get('days', []))} days)")
        time.sleep(REQUEST_DELAY)

    df_out = pd.DataFrame(rows_all)

    # Normalise date column
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")

    # Drop any rows outside the requested range (API may return extra days)
    df_out = df_out[(df_out["date"] >= START) & (df_out["date"] <= END)].copy()

    df_out = df_out.sort_values(["node", "date"]).reset_index(drop=True)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUT_PATH}  ({len(df_out)} rows, {len(df_out.columns)} columns)")


if __name__ == "__main__":
    main()
