import os, time
import requests
import pandas as pd
from pathlib import Path

# ======================
# CONFIG
# ======================
API_KEY = os.getenv("VC_API_KEY", "Y4NHLNSCH7YTG8ZMDRWAVGJFF")
START = "2025-09-01"
END   = "2026-01-31"

NODES_CSV = "nodes.csv"  # bạn tạo file này
OUT_PATH = Path("weather_dec2026_all_nodes.csv")
params = {
    "include": "days",
    "unitGroup": "metric",
    "lang": "vi",
    "timezone": "Asia/Ho_Chi_Minh",
    "key": API_KEY,
}

session = requests.Session()

def fetch_range(lat, lon, start, end):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start}/{end}"
    r = session.get(url, params=params, timeout=35)
    r.raise_for_status()
    return r.json()

# ======================
# LOAD nodes.csv
# ======================
nodes_df = pd.read_csv(NODES_CSV)
need_cols = {"node","lat","lon"}
if not need_cols.issubset(nodes_df.columns):
    raise ValueError(f"nodes.csv must have columns: {need_cols}, but got {list(nodes_df.columns)}")

rows_all = []

print("Nodes:", len(nodes_df), "| Range:", START, "->", END)

for i, row in nodes_df.iterrows():
    node = str(row["node"])
    lat  = float(row["lat"])
    lon  = float(row["lon"])

    print(f"[{i+1}/{len(nodes_df)}] Fetching node={node} ({lat},{lon}) ...")

    data = fetch_range(lat, lon, START, END)

    meta_common = {
        "resolvedAddress": data.get("resolvedAddress"),
        "tz": data.get("timezone"),
        "tzOffset": data.get("tzoffset"),
        "queryCost": data.get("queryCost"),
    }

    for d in data.get("days", []):
        rec = dict(d)
        rec.pop("hours", None)

        # list -> string
        if isinstance(rec.get("preciptype"), list):
            rec["preciptype"] = ",".join(rec["preciptype"])
        if isinstance(rec.get("stations"), list):
            rec["stations"] = ",".join(rec["stations"])

        rec.update(meta_common)
        rec.update({
            "node": node,
            "lat": lat,
            "lon": lon,
            "date": rec.get("datetime"),   # ✅ ngày thật của record
        })

        rows_all.append(rec)

    time.sleep(0.25)  # nghỉ nhẹ tránh rate limit

df_out = pd.DataFrame(rows_all)

# chuẩn hoá date
df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")

# lọc đúng range (phòng API trả thừa)
df_out = df_out[(df_out["date"] >= START) & (df_out["date"] <= END)].copy()

# sort đẹp
df_out = df_out.sort_values(["node","date"]).reset_index(drop=True)

df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Saved:", OUT_PATH, "| rows:", len(df_out))
print("Columns:", len(df_out.columns))
df_out.head(3)
