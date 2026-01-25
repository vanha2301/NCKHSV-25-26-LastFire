import requests
import pandas as pd
from pathlib import Path

API_KEY = "MSVUGTQCBSKWMMJRZNJYFBVRY"

lat = 11.900
lon = 108.400

params = {
    "include": "current,days,alerts",
    "unitGroup": "metric",
    "lang": "vi",
    "timezone": "Asia/Ho_Chi_Minh",
    "key": API_KEY,
}

_ = 0
while True:
    df = pd.read_csv("date.csv", dtype=str)
    outDate = df[["acp_date"]].copy()
    day = outDate['acp_date'].iloc[_]

    r = requests.get( f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{day}",
                    params=params,
                    timeout=25
    )

    r.raise_for_status()
    data = r.json()
    rows = []

    meta_common = {
        "resolvedAddress": data.get("resolvedAddress"),
        "tz": data.get("timezone"),
        "tzOffset": data.get("tzoffset"),
        "queryCost": data.get("queryCost"),
    }

    for d in data.get("days", []):

        rec = dict(d)

        rec.pop("hours", None)

        if isinstance(rec.get("preciptype"), list):
            rec["preciptype"] = ",".join(rec["preciptype"])
        
        if isinstance(rec.get("stations"), list):
            rec["stations"] = ",".join(rec["stations"])
        
        rec.update(meta_common)
        rec.update({
            "lat": lat,
            "lon": lon,
            "date": day,
        })

        rows.append(rec)
    
    df_out = pd.DataFrame(rows)
    out_path = Path("weather_data_73.csv")
    if not out_path.exists():
        df_out.to_csv(out_path, index=False, mode='w', encoding='utf-8-sig')
    else:
        df_out.to_csv(out_path, index=False, mode='a', header=False, encoding='utf-8-sig')

    _ += 1
    print(f"Processed date: {day}")
    if _ >= 5021:
        break