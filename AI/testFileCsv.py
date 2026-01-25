import pandas as pd

rows = [
    {"longitude": 108.25682, "latitude": 12.42611, "date": "2012-02-08"},
    {"longitude": 108.25694, "latitude": 12.42818, "date": "2012-02-08"},
]

df = pd.DataFrame(rows)
df.to_csv("lon_lat_date.csv", index=False, encoding="utf-8-sig")
