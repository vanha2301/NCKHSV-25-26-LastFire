import requests
import pandas as pd
from pathlib import Path

API_KEY = "7FS98E97WQXQMNHGK9FNBTB9Z"
# lat, lon = 12.42611, 108.25682
# day = "2012-02-09"

# url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{day}"
params = {
    "unitGroup": "metric",
    "include": "current,days,hours,alerts,obs",   # lấy tối đa các khối dữ liệu
    "lang": "en",                                 # EN để 'conditions' dễ đọc (đổi 'vi' nếu muốn)
    "timezone": "Asia/Ho_Chi_Minh",
    "key": API_KEY,
    # Không đặt "elements" để KHÔNG giới hạn field
}
_ = 0
while True:
    df = pd.read_csv("NASA/DL_FIRE_SV-C2_684272/fire_archive_SV-C2_684272.csv", dtype=str)  # đọc dạng chuỗi để giữ nguyên format
    out = df[["longitude", "latitude", "acq_date"]].copy()
    lon = out['longitude'].iloc[_]
    lat = out['latitude'].iloc[_]
    day = out['acq_date'].iloc[_]
    r = requests.get( f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{day}", params=params, timeout=25)
    r.raise_for_status()
    data = r.json()

    rows = []

    # Một số meta cấp response (nếu muốn lưu để trace)
    meta_common = {
        "resolvedAddress": data.get("resolvedAddress"),
        "tz": data.get("timezone"),
        "tzOffset": data.get("tzoffset"),
    }

    for d in data.get("days", []):
        # gom field cấp ngày (đổi tên với tiền tố day_)
        day_blob = {}
        for k, v in d.items():
            if k == "hours": 
                continue
            day_blob[f"day_{k}"] = v

        # lặp từng giờ và merge mọi key có mặt trong 'hour'
        for h in d.get("hours", []):
            # Bản sao keys giờ
            rec = dict(h)

            # Chuẩn hóa vài trường danh sách/phức tạp về string cho CSV
            if isinstance(rec.get("preciptype"), list):
                rec["preciptype"] = ",".join(rec["preciptype"])
            if isinstance(rec.get("stations"), list):
                rec["stations"] = ",".join(rec["stations"])

            # Bổ sung khóa thời gian cấp ngày + meta + tọa độ nguồn nếu cần
            rec.update({
                "date": d.get("datetime"),          # ngày cha
                "latitude_src": lat,
                "longitude_src": lon,
            })
            rec.update(day_blob)
            rec.update(meta_common)

            rows.append(rec)

    # Đổ DataFrame
    df = pd.DataFrame(rows)

    # (Tuỳ chọn) Sắp xếp cột: đưa vài cột quan trọng lên đầu
    front_cols = [c for c in [
        "date", "datetime", "datetimeEpoch", 
        "temp", "feelslike", "dew", "humidity",
        "precip", "precipprob", "preciptype",
        "snow", "snowdepth",
        "windspeed", "windgust", "winddir",
        "pressure", "visibility", "cloudcover",
        "uvindex", "solarradiation", "solarenergy",
        "conditions", "icon", "source", "stations",
        "day_tempmin", "day_tempmax", "day_sunrise", "day_sunset",
        "resolvedAddress", "tz", "tzOffset",
        "latitude_src", "longitude_src",
    ] if c in df.columns]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]

    out_path = Path("weather_hours_everything.csv")
    df.to_csv(out_path,mode="a",header=True, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {out_path.resolve()}")
    print(df.head(3).T)  # xem nhanh các cột/giá trị
    _ += 1
    if _ >= 5:
        break
