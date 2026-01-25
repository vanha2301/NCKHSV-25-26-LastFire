import pandas as pd

csv_path = "NASA/DL_FIRE_SV-C2_684272/fire_archive_SV-C2_684272.csv"  # đổi path nếu cần

df = pd.read_csv(csv_path, dtype=str)  # đọc dạng chuỗi để giữ nguyên format
out = df[["longitude", "latitude", "acq_date"]].copy()

print(out['acq_date'].iloc[2])