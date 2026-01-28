# predict.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from zoneinfo import ZoneInfo  
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
# ======================
# CONFIG
# ======================
CSV_PATH   = r"D:\NCKHSV-25-26-LastFire\weather_dec2025_all_nodes.csv"
MODEL_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\best_timegnn.pt"
A_PATH     = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\A_norm.npy"
MU_PATH    = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\mu.npy"
SD_PATH    = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\sd.npy"
NODES_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\nodes.npy"
FEATS_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\feature_cols.npy"

def get_base_day_vn() -> pd.Timestamp:
    """
    BASE_DAY = ngày hiện tại theo giờ VN, làm tròn về 00:00 (date floor).
    Forecast window sẽ là BASE_DAY+1 -> BASE_DAY+7.
    """
    now_vn = pd.Timestamp.now(tz=VN_TZ)
    base = now_vn.floor("D").tz_localize(None)  # bỏ timezone để khớp với df date naive
    return base

BASE_DAY = get_base_day_vn()
# BASE_DAY = pd.Timestamp("2025-09-30")  # chỉnh thủ công cho mục đích test

L  = 30
TH = 0.53
WARN_MISSING_RATE = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# FEATURES ENGINEERING
# ======================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df["node"] = df["node"].astype(str)
    df = df.sort_values(["node", "date"]).reset_index(drop=True)

    doy = df["date"].dt.dayofyear.values.astype(np.float32)
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype(np.float32)

    def roll(g, w, how="mean"):
        r = g.rolling(w, min_periods=1)
        if how == "mean": return r.mean()
        if how == "sum":  return r.sum()
        if how == "min":  return r.min()
        if how == "max":  return r.max()
        raise ValueError(how)

    base = [c for c in ["temp","humidity","precip","windspeed","cloudcover","uvindex"] if c in df.columns]
    for c in base:
        s = df.groupby("node")[c]
        df[f"{c}_mean7"]  = roll(s, 7,  "mean").reset_index(level=0, drop=True).astype(np.float32)
        df[f"{c}_mean14"] = roll(s, 14, "mean").reset_index(level=0, drop=True).astype(np.float32)
        df[f"{c}_mean30"] = roll(s, 30, "mean").reset_index(level=0, drop=True).astype(np.float32)

    if "precip" in df.columns:
        s = df.groupby("node")["precip"]
        df["precip_sum7"]  = roll(s, 7,  "sum").reset_index(level=0, drop=True).astype(np.float32)
        df["precip_sum30"] = roll(s, 30, "sum").reset_index(level=0, drop=True).astype(np.float32)

    return df


# ======================
# MODEL
# ======================
class TimeGCN_GRU(nn.Module):
    def __init__(self, A_norm, in_dim, gcn_dim=32, gru_dim=64, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))  # [N,N]
        self.gcn = nn.Linear(in_dim, gcn_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=gcn_dim, hidden_size=gru_dim, batch_first=True)
        self.head = nn.Linear(gru_dim, 1)

    def forward(self, x_seq):
        B, Lx, N, F = x_seq.shape
        A = self.A

        hs = []
        for t in range(Lx):
            xt = x_seq[:, t, :, :]
            xmix = torch.einsum("ij,bjf->bif", A, xt)
            ht = self.drop(self.act(self.gcn(xmix)))
            hs.append(ht)

        h = torch.stack(hs, dim=1)  # [B,L,N,gcn_dim]

        logits = []
        for i in range(N):
            hi = h[:, :, i, :]
            _, hn = self.gru(hi)
            last = hn[-1]
            logit = self.head(self.drop(last)).squeeze(-1)
            logits.append(logit)

        return torch.stack(logits, dim=1)  # [B,N]


def risk_level(p: float) -> str:
    if p >= 0.75: return "HIGH"
    if p >= TH:   return "WARNING"
    if p >= 0.30: return "WATCH"
    return "LOW"

def advice_text(risk: str) -> str:
    # ✅ index.html của bạn có cột advice, nên thêm cho đẹp
    if risk == "HIGH":    return "High Alert (Priority Inspection)"
    if risk == "WARNING": return "High Alert (Prepare Plan)"
    if risk == "WATCH":   return "Monitor Closely (Early Warning)"
    return "Normal"


# ======================
# MAIN ENTRY FOR FASTAPI
# ======================
def run_prediction():
    # 1) load artifacts
    A_norm = np.load(A_PATH).astype(np.float32)
    mu = np.load(MU_PATH).astype(np.float32).reshape(1, -1)
    sd = np.load(SD_PATH).astype(np.float32).reshape(1, -1)
    nodes = [str(x) for x in np.load(NODES_PATH, allow_pickle=True).tolist()]
    feature_cols = np.load(FEATS_PATH, allow_pickle=True).tolist()

    F = len(feature_cols)
    N = len(nodes)

    if mu.shape[-1] != F or sd.shape[-1] != F:
        raise ValueError(f"mu/sd mismatch: mu={mu.shape}, sd={sd.shape}, F={F}")

    if A_norm.shape != (N, N):
        raise ValueError(f"A_norm shape {A_norm.shape} != ({N},{N})")

    # 2) load model
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    gcn_dim = state["gcn.weight"].shape[0]
    gru_dim = state["head.weight"].shape[1]

    model = TimeGCN_GRU(A_norm, in_dim=F, gcn_dim=gcn_dim, gru_dim=gru_dim, dropout=0.0).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) read csv
    df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    if "date" not in df.columns or "node" not in df.columns:
        raise ValueError("CSV must contain columns: node, date")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).copy()
    df["node"] = df["node"].astype(str)

    df = add_engineered_features(df)
    min_date = df["date"].min()
    max_date = df["date"].max()

    base_day = max_date if BASE_DAY is None else pd.Timestamp(BASE_DAY).floor("D")
    if base_day > max_date:
        raise ValueError(f"BASE_DAY={base_day.date()} nhưng CSV chỉ có tới {max_date.date()}")

    start = base_day - pd.Timedelta(days=L-1)
    full_dates = pd.date_range(start, base_day, freq="D")

    # lat/lon map
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    node_latlon = {n: (np.nan, np.nan) for n in nodes}
    if lat_col and lon_col:
        tmp = df.groupby("node")[[lat_col, lon_col]].mean(numeric_only=True).reset_index()
        for _, r in tmp.iterrows():
            node_latlon[str(r["node"])] = (float(r[lat_col]), float(r[lon_col]))

    # 4) feature check + build X
    base_weather_cols = [c for c in feature_cols if c != "weather_missing"]
    missing_cols = [c for c in base_weather_cols if c not in df.columns]

    X_list = []
    node_missing_rate = {}

    for n in nodes:
        g = df[(df["node"] == n) & (df["date"] <= base_day)].copy()
        if g.empty:
            raise ValueError(f"Thiếu data node={n} trong CSV.")

        present_cols = [c for c in base_weather_cols if c in g.columns]
        if present_cols:
            g = g.groupby("date", as_index=False)[present_cols].mean(numeric_only=True)
        else:
            g = g[["date"]].drop_duplicates()

        g = g.set_index("date").reindex(full_dates)

        for c in base_weather_cols:
            if c not in g.columns:
                g[c] = np.nan

        miss_flag = g[base_weather_cols].isna().any(axis=1).astype(np.float32).to_numpy()
        node_missing_rate[n] = float(miss_flag.mean())

        g[base_weather_cols] = g[base_weather_cols].ffill()

        for c in base_weather_cols:
            if g[c].isna().any():
                idx = feature_cols.index(c)
                g[c] = g[c].fillna(float(mu[0, idx]))

        if "weather_missing" in feature_cols:
            g["weather_missing"] = miss_flag

        X_list.append(g[feature_cols].to_numpy(dtype=np.float32))

    X_raw = np.stack(X_list, axis=1)  # (L,N,F)
    Xz = (X_raw - mu.reshape(1,1,-1)) / (sd.reshape(1,1,-1) + 1e-6)

    xz_std = float(np.std(Xz))
    zero_like = float(np.mean(np.abs(Xz) < 1e-3))

    # 5) predict
    x_seq = torch.tensor(Xz, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,L,N,F]
    with torch.no_grad():
        prob = torch.sigmoid(model(x_seq)).detach().cpu().numpy()[0]  # [N]

    alert = (prob >= TH).astype(int)

    # 6) meta + out2
    meta = {
        "base_day": str(base_day.date()),
        "history_days": int(L),
        "th_alert": float(TH),
        "forecast_from": str((base_day + pd.Timedelta(days=1)).date()),
        "forecast_to": str((base_day + pd.Timedelta(days=7)).date()),
        "csv_min_date": str(min_date.date()),
        "csv_max_date": str(max_date.date()),
        "window_from": str(full_dates[0].date()),
        "window_to": str(full_dates[-1].date()),
        "device": DEVICE,
        "N_nodes": int(N),
        "F_feats": int(F),
        "xz_std": xz_std,
        "xz_zero_like_ratio": zero_like,
        "missing_feature_cols_count": int(len(missing_cols)),
        "missing_feature_cols": missing_cols[:200],
        "warn_missing_rate_threshold": float(WARN_MISSING_RATE),
    }

    risks = [risk_level(float(p)) for p in prob]

    out2 = pd.DataFrame({
        "node": nodes,
        "lat": [node_latlon[n][0] for n in nodes],
        "lon": [node_latlon[n][1] for n in nodes],
        "prob_fire_next7": prob,
        "prob_%": prob * 100.0,
        "risk": risks,
        "alert_next7": alert,
        "missing_rate_Ldays": [node_missing_rate[n] for n in nodes],
        "advice": [advice_text(r) for r in risks],  # ✅ để khớp index.html
    }).sort_values("prob_fire_next7", ascending=False).reset_index(drop=True)

    return meta, out2


# optional: chạy test bằng python predict.py
if __name__ == "__main__":
    meta, out2 = run_prediction()
    print(meta)
    print(out2.head(10).to_string(index=False))
