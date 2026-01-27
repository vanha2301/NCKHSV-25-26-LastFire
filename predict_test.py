# =========================================================
# LastFire / TimeGNN (GCN+GRU) - Robust Inference Script
# - Auto BASE_DAY = max date in CSV (tránh chọn nhầm 2025-09-30)
# - Check missing columns vs feature_cols.npy
# - Report missing rate per node
# - Warn if Xz ~ 0 (input phẳng -> dễ ra ~27% hoài)
# =========================================================

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

# Nếu bạn muốn set tay:
BASE_DAY = pd.Timestamp("2026-01-30")
# BASE_DAY = None

L  = 30     # history window (giống lúc train)
TH = 0.53   # threshold cảnh báo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# nếu node missing quá nhiều (vd > 0.5 = 50% ngày bị thiếu) thì vẫn dự đoán nhưng sẽ warn
WARN_MISSING_RATE = 0.50

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df["node"] = df["node"].astype(str)
    df = df.sort_values(["node", "date"]).reset_index(drop=True)

    # doy sin/cos
    doy = df["date"].dt.dayofyear.values.astype(np.float32)
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype(np.float32)

    # rolling helper (giống lúc train)
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
# MODEL (giống train)
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
        # x_seq: [B,L,N,F]
        B, L, N, F = x_seq.shape
        A = self.A

        hs = []
        for t in range(L):
            xt = x_seq[:, t, :, :]                          # [B,N,F]
            xmix = torch.einsum("ij,bjf->bif", A, xt)        # [B,N,F]
            ht = self.drop(self.act(self.gcn(xmix)))         # [B,N,gcn_dim]
            hs.append(ht)

        h = torch.stack(hs, dim=1)                           # [B,L,N,gcn_dim]

        logits = []
        for i in range(N):
            hi = h[:, :, i, :]                               # [B,L,gcn_dim]
            _, hn = self.gru(hi)                             # [1,B,gru_dim]
            last = hn[-1]                                    # [B,gru_dim]
            logit = self.head(self.drop(last)).squeeze(-1)   # [B]
            logits.append(logit)

        return torch.stack(logits, dim=1)                    # [B,N]


def safe_display(df, n=20):
    """In notebook thì display đẹp; chạy script thì print."""
    try:
        from IPython.display import display
        display(df.head(n))
    except Exception:
        print(df.head(n).to_string(index=False))


# ======================
# LOAD ARTIFACTS
# ======================
A_norm = np.load(A_PATH).astype(np.float32)
mu = np.load(MU_PATH).astype(np.float32)  # (1,F)
sd = np.load(SD_PATH).astype(np.float32)  # (1,F)

nodes = np.load(NODES_PATH, allow_pickle=True).tolist()
feature_cols = np.load(FEATS_PATH, allow_pickle=True).tolist()

# ép kiểu node về str cho chắc lọc đúng
nodes = [str(x) for x in nodes]

F = len(feature_cols)
N = len(nodes)

if mu.shape[-1] != F or sd.shape[-1] != F:
    raise ValueError(f"mu/sd F mismatch: mu={mu.shape}, sd={sd.shape}, feature_cols={F}")

if A_norm.shape[0] != N or A_norm.shape[1] != N:
    raise ValueError(f"A_norm shape {A_norm.shape} != (N,N)=({N},{N}). Check nodes order!")

ckpt = torch.load(MODEL_PATH, map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

gcn_dim = state["gcn.weight"].shape[0]
gru_dim = state["head.weight"].shape[1]

model = TimeGCN_GRU(A_norm, in_dim=F, gcn_dim=gcn_dim, gru_dim=gru_dim, dropout=0.0).to(DEVICE)
model.load_state_dict(state, strict=True)
model.eval()

print(f"Loaded model OK | N={N} F={F} gcn_dim={gcn_dim} gru_dim={gru_dim} DEVICE={DEVICE}")

# ======================
# READ CSV
# ======================
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")

if "date" not in df.columns or "node" not in df.columns:
    raise ValueError("CSV must contain columns: node, date")

df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
df = df.dropna(subset=["date"]).copy()

df["node"] = df["node"].astype(str)

df = add_engineered_features(df)
min_date = df["date"].min()
max_date = df["date"].max()
print("CSV date range:", min_date.date(), "->", max_date.date())

if BASE_DAY is None:
    BASE_DAY = max_date
else:
    BASE_DAY = pd.Timestamp(BASE_DAY).floor("D")

if BASE_DAY > max_date:
    raise ValueError(f"BASE_DAY={BASE_DAY.date()} nhưng CSV chỉ có tới {max_date.date()}")

start = BASE_DAY - pd.Timedelta(days=L-1)
full_dates = pd.date_range(start, BASE_DAY, freq="D")
print("Using BASE_DAY:", BASE_DAY.date(), "| window:", full_dates[0].date(), "->", full_dates[-1].date(), f"(L={L})")

# lat/lon (nếu có)
lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)

node_latlon = {n: (np.nan, np.nan) for n in nodes}
if lat_col and lon_col:
    tmp = df.groupby("node")[[lat_col, lon_col]].mean(numeric_only=True).reset_index()
    for _, r in tmp.iterrows():
        node_latlon[str(r["node"])] = (float(r[lat_col]), float(r[lon_col]))

# ======================
# FEATURE CHECK
# ======================
base_weather_cols = [c for c in feature_cols if c != "weather_missing"]

missing_cols = [c for c in base_weather_cols if c not in df.columns]
if missing_cols:
    print(f"⚠️ Missing {len(missing_cols)} feature columns in CSV (will be filled by mu):")
    print("   ", missing_cols[:30], ("..." if len(missing_cols) > 30 else ""))

# ======================
# BUILD INPUT WINDOW [L,N,F]
# ======================
X_list = []
node_missing_rate = {}

for n in nodes:
    # lấy data của node tới BASE_DAY
    g = df[(df["node"] == n) & (df["date"] <= BASE_DAY)].copy()
    if g.empty:
        raise ValueError(f"Thiếu data node={n} trong CSV (không có dòng nào <= BASE_DAY).")

    # nếu có nhiều dòng cùng ngày -> lấy mean numeric
    present_cols = [c for c in base_weather_cols if c in g.columns]
    # tránh case present_cols rỗng (CSV không có feature nào)
    if len(present_cols) == 0:
        # vẫn chạy được nhưng sẽ fill toàn mu -> output gần hằng
        present_cols = []

    if present_cols:
        g = g.groupby("date", as_index=False)[present_cols].mean(numeric_only=True)
    else:
        g = g[["date"]].drop_duplicates()

    g = g.set_index("date").reindex(full_dates)

    # tạo đủ cột theo feature_cols
    for c in base_weather_cols:
        if c not in g.columns:
            g[c] = np.nan

    # missing flag theo ngày (trước khi fill)
    miss_flag = g[base_weather_cols].isna().any(axis=1).astype(np.float32).to_numpy()
    miss_rate = float(miss_flag.mean())
    node_missing_rate[n] = miss_rate

    # fill forward (causal)
    g[base_weather_cols] = g[base_weather_cols].ffill()

    # fill còn lại bằng mu (đầu chuỗi / cột thiếu hoàn toàn)
    for c in base_weather_cols:
        if g[c].isna().any():
            idx = feature_cols.index(c)
            g[c] = g[c].fillna(float(mu[0, idx]))

    if "weather_missing" in feature_cols:
        g["weather_missing"] = miss_flag

    Xn = g[feature_cols].to_numpy(dtype=np.float32)  # (L,F)
    if Xn.shape != (L, F):
        raise ValueError(f"Node {n} got shape {Xn.shape}, expected {(L,F)}")
    X_list.append(Xn)

X_raw = np.stack(X_list, axis=1)  # (L,N,F)
Xz = (X_raw - mu.reshape(1,1,-1)) / (sd.reshape(1,1,-1) + 1e-6)

# ======================
# DEBUG: detect "flat input"
# ======================
xz_std = float(np.std(Xz))
zero_like = float(np.mean(np.abs(Xz) < 1e-3))
print(f"Xz std={xz_std:.6f} | Xz~0 ratio={zero_like:.3f}")

if xz_std < 0.05 or zero_like > 0.70:
    print("⚠️ WARNING: Input looks very 'flat' (near zero after standardize).")
    print("   => Model may output near-constant probs (vd ~27%).")
    print("   Check: BASE_DAY đúng chưa? CSV có thiếu nhiều feature_cols không? missing rate per node bên dưới.")

# top nodes missing nhiều
miss_df = pd.DataFrame({"node": list(node_missing_rate.keys()), "missing_rate": list(node_missing_rate.values())})
miss_df = miss_df.sort_values("missing_rate", ascending=False).reset_index(drop=True)
print("\nTop 10 nodes with highest missing_rate:")
safe_display(miss_df, n=10)

high_miss = miss_df[miss_df["missing_rate"] >= WARN_MISSING_RATE]
if len(high_miss) > 0:
    print(f"\n⚠️ {len(high_miss)} nodes have missing_rate >= {WARN_MISSING_RATE:.2f}. Predictions may be unreliable for them.")

# ======================
# PREDICT
# ======================
x_seq = torch.tensor(Xz, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,L,N,F]
with torch.no_grad():
    prob = torch.sigmoid(model(x_seq)).detach().cpu().numpy()[0]  # [N]

alert = (prob >= TH).astype(int)

def risk_level(p: float) -> str:
    # bạn có thể chỉnh ngưỡng theo ý
    if p >= 0.75:
        return "HIGH"
    if p >= TH:
        return "WARNING"
    if p >= 0.30:
        return "WATCH"
    return "LOW"

out = pd.DataFrame({
    "node": nodes,
    "lat": [node_latlon[n][0] for n in nodes],
    "lon": [node_latlon[n][1] for n in nodes],
    "prob_fire_next7": prob,
    "prob_%": prob * 100.0,
    "risk": [risk_level(float(p)) for p in prob],
    "alert_next7": alert,
    "missing_rate_Ldays": [node_missing_rate[n] for n in nodes],
}).sort_values("prob_fire_next7", ascending=False).reset_index(drop=True)

print(f"\n=== PREDICT next7 from base_day={BASE_DAY.date()} | TH={TH:.2f} ===")
safe_display(out, n=30)

# Save
save_path = f"pred_next7_{BASE_DAY.date()}.csv"
out.to_csv(save_path, index=False)
print("\nSaved:", save_path)

if out["alert_next7"].any():
    print("\n⚠️  CẢNH BÁO: Có node vượt threshold nguy cơ cháy trong 7 ngày tới.")
else:
    print("\n✅  OK: Không node nào vượt threshold trong 7 ngày tới.")
