"""
predict_v2.py
-------------
Fire risk prediction using a Time-GCN + GRU model (TimeGCN_GRU).

Loads pre-trained model artifacts, prepares a 30-day weather feature
window from historical CSV data, and outputs per-node fire probabilities
and risk levels for the next 7 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# ---------------------------------------------------------------------------
# Config — file paths & hyper-parameters
# ---------------------------------------------------------------------------
CSV_PATH   = "WeatherPredict/weather_all_nodes.csv"
MODEL_PATH = "LastFile_1.0/best_timegnn.pt"
A_PATH     = "LastFile_1.0/A_norm.npy"
MU_PATH    = "LastFile_1.0/mu.npy"
SD_PATH    = "LastFile_1.0/sd.npy"
NODES_PATH = "LastFile_1.0/nodes.npy"
FEATS_PATH = "LastFile_1.0/feature_cols.npy"

# Sequence length (look-back window in days)
L = 30
# Alert threshold (probability ≥ TH triggers an alert)
TH = 0.53
# Warn when missing-data rate in the window exceeds this fraction
WARN_MISSING_RATE = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Base day (Vietnam local time, rounded to midnight)
# ---------------------------------------------------------------------------
def get_base_day_vn() -> pd.Timestamp:
    """
    Return today's date in the Vietnam timezone (UTC+7), truncated to midnight.
    The forecast window covers BASE_DAY+1 through BASE_DAY+7.
    """
    now_vn = pd.Timestamp.now(tz=VN_TZ)
    return now_vn.floor("D").tz_localize(None)


BASE_DAY = get_base_day_vn()
# Uncomment the line below to hard-code a date for testing:
# BASE_DAY = pd.Timestamp("2025-09-30")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window statistics and cyclical day-of-year features.

    New columns added for each base weather variable:
        <col>_mean7, <col>_mean14, <col>_mean30

    For precipitation specifically:
        precip_sum7, precip_sum30

    Cyclical encoding of day-of-year:
        doy_sin, doy_cos
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df["node"] = df["node"].astype(str)
    df = df.sort_values(["node", "date"]).reset_index(drop=True)

    # Day-of-year cyclical features
    doy = df["date"].dt.dayofyear.values.astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    def roll(group, window, how="mean"):
        r = group.rolling(window, min_periods=1)
        if how == "mean": return r.mean()
        if how == "sum":  return r.sum()
        if how == "min":  return r.min()
        if how == "max":  return r.max()
        raise ValueError(f"Unknown aggregation: {how!r}")

    base_cols = [
        c for c in ["temp", "humidity", "precip", "windspeed", "cloudcover", "uvindex"]
        if c in df.columns
    ]
    for col in base_cols:
        series = df.groupby("node")[col]
        for window in (7, 14, 30):
            df[f"{col}_mean{window}"] = (
                roll(series, window, "mean")
                .reset_index(level=0, drop=True)
                .astype(np.float32)
            )

    if "precip" in df.columns:
        series = df.groupby("node")["precip"]
        df["precip_sum7"]  = roll(series, 7,  "sum").reset_index(level=0, drop=True).astype(np.float32)
        df["precip_sum30"] = roll(series, 30, "sum").reset_index(level=0, drop=True).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class TimeGCN_GRU(nn.Module):
    """
    Spatio-temporal model combining a Graph Convolutional Network (GCN)
    and a Gated Recurrent Unit (GRU).

    Architecture:
        For each time step t:
            x_mix = A_norm @ x_t          (graph neighbourhood aggregation)
            h_t   = dropout(relu(GCN(x_mix)))

        For each node n:
            _, h_n = GRU(h[:, :, n, :])
            logit  = Linear(dropout(h_n[-1]))

    Args:
        A_norm  : Normalised adjacency matrix, shape (N, N).
        in_dim  : Number of input features per node.
        gcn_dim : Hidden dimension of the GCN linear layer.
        gru_dim : Hidden dimension of the GRU.
        dropout : Dropout probability applied after GCN and before the head.
    """

    def __init__(self, A_norm: np.ndarray, in_dim: int,
                 gcn_dim: int = 32, gru_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))
        self.gcn  = nn.Linear(in_dim, gcn_dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gru  = nn.GRU(input_size=gcn_dim, hidden_size=gru_dim, batch_first=True)
        self.head = nn.Linear(gru_dim, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Tensor of shape (B, L, N, F) — batch of node feature sequences.

        Returns:
            Tensor of shape (B, N) — raw logits per node.
        """
        B, Lx, N, _ = x_seq.shape
        A = self.A

        # GCN step over each time slice
        hs = []
        for t in range(Lx):
            xt   = x_seq[:, t, :, :]                           # (B, N, F)
            xmix = torch.einsum("ij,bjf->bif", A, xt)          # (B, N, F)
            ht   = self.drop(self.act(self.gcn(xmix)))          # (B, N, gcn_dim)
            hs.append(ht)

        h = torch.stack(hs, dim=1)  # (B, L, N, gcn_dim)

        # GRU step per node
        logits = []
        for i in range(N):
            hi   = h[:, :, i, :]                       # (B, L, gcn_dim)
            _, hn = self.gru(hi)                        # hn: (1, B, gru_dim)
            last  = hn[-1]                              # (B, gru_dim)
            logit = self.head(self.drop(last)).squeeze(-1)  # (B,)
            logits.append(logit)

        return torch.stack(logits, dim=1)  # (B, N)


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------
def risk_level(p: float) -> str:
    """Map a probability in [0, 1] to a risk category string."""
    if p >= 0.75: return "HIGH"
    if p >= TH:   return "WARNING"
    if p >= 0.30: return "WATCH"
    return "LOW"


def advice_text(risk: str) -> str:
    """Return a short action-advice string for a given risk category."""
    mapping = {
        "HIGH":    "High Alert (Priority Inspection)",
        "WARNING": "High Alert (Prepare Plan)",
        "WATCH":   "Monitor Closely (Early Warning)",
        "LOW":     "Normal",
    }
    return mapping.get(risk, "Normal")


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------
def run_prediction():
    """
    Execute the full prediction pipeline and return results.

    Steps:
        1. Load model artifacts (adjacency matrix, normalisation stats, metadata).
        2. Load and reconstruct the model from checkpoint.
        3. Read the weather CSV and apply feature engineering.
        4. Build the (L, N, F) feature tensor for the current window.
        5. Run inference and compute fire probabilities.
        6. Assemble and return metadata dict + result DataFrame.

    Returns:
        meta (dict): Run metadata (dates, counts, diagnostics).
        out  (pd.DataFrame): Per-node predictions, sorted by probability descending.
    """
    # 1. Load artifacts
    A_norm       = np.load(A_PATH).astype(np.float32)
    mu           = np.load(MU_PATH).astype(np.float32).reshape(1, -1)
    sd           = np.load(SD_PATH).astype(np.float32).reshape(1, -1)
    nodes        = [str(x) for x in np.load(NODES_PATH, allow_pickle=True).tolist()]
    feature_cols = np.load(FEATS_PATH, allow_pickle=True).tolist()

    N, F = len(nodes), len(feature_cols)

    if mu.shape[-1] != F or sd.shape[-1] != F:
        raise ValueError(f"mu/sd shape mismatch — mu={mu.shape}, sd={sd.shape}, F={F}")
    if A_norm.shape != (N, N):
        raise ValueError(f"A_norm shape {A_norm.shape} expected ({N}, {N})")

    # 2. Load model checkpoint
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    gcn_dim = state["gcn.weight"].shape[0]
    gru_dim = state["head.weight"].shape[1]

    model = TimeGCN_GRU(A_norm, in_dim=F, gcn_dim=gcn_dim, gru_dim=gru_dim, dropout=0.0).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3. Load and preprocess CSV
    df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    if not {"date", "node"}.issubset(df.columns):
        raise ValueError("CSV must contain at minimum the columns: 'node', 'date'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).copy()
    df["node"] = df["node"].astype(str)
    df = add_engineered_features(df)

    min_date = df["date"].min()
    max_date = df["date"].max()

    base_day = max_date if BASE_DAY is None else pd.Timestamp(BASE_DAY).floor("D")
    if base_day > max_date:
        raise ValueError(
            f"BASE_DAY {base_day.date()} is beyond the latest date in CSV ({max_date.date()})"
        )

    start      = base_day - pd.Timedelta(days=L - 1)
    full_dates = pd.date_range(start, base_day, freq="D")

    # Resolve lat/lon column names
    lat_col = next((c for c in ("lat", "latitude") if c in df.columns), None)
    lon_col = next((c for c in ("lon", "longitude") if c in df.columns), None)
    node_latlon = {n: (float("nan"), float("nan")) for n in nodes}
    if lat_col and lon_col:
        tmp = df.groupby("node")[[lat_col, lon_col]].mean(numeric_only=True).reset_index()
        for _, row in tmp.iterrows():
            node_latlon[str(row["node"])] = (float(row[lat_col]), float(row[lon_col]))

    # 4. Build feature tensor
    base_weather_cols = [c for c in feature_cols if c != "weather_missing"]
    missing_cols      = [c for c in base_weather_cols if c not in df.columns]

    X_list            = []
    node_missing_rate = {}

    for node in nodes:
        g = df[(df["node"] == node) & (df["date"] <= base_day)].copy()
        if g.empty:
            raise ValueError(f"No data found in CSV for node={node!r}")

        present = [c for c in base_weather_cols if c in g.columns]
        g = g.groupby("date", as_index=False)[present].mean(numeric_only=True) if present \
            else g[["date"]].drop_duplicates()

        g = g.set_index("date").reindex(full_dates)
        for c in base_weather_cols:
            if c not in g.columns:
                g[c] = np.nan

        miss_flag = g[base_weather_cols].isna().any(axis=1).astype(np.float32).to_numpy()
        node_missing_rate[node] = float(miss_flag.mean())

        g[base_weather_cols] = g[base_weather_cols].ffill()
        for c in base_weather_cols:
            if g[c].isna().any():
                g[c] = g[c].fillna(float(mu[0, feature_cols.index(c)]))

        if "weather_missing" in feature_cols:
            g["weather_missing"] = miss_flag

        X_list.append(g[feature_cols].to_numpy(dtype=np.float32))

    X_raw = np.stack(X_list, axis=1)                                    # (L, N, F)
    Xz    = (X_raw - mu.reshape(1, 1, -1)) / (sd.reshape(1, 1, -1) + 1e-6)

    xz_std       = float(np.std(Xz))
    xz_zero_ratio = float(np.mean(np.abs(Xz) < 1e-3))

    # 5. Inference
    x_seq = torch.tensor(Xz, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, L, N, F)
    with torch.no_grad():
        prob = torch.sigmoid(model(x_seq)).cpu().numpy()[0]  # (N,)

    alert = (prob >= TH).astype(int)

    # 6. Assemble results
    meta = {
        "base_day":                    str(base_day.date()),
        "history_days":                int(L),
        "th_alert":                    float(TH),
        "forecast_from":               str((base_day + pd.Timedelta(days=1)).date()),
        "forecast_to":                 str((base_day + pd.Timedelta(days=7)).date()),
        "csv_min_date":                str(min_date.date()),
        "csv_max_date":                str(max_date.date()),
        "window_from":                 str(full_dates[0].date()),
        "window_to":                   str(full_dates[-1].date()),
        "device":                      DEVICE,
        "N_nodes":                     int(N),
        "F_feats":                     int(F),
        "xz_std":                      xz_std,
        "xz_zero_like_ratio":          xz_zero_ratio,
        "missing_feature_cols_count":  int(len(missing_cols)),
        "missing_feature_cols":        missing_cols[:200],
        "warn_missing_rate_threshold": float(WARN_MISSING_RATE),
    }

    risks = [risk_level(float(p)) for p in prob]
    out   = pd.DataFrame({
        "node":              nodes,
        "lat":               [node_latlon[n][0] for n in nodes],
        "lon":               [node_latlon[n][1] for n in nodes],
        "prob_fire_next7":   prob,
        "prob_%":            prob * 100.0,
        "risk":              risks,
        "alert_next7":       alert,
        "missing_rate_Ldays":[node_missing_rate[n] for n in nodes],
        "advice":            [advice_text(r) for r in risks],
    }).sort_values("prob_fire_next7", ascending=False).reset_index(drop=True)

    return meta, out


# ---------------------------------------------------------------------------
# CLI entry point (quick smoke-test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    meta, out = run_prediction()
    print(meta)
    print(out.head(10).to_string(index=False))
