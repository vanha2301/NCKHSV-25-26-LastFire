"""
TimeGNN GNN(GCN + GRU) inference script
-----------------------------------
Mục tiêu:
- Dùng L ngày lịch sử (history window) đến BASE_DAY để dự đoán xác suất cháy (prob) trong 7 ngày tới cho từng node.
- Chuẩn hóa input theo (mu, sd) đã lưu từ lúc train.
- Trả ra bảng kết quả + risk level + alert.

Yêu cầu file artifacts:
- best_timegnn.pt   (model weights)
- A_norm.npy        (adjacency normalized)
- mu.npy, sd.npy    (chuẩn hóa feature)
- nodes.npy         (danh sách node theo đúng thứ tự train)
- feature_cols.npy  (danh sách feature theo đúng thứ tự train)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================================================
# CONFIG
# =========================================================
CSV_PATH   = r"D:\NCKHSV-25-26-LastFire\weather_dec2025_all_nodes.csv"
MODEL_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\best_timegnn.pt"
A_PATH     = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\A_norm.npy"
MU_PATH    = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\mu.npy"
SD_PATH    = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\sd.npy"
NODES_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\nodes.npy"
FEATS_PATH = r"D:\NCKHSV-25-26-LastFire\LastFile_1.0\feature_cols.npy"

BASE_DAY = pd.Timestamp("2025-12-31")  # dự đoán cho 2026-01-01 -> 2026-01-07
HISTORY_DAYS = 30                      # L (giống train)
TH_ALERT = 0.53                        # ngưỡng bật cảnh báo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Risk levels chỉ để diễn giải khi báo cáo
RISK_LEVELS = [
    (0.00, 0.45, "LOW",     "🟢", "Bình thường"),
    (0.45, 0.55, "WATCH",   "🟡", "Theo dõi sát (cảnh báo sớm)"),
    (0.55, 0.70, "WARNING", "🟠", "Cảnh giác cao (chuẩn bị phương án)"),
    (0.70, 1.01, "HIGH",    "🔴", "Cảnh báo cao (ưu tiên kiểm tra)"),
]

# =========================================================
# MODEL
# =========================================================
class TimeGCN_GRU(nn.Module):
    """
    Input:  x_seq [B, L, N, F]
    - GCN (linear) theo A_norm cho từng time-step
    - GRU theo trục thời gian cho từng node
    Output: logits [B, N]  (sau đó sigmoid => prob)
    """
    def __init__(self, A_norm: np.ndarray, in_dim: int, gcn_dim=32, gru_dim=64, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))  # [N,N]
        self.gcn = nn.Linear(in_dim, gcn_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=gcn_dim, hidden_size=gru_dim, batch_first=True)
        self.head = nn.Linear(gru_dim, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x_seq.shape
        A = self.A

        # Spatial mixing (GCN) per time-step
        hs = []
        for t in range(L):
            xt = x_seq[:, t, :, :]                          # [B,N,F]
            xmix = torch.einsum("ij,bjf->bif", A, xt)        # [B,N,F]
            ht = self.drop(self.act(self.gcn(xmix)))         # [B,N,gcn_dim]
            hs.append(ht)
        h = torch.stack(hs, dim=1)                           # [B,L,N,gcn_dim]

        # Temporal modeling (GRU) per node
        logits = []
        for i in range(N):
            hi = h[:, :, i, :]                               # [B,L,gcn_dim]
            _, hn = self.gru(hi)                             # [1,B,gru_dim]
            last = hn[-1]                                    # [B,gru_dim]
            logit = self.head(self.drop(last)).squeeze(-1)   # [B]
            logits.append(logit)

        return torch.stack(logits, dim=1)                    # [B,N]


# =========================================================
# HELPERS
# =========================================================
def load_artifacts():
    A_norm = np.load(A_PATH).astype(np.float32)
    mu = np.load(MU_PATH).astype(np.float32)   # shape (1,F) hoặc (F,)
    sd = np.load(SD_PATH).astype(np.float32)   # shape (1,F) hoặc (F,)
    nodes = np.load(NODES_PATH, allow_pickle=True).tolist()
    feature_cols = np.load(FEATS_PATH, allow_pickle=True).tolist()

    # reshape chuẩn (1,F)
    mu = mu.reshape(1, -1)
    sd = sd.reshape(1, -1)

    return A_norm, mu, sd, nodes, feature_cols


def load_model(A_norm, feature_cols):
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # infer dims từ weights để chắc chắn khớp
    gcn_dim = state["gcn.weight"].shape[0]
    gru_dim = state["head.weight"].shape[1]

    model = TimeGCN_GRU(
        A_norm=A_norm,
        in_dim=len(feature_cols),
        gcn_dim=gcn_dim,
        gru_dim=gru_dim,
        dropout=0.0,  # inference thường tắt dropout để ổn định
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, gcn_dim, gru_dim


def read_weather_table(path: str) -> pd.DataFrame:
    """
    Input file có thể là .csv hoặc .xls/.xlsx.
    - Với .xls/.xlsx: dùng read_excel.
    - Với .csv: dùng read_csv.
    """
    lower = path.lower()
    if lower.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return df


def get_latlon_map(df: pd.DataFrame) -> dict:
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)

    node_latlon = {}
    if lat_col and lon_col:
        tmp = df.groupby("node")[[lat_col, lon_col]].mean(numeric_only=True).reset_index()
        for _, r in tmp.iterrows():
            node_latlon[str(r["node"])] = (float(r[lat_col]), float(r[lon_col]))
    else:
        for n in df["node"].astype(str).unique():
            node_latlon[str(n)] = (np.nan, np.nan)
    return node_latlon


def build_history_window(
    df: pd.DataFrame,
    nodes: list,
    feature_cols: list,
    mu: np.ndarray,
    base_day: pd.Timestamp,
    history_days: int
) -> np.ndarray:
    """
    Tạo tensor input Xz có shape [1, L, N, F] (standardized)
    Quy tắc xử lý thiếu dữ liệu:
    - Reindex theo đủ L ngày (start->end)
    - miss_flag = 1 nếu trong ngày đó có bất kỳ feature NaN (trước khi fill)
    - fill missing theo ffill (causal)
    - nếu vẫn NaN (đầu chuỗi / cột thiếu hoàn toàn) -> fill bằng mu (từ train)
    """

    if "date" not in df.columns or "node" not in df.columns:
        raise ValueError("Input data phải có cột: node, date")

    df = df.copy()
    df["node"] = df["node"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")

    end = base_day
    start = end - pd.Timedelta(days=history_days - 1)
    full_dates = pd.date_range(start, end, freq="D")

    # tách ra danh sách feature thời tiết (trừ cờ missing nếu có)
    base_weather_cols = [c for c in feature_cols if c != "weather_missing"]
    F = len(feature_cols)

    X_list = []
    missing_nodes = []

    for n in nodes:
        n = str(n)
        g = df[(df["node"] == n) & (df["date"] <= base_day)].copy()

        if g.empty:
            missing_nodes.append(n)
            continue

        # ép đủ L ngày
        g = g.sort_values("date").set_index("date").reindex(full_dates).reset_index()
        g = g.rename(columns={"index": "date"})

        # đảm bảo đủ cột feature
        for c in base_weather_cols:
            if c not in g.columns:
                g[c] = np.nan

        # missing flag theo ngày (trước khi fill)
        miss_flag = g[base_weather_cols].isna().any(axis=1).astype(np.float32).to_numpy()

        # causal fill: forward-fill
        g[base_weather_cols] = g[base_weather_cols].ffill()

        # còn NaN -> fill bằng mu (train mean)
        for c in base_weather_cols:
            if g[c].isna().any():
                idx = feature_cols.index(c)
                g[c] = g[c].fillna(float(mu[0, idx]))

        # thêm weather_missing nếu model cần
        if "weather_missing" in feature_cols:
            g["weather_missing"] = miss_flag

        Xn = g[feature_cols].to_numpy(dtype=np.float32)  # (L,F)
        if Xn.shape != (history_days, F):
            raise ValueError(f"Node {n} có shape {Xn.shape}, expected {(history_days, F)}")

        X_list.append(Xn)

    if missing_nodes:
        raise ValueError(f"Thiếu dữ liệu cho các node: {missing_nodes}")

    # stack -> (L,N,F)
    X_raw = np.stack(X_list, axis=1)
    return X_raw


def standardize(X_raw: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    X_raw: (L,N,F)
    mu, sd: (1,F)
    => Xz: (L,N,F)
    """
    return (X_raw - mu.reshape(1, 1, -1)) / (sd.reshape(1, 1, -1) + 1e-6)


def add_risk_levels(out: pd.DataFrame, prob_col="prob_fire_next7", th_alert=0.53) -> pd.DataFrame:
    p = out[prob_col].astype(float).clip(0, 1)
    out2 = out.copy()

    out2["prob_%"] = (p * 100).round(2)
    out2["alert_next7"] = (p >= float(th_alert)).astype(np.int8)

    risk_name, risk_icon, advice = [], [], []
    for v in p.to_numpy():
        for lo, hi, name, icon, adv in RISK_LEVELS:
            if lo <= v < hi:
                risk_name.append(name)
                risk_icon.append(icon)
                advice.append(adv)
                break

    out2["risk"] = [f"{ic} {nm}" for ic, nm in zip(risk_icon, risk_name)]
    out2["advice"] = advice

    return out2.sort_values(prob_col, ascending=False).reset_index(drop=True)


def print_report(out2: pd.DataFrame, base_day: pd.Timestamp, th_alert: float):
    f_from = (base_day + pd.Timedelta(days=1)).date()
    f_to = (base_day + pd.Timedelta(days=7)).date()

    print("\n" + "=" * 90)
    print("TimeGNN NEXT-7D FIRE RISK REPORT")
    print(f"BASE_DAY: {base_day.date()} | FORECAST: {f_from} -> {f_to} | TH_ALERT: {th_alert}")
    print("=" * 90)

    n_alert = int(out2["alert_next7"].sum())
    print(f"Alerts: {n_alert}/{len(out2)} nodes (prob >= {th_alert})")
    print("-" * 90)

    # bảng top (console)
    show_cols = ["node", "lat", "lon", "prob_fire_next7", "prob_%", "risk", "alert_next7", "advice"]
    view = out2[show_cols].copy()
    view["prob_fire_next7"] = view["prob_fire_next7"].map(lambda x: f"{float(x):.4f}")
    view["prob_%"] = view["prob_%"].map(lambda x: f"{float(x):.2f}%")
    print(view.to_string(index=False))
    print("-" * 90)

    # summary theo risk
    print("Risk summary:")
    print(out2["risk"].value_counts().to_string())

    if n_alert == 0:
        print("\n✅ OK: Không node nào vượt threshold trong 7 ngày tới.")
    else:
        print("\n⚠️  CẢNH BÁO: Có node vượt threshold — ưu tiên kiểm tra các node alert_next7=1.")


# =========================================================
# MAIN
# =========================================================
def main():
    # 1) Load artifacts
    A_norm, mu, sd, nodes, feature_cols = load_artifacts()
    N, F = len(nodes), len(feature_cols)

    # 2) Load model
    model, gcn_dim, gru_dim = load_model(A_norm, feature_cols)

    # 3) Read weather table
    df = read_weather_table(CSV_PATH)

    # basic validation on date range
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    max_date = df["date"].max()
    if BASE_DAY > max_date:
        raise ValueError(f"BASE_DAY={BASE_DAY.date()} nhưng data chỉ có tới {max_date.date()}")

    node_latlon = get_latlon_map(df)

    print(f"Loaded model OK | N={N} F={F} gcn_dim={gcn_dim} gru_dim={gru_dim} DEVICE={DEVICE}")
    print(f"Data range: {df['date'].min().date()} -> {max_date.date()}")

    # 4) Build input window (L,N,F) then standardize
    X_raw = build_history_window(
        df=df,
        nodes=[str(n) for n in nodes],
        feature_cols=feature_cols,
        mu=mu,
        base_day=BASE_DAY,
        history_days=HISTORY_DAYS,
    )
    Xz = standardize(X_raw, mu, sd)

    # 5) Predict
    x_seq = torch.tensor(Xz, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,L,N,F]
    with torch.no_grad():
        prob = torch.sigmoid(model(x_seq)).cpu().numpy()[0]  # [N]

    # 6) Assemble output table
    nodes_str = [str(n) for n in nodes]
    out = pd.DataFrame({
        "node": nodes_str,
        "lat": [node_latlon.get(n, (np.nan, np.nan))[0] for n in nodes_str],
        "lon": [node_latlon.get(n, (np.nan, np.nan))[1] for n in nodes_str],
        "prob_fire_next7": prob,
    }).sort_values("prob_fire_next7", ascending=False).reset_index(drop=True)

    # 7) Risk levels + report
    out2 = add_risk_levels(out, prob_col="prob_fire_next7", th_alert=TH_ALERT)

    # In notebook thì có thể display(out2). Ngoài console thì in report:
    try:
        display(out2)  # type: ignore
    except Exception:
        pass

    print_report(out2, base_day=BASE_DAY, th_alert=TH_ALERT)

    # (tuỳ chọn) lưu ra file để nộp báo cáo
    # out2.to_csv("predict_next7_report.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
