"""
app.py
------
FastAPI application serving the fire-risk prediction dashboard.

Endpoints:
    GET  /api/latest  — Return cached prediction results (runs prediction
                        on first request if cache is empty).
    POST /api/run     — Force a fresh prediction run and update the cache.
    GET  /            — Serve the static front-end from the 'web/' directory.
"""

from datetime import datetime, timezone
import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from predict_v2 import run_prediction

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Last Fire — Fire Risk Prediction API",
    description="Time-GCN + GRU model for 7-day forest fire risk assessment.",
    version="1.0.0",
)

# In-memory result cache (populated on the first request or via /api/run)
_CACHE: dict = {"ts": None, "meta": None, "rows": []}


def _df_to_rows(df) -> list[dict]:
    """Convert a DataFrame to a list of JSON-serialisable row dicts."""
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/latest", summary="Get latest prediction results")
def get_latest():
    """
    Return the most recent prediction result from the in-memory cache.
    If the cache is empty (first request after server start), the prediction
    is run synchronously before responding.
    """
    if not _CACHE["rows"]:
        meta, df = run_prediction()
        _CACHE["ts"]   = datetime.now(timezone.utc).isoformat()
        _CACHE["meta"] = meta
        _CACHE["rows"] = _df_to_rows(df)

    return JSONResponse({
        "ts":   _CACHE["ts"],
        "meta": _CACHE["meta"],
        "rows": _CACHE["rows"],
    })


@app.post("/api/run", summary="Trigger a fresh prediction run")
async def run_now():
    """
    Execute the prediction pipeline in a background thread (non-blocking),
    update the cache, and return a confirmation with the new timestamp.
    """
    meta, df = await asyncio.to_thread(run_prediction)
    _CACHE["ts"]   = datetime.now(timezone.utc).isoformat()
    _CACHE["meta"] = meta
    _CACHE["rows"] = _df_to_rows(df)

    return JSONResponse({
        "ok":    True,
        "ts":    _CACHE["ts"],
        "count": len(_CACHE["rows"]),
    })


# ---------------------------------------------------------------------------
# Static front-end (must be mounted last)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="web", html=True), name="static")
