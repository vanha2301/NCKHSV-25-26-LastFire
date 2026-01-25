from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import asyncio

from predict import run_prediction

app = FastAPI()

CACHE = {"ts": None, "meta": None, "rows": []}

def df_to_rows(df):
    return df.to_dict(orient="records")

@app.get("/api/latest")
def latest():
    if not CACHE["rows"]:
        meta, df = run_prediction()
        CACHE["ts"] = datetime.utcnow().isoformat()
        CACHE["meta"] = meta
        CACHE["rows"] = df_to_rows(df)
    return JSONResponse({"ts": CACHE["ts"], "meta": CACHE["meta"], "rows": CACHE["rows"]})

@app.post("/api/run")
async def run_now():
    meta, df = await asyncio.to_thread(run_prediction)
    CACHE["ts"] = datetime.utcnow().isoformat()
    CACHE["meta"] = meta
    CACHE["rows"] = df_to_rows(df)
    return JSONResponse({"ok": True, "ts": CACHE["ts"], "count": len(CACHE["rows"])})

# ✅ đặt mount xuống cuối
app.mount("/", StaticFiles(directory="web", html=True), name="web")
