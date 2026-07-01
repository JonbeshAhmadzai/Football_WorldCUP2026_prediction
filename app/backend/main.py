from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backend import services


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIST = BASE_DIR / "app" / "frontend" / "dist"

api = FastAPI(title="World Cup 2026 Prediction API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    model_label: str | None = None
    country: str = ""


class BacktestRequest(BaseModel):
    model_label: str | None = None
    source: Literal["selected", "augmented", "live"] = "selected"
    limit: int = 25


@api.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "react-fastapi",
        "airflow_dag": "src/scheduler/airflow_augmented_worldcup_dag.py",
    }


@api.get("/api/options")
def options():
    return services.options_payload()


@api.get("/api/models")
def models():
    return {"models": services.model_registry()}


@api.get("/api/model-health")
def model_health():
    return services.model_health()


@api.get("/api/pipeline/status")
def pipeline_status():
    return services.pipeline_status()


@api.get("/api/dashboard")
def dashboard(
    model: str | None = None,
    team: str = "All teams",
    metric: str = "win_pct",
    sort: Literal["asc", "desc"] = "desc",
    top: int = Query(default=12, ge=5, le=48),
    simulate_bracket: bool = False,
):
    try:
        return services.dashboard(model, team, metric, sort, top, simulate_bracket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/api/refresh-live")
def refresh_live(force: bool = False):
    result = services.refresh_live_snapshot(force=force)
    if result.get("last_ok") is False and not result.get("skipped"):
        raise HTTPException(status_code=500, detail=result)
    return result


@api.post("/api/predict-match")
def predict_match(payload: MatchPredictionRequest):
    try:
        return services.predict_match(
            payload.home_team,
            payload.away_team,
            payload.model_label,
            payload.country,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/api/backtest")
def backtest(payload: BacktestRequest):
    try:
        return services.backtest(payload.model_label, payload.source, payload.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        api.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@api.get("/{full_path:path}")
def serve_react_app(full_path: str):
    index_path = FRONTEND_DIST / "index.html"
    requested_path = FRONTEND_DIST / full_path
    if full_path and requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(
        status_code=404,
        detail="React build not found. Run `npm install` and `npm run build` inside app/frontend.",
    )


app = api
