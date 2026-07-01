from __future__ import annotations

import json
import math
import pickle
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backend.quality import data_quality_report
from src.storage import live_store


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "src" / "modeling"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
SIMULATION_RUNS_DIR = MODEL_DIR / "simulation_runs"
SCORE_ARTIFACTS_DIR = MODEL_DIR / "score_artifacts"
SCRAPER_SCRIPT = BASE_DIR / "src" / "scraping" / "scrape_latest_results.py"
LIVE_REFRESH_MIN_SECONDS = 20

_refresh_lock = threading.Lock()
_refresh_state: dict[str, Any] = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_ok": None,
    "last_message": "No live refresh has run from the app yet.",
}

MAX_SCORE_GRID = 6

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_recent_wins",
    "home_recent_draws",
    "home_recent_losses",
    "home_recent_goals_for",
    "home_recent_goals_against",
    "home_recent_goal_diff",
    "away_recent_wins",
    "away_recent_draws",
    "away_recent_losses",
    "away_recent_goals_for",
    "away_recent_goals_against",
    "away_recent_goal_diff",
    "recent_win_diff",
    "recent_goal_diff_diff",
    "neutral",
]

MODEL_KIND_LABELS = {
    "baseline": "Original cloned model",
    "augmented": "Historical + ESPN live model",
    "historical_live": "Historical + ESPN live model",
    "live_result": "Live result model",
}

KNOCKOUT_STAGE_ORDER = [
    "round-of-32",
    "round-of-16",
    "quarterfinals",
    "semifinals",
    "3rd-place-match",
    "final",
]

KNOCKOUT_STAGE_LABELS = {
    "round-of-32": "Round of 32",
    "round-of-16": "Round of 16",
    "quarterfinals": "Quarter-finals",
    "semifinals": "Semi-finals",
    "3rd-place-match": "Third place",
    "final": "Final",
}

PLACEHOLDER_TOKENS = [
    "Winner",
    "Loser",
    "Finalist",
    "Best 3rd",
    "Round of",
    "Quarterfinal",
    "Semifinal",
    "3rd Place",
    "R32",
    "QF",
    "SF",
]

DEFAULT_FORM = {
    "last5_wins": 2,
    "last5_draws": 1,
    "last5_losses": 2,
    "last5_goals_for": 5,
    "last5_goals_against": 5,
    "last5_goal_difference": 0,
}

TEAM_NAME_MAP = {
    "Congo DR": "DR Congo",
    "Türkiye": "Turkey",
    "USA": "United States",
}


def clean_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.head(limit).copy() if limit else df.copy()
    return [{key: clean_value(value) for key, value in row.items()} for row in out.to_dict("records")]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if path.name == "worldcup_2026_live_results.csv" and live_store.is_enabled():
        try:
            db_results = live_store.load_live_results()
            if not db_results.empty:
                return db_results
        except Exception:
            pass
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def path_modified_at(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def refresh_live_snapshot(force: bool = False) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    with _refresh_lock:
        if _refresh_state["running"]:
            return {**_refresh_state, "skipped": True, "message": "ESPN refresh is already running."}

        last_started_at = _refresh_state.get("last_started_at")
        if not force and last_started_at:
            age = (now - datetime.fromisoformat(last_started_at)).total_seconds()
            if age < LIVE_REFRESH_MIN_SECONDS:
                return {
                    **_refresh_state,
                    "skipped": True,
                    "message": f"ESPN refresh skipped; last refresh started {age:.0f}s ago.",
                }

        _refresh_state.update(
            {
                "running": True,
                "last_started_at": now.isoformat(timespec="seconds"),
                "last_finished_at": None,
                "last_ok": None,
                "last_message": "Refreshing ESPN scoreboard...",
            }
        )

    try:
        result = subprocess.run(
            [sys.executable, str(SCRAPER_SCRIPT)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=75,
            check=False,
        )
        ok = result.returncode == 0
        output = (result.stdout or result.stderr or "").strip().splitlines()
        if ok:
            refreshed = read_csv(DATA_DIR / "worldcup_2026_live_results.csv")
            if not refreshed.empty:
                live_count = int(refreshed["is_live"].sum()) if "is_live" in refreshed else 0
                finished_count = int(refreshed["is_finished"].sum()) if "is_finished" in refreshed else 0
                scheduled_count = int(refreshed["is_scheduled"].sum()) if "is_scheduled" in refreshed else 0
                storage = "PostgreSQL" if live_store.is_enabled() else "local CSV"
                message = (
                    f"ESPN refreshed in {storage}: "
                    f"{live_count} live, {finished_count} finished, {scheduled_count} scheduled."
                )
            else:
                message = "ESPN refresh completed."
        else:
            message = output[-1] if output else "ESPN refresh failed."
        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _refresh_lock:
            _refresh_state.update(
                {
                    "running": False,
                    "last_finished_at": finished_at,
                    "last_ok": ok,
                    "last_message": message,
                    "returncode": result.returncode,
                }
            )
            return {**_refresh_state, "skipped": False}
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _refresh_lock:
            _refresh_state.update(
                {
                    "running": False,
                    "last_finished_at": finished_at,
                    "last_ok": False,
                    "last_message": str(exc),
                    "returncode": None,
                }
            )
            return {**_refresh_state, "skipped": False}


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def add_current_artifact_option(
    options: dict[str, dict[str, Any]],
    artifact_dir_name: str,
    label: str,
    kind: str,
    fallback_patterns: list[str] | None = None,
):
    run_dir = ARTIFACTS_DIR / artifact_dir_name
    if not (run_dir / "model.pkl").exists() and fallback_patterns:
        for pattern in fallback_patterns:
            fallback = next(iter(sorted(ARTIFACTS_DIR.glob(pattern), reverse=True)), None)
            if fallback and (fallback / "model.pkl").exists():
                run_dir = fallback
                break

    model_path = run_dir / "model.pkl"
    encoder_path = run_dir / "label_encoder.pkl"
    if not model_path.exists() or not encoder_path.exists():
        return

    simulation_path = SIMULATION_RUNS_DIR / f"simulation_results_{artifact_dir_name}.csv"
    if not simulation_path.exists() and fallback_patterns:
        for pattern in fallback_patterns:
            run_prefix = pattern.replace("*", "")
            fallback_sim = next(
                iter(sorted(SIMULATION_RUNS_DIR.glob(f"simulation_results_{run_prefix}*.csv"), reverse=True)),
                None,
            )
            if fallback_sim:
                simulation_path = fallback_sim
                break

    options[label] = {
        "label": label,
        "model": model_path,
        "encoder": encoder_path,
        "metrics": run_dir / "metrics.json",
        "simulation": simulation_path if simulation_path.exists() else None,
        "kind": kind,
        "run_id": artifact_dir_name,
    }


def artifact_options() -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    add_current_artifact_option(
        options,
        "historical_live",
        "Historical + ESPN live model",
        "historical_live",
        fallback_patterns=["augmented_*"],
    )
    add_current_artifact_option(
        options,
        "live_result",
        "Live result model",
        "live_result",
        fallback_patterns=["live_result_*", "live_only_*"],
    )
    return options


def model_registry() -> list[dict[str, Any]]:
    registry = []
    for label, selected in artifact_options().items():
        metrics = read_json(selected["metrics"])
        features_path = resolve_features_path(metrics, selected["kind"])
        simulation_path = selected.get("simulation")
        registry.append(
            {
                "label": label,
                "kind": selected["kind"],
                "kind_label": MODEL_KIND_LABELS.get(selected["kind"], selected["kind"]),
                "run_id": selected["run_id"],
                "model_path": relative_path(selected["model"]),
                "encoder_path": relative_path(selected["encoder"]),
                "metrics_path": relative_path(selected["metrics"]),
                "features_path": relative_path(features_path),
                "simulation_path": relative_path(simulation_path),
                "model_modified_at": path_modified_at(selected["model"]),
                "metrics_modified_at": path_modified_at(selected["metrics"]),
                "simulation_modified_at": path_modified_at(simulation_path),
                "last_trained_at": metrics.get("last_trained_at"),
                "training_rows": clean_value(metrics.get("training_rows")),
                "eval_accuracy": clean_value(metrics.get("eval_accuracy")),
                "in_sample_accuracy": clean_value(metrics.get("in_sample_accuracy")),
                "warning": metrics.get("warning"),
            }
        )
    return registry


def model_health() -> dict[str, Any]:
    models = model_registry()
    score_dir = latest_score_artifact()
    score_metrics = read_json(score_dir / "metrics.json") if score_dir else {}
    quality = data_quality_report()
    return {
        "status": quality["status"],
        "models": models,
        "score_model": {
            "path": relative_path(score_dir),
            "last_trained_at": score_metrics.get("last_trained_at"),
            "training_rows": clean_value(score_metrics.get("training_rows")),
            "top3_scoreline_accuracy": clean_value(score_metrics.get("eval_metrics", {}).get("top3_scoreline_accuracy")),
            "modified_at": path_modified_at(score_dir / "metrics.json") if score_dir else None,
        },
        "data_quality": quality,
    }


def pipeline_status() -> dict[str, Any]:
    quality = data_quality_report()
    important_files = [
        DATA_DIR / "worldcup_2026_live_results.csv",
        DATA_DIR / "matches_augmented.csv",
        DATA_DIR / "training_dataset_augmented.csv",
        MODEL_DIR / "training_features_augmented.csv",
        ARTIFACTS_DIR / "historical_live" / "model.pkl",
        ARTIFACTS_DIR / "live_result" / "model.pkl",
        SIMULATION_RUNS_DIR / "simulation_results_historical_live.csv",
        SIMULATION_RUNS_DIR / "simulation_results_live_result.csv",
    ]
    files = [
        {
            "path": relative_path(path),
            "exists": path.exists(),
            "modified_at": path_modified_at(path),
        }
        for path in important_files
    ]
    return {
        "status": quality["status"],
        "airflow_dag": "src/scheduler/airflow_augmented_worldcup_dag.py",
        "refresh_state": _refresh_state,
        "files": files,
        "data_quality_summary": quality["summary"],
    }


def default_model_label() -> str:
    options = artifact_options()
    labels = list(options)
    return next((label for label in labels if label == "Historical + ESPN live model"), labels[0] if labels else "")


def latest_score_artifact() -> Path | None:
    current_dir = SCORE_ARTIFACTS_DIR / "exact_score"
    if (current_dir / "home_goal_model.pkl").exists() and (current_dir / "away_goal_model.pkl").exists():
        return current_dir

    run_dirs = sorted(SCORE_ARTIFACTS_DIR.glob("exact_score_augmented_*"), reverse=True)
    for run_dir in run_dirs:
        if (run_dir / "home_goal_model.pkl").exists() and (run_dir / "away_goal_model.pkl").exists():
            return run_dir
    return None


def latest_fetch_label(results: pd.DataFrame) -> str:
    if results.empty or "fetched_at" not in results:
        return "No ESPN snapshot loaded"
    fetched = pd.to_datetime(results["fetched_at"], errors="coerce", utc=True).dropna()
    if fetched.empty:
        return "No ESPN fetch timestamp"
    return fetched.max().tz_convert("Europe/Brussels").strftime("%b %d, %Y %H:%M Brussels")


def truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def clean_team_name(team: Any) -> str:
    if pd.isna(team):
        return ""
    text = str(team).strip()
    return TEAM_NAME_MAP.get(text, text)


def is_placeholder_team(team: str) -> bool:
    text = str(team)
    return bool(re.match(r"^[12][A-L]$", text)) or any(token in text for token in PLACEHOLDER_TOKENS)


def all_teams() -> list[str]:
    teams: set[str] = set()
    fixtures = read_csv(DATA_DIR / "upcoming_match_features.csv")
    results = read_csv(DATA_DIR / "worldcup_2026_live_results.csv")
    elo = read_csv(DATA_DIR / "elo_latest.csv")
    for df, columns in [
        (fixtures, ["team1", "team2", "team1_normalized", "team2_normalized"]),
        (results, ["home_team", "away_team"]),
        (elo, ["country"]),
    ]:
        for column in columns:
            if column in df:
                teams.update(clean_team_name(value) for value in df[column].dropna())
    return sorted(team for team in teams if team and not is_placeholder_team(team))


def predictor_teams() -> list[str]:
    elo = read_csv(DATA_DIR / "elo_latest.csv")
    if elo.empty or "country" not in elo:
        return []
    teams = [clean_team_name(team) for team in elo["country"].dropna()]
    return sorted(team for team in teams if team and not is_placeholder_team(team))


def team_code_map() -> dict[str, str]:
    elo = read_csv(DATA_DIR / "elo_latest.csv")
    if elo.empty or not {"country", "country_code"}.issubset(elo.columns):
        return {}
    code_overrides = {
        "England": "GB",
        "Scotland": "GB",
    }
    codes = {}
    for _, row in elo.iterrows():
        team = clean_team_name(row.get("country"))
        code = str(row.get("country_code") or "").strip().upper()
        if team and code:
            codes[team] = code_overrides.get(team, code)
    return codes


def team_lookup(df: pd.DataFrame, name_col: str) -> dict[str, pd.Series]:
    if df.empty or name_col not in df:
        return {}
    return {
        clean_team_name(row[name_col]).casefold(): row
        for _, row in df.iterrows()
        if clean_team_name(row[name_col])
    }


def model_bundle(model_label: str) -> tuple[dict[str, Any], Any, Any]:
    options = artifact_options()
    if model_label not in options:
        model_label = default_model_label()
    selected = options[model_label]
    return selected, load_pickle(selected["model"]), load_pickle(selected["encoder"])


def score_bundle() -> tuple[tuple[Any, Any] | None, dict[str, Any], Path | None]:
    score_dir = latest_score_artifact()
    if not score_dir:
        return None, {}, None
    return (
        (load_pickle(score_dir / "home_goal_model.pkl"), load_pickle(score_dir / "away_goal_model.pkl")),
        read_json(score_dir / "metrics.json"),
        score_dir,
    )


def make_prediction_frame(fixtures: pd.DataFrame, model, encoder) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    df = fixtures.copy()
    if "team1_normalized" not in df:
        df["team1_normalized"] = df.get("team1")
    if "team2_normalized" not in df:
        df["team2_normalized"] = df.get("team2")
    df["neutral"] = (
        df["country"].fillna("").str.casefold()
        != df["team1_normalized"].fillna("").str.casefold()
    ).astype(int)
    feature_df = pd.DataFrame(
        {
            "home_elo": df["team1_elo"],
            "away_elo": df["team2_elo"],
            "elo_diff": df["elo_diff"],
            "home_recent_wins": df["team1_last5_wins"],
            "home_recent_draws": df["team1_last5_draws"],
            "home_recent_losses": df["team1_last5_losses"],
            "home_recent_goals_for": df["team1_last5_goals_for"],
            "home_recent_goals_against": df["team1_last5_goals_against"],
            "home_recent_goal_diff": df["team1_last5_goal_difference"],
            "away_recent_wins": df["team2_last5_wins"],
            "away_recent_draws": df["team2_last5_draws"],
            "away_recent_losses": df["team2_last5_losses"],
            "away_recent_goals_for": df["team2_last5_goals_for"],
            "away_recent_goals_against": df["team2_last5_goals_against"],
            "away_recent_goal_diff": df["team2_last5_goal_difference"],
            "recent_win_diff": df["last5_win_difference"],
            "recent_goal_diff_diff": df["last5_goal_diff_difference"],
            "neutral": df["neutral"],
        }
    )
    known_mask = feature_df[FEATURE_COLS].notna().all(axis=1)
    pred_source = df.loc[known_mask].copy()
    if pred_source.empty:
        return pd.DataFrame()
    proba = model.predict_proba(feature_df.loc[known_mask, FEATURE_COLS])
    labels = list(encoder.classes_)
    proba_df = pd.DataFrame(proba, columns=[f"prob_{label}" for label in labels])
    out = pred_source.reset_index(drop=True)
    out = pd.concat([out, proba_df], axis=1)
    out["team1_win_pct"] = out.get("prob_H", 0) * 100
    out["draw_pct"] = out.get("prob_D", 0) * 100
    out["team2_win_pct"] = out.get("prob_A", 0) * 100
    choices = out[["team1_win_pct", "draw_pct", "team2_win_pct"]].idxmax(axis=1)
    out["prediction"] = "Draw"
    out.loc[choices == "team1_win_pct", "prediction"] = out.loc[choices == "team1_win_pct", "team1"]
    out.loc[choices == "team2_win_pct", "prediction"] = out.loc[choices == "team2_win_pct", "team2"]
    out["confidence_pct"] = out[["team1_win_pct", "draw_pct", "team2_win_pct"]].max(axis=1)
    for col in ["team1_win_pct", "draw_pct", "team2_win_pct", "confidence_pct"]:
        out[col] = out[col].round(1)
    return out


def match_features(home_team: str, away_team: str, country: str, elo_lookup: dict[str, pd.Series], form_lookup: dict[str, pd.Series]) -> pd.DataFrame | None:
    home_elo = elo_lookup.get(home_team.casefold())
    away_elo = elo_lookup.get(away_team.casefold())
    home_form = form_lookup.get(home_team.casefold())
    away_form = form_lookup.get(away_team.casefold())
    if home_form is None:
        home_form = DEFAULT_FORM
    if away_form is None:
        away_form = DEFAULT_FORM
    if home_elo is None or away_elo is None:
        return None
    home_rating = float(home_elo["rating"])
    away_rating = float(away_elo["rating"])
    home_wins = float(home_form["last5_wins"])
    away_wins = float(away_form["last5_wins"])
    home_goal_diff = float(home_form["last5_goal_difference"])
    away_goal_diff = float(away_form["last5_goal_difference"])
    host = clean_team_name(country)
    neutral = int(not host or host.casefold() not in {home_team.casefold(), away_team.casefold()})
    return pd.DataFrame(
        [
            {
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_diff": home_rating - away_rating,
                "home_recent_wins": home_wins,
                "home_recent_draws": float(home_form["last5_draws"]),
                "home_recent_losses": float(home_form["last5_losses"]),
                "home_recent_goals_for": float(home_form["last5_goals_for"]),
                "home_recent_goals_against": float(home_form["last5_goals_against"]),
                "home_recent_goal_diff": home_goal_diff,
                "away_recent_wins": away_wins,
                "away_recent_draws": float(away_form["last5_draws"]),
                "away_recent_losses": float(away_form["last5_losses"]),
                "away_recent_goals_for": float(away_form["last5_goals_for"]),
                "away_recent_goals_against": float(away_form["last5_goals_against"]),
                "away_recent_goal_diff": away_goal_diff,
                "recent_win_diff": home_wins - away_wins,
                "recent_goal_diff_diff": home_goal_diff - away_goal_diff,
                "neutral": neutral,
            }
        ]
    )


def result_probabilities(model, encoder, features: pd.DataFrame) -> dict[str, float]:
    probabilities = model.predict_proba(features[FEATURE_COLS])[0]
    by_label = dict(zip(encoder.classes_, probabilities))
    return {
        "home_win": float(by_label.get("H", 0)),
        "away_win": float(by_label.get("A", 0)),
        "draw": float(by_label.get("D", 0)),
    }


def expected_goals(features: pd.DataFrame, score_models: tuple[Any, Any] | None) -> tuple[float, float] | None:
    if score_models is None:
        return None
    home_goal_model, away_goal_model = score_models
    home_expected = float(home_goal_model.predict(features[FEATURE_COLS])[0])
    away_expected = float(away_goal_model.predict(features[FEATURE_COLS])[0])
    return min(max(home_expected, 0.05), MAX_SCORE_GRID), min(max(away_expected, 0.05), MAX_SCORE_GRID)


def scorelines_from_expected(team1_expected: float, team2_expected: float, advancing_team: str, team1: str) -> dict[str, Any]:
    probabilities = []
    for team1_score in range(MAX_SCORE_GRID + 1):
        for team2_score in range(MAX_SCORE_GRID + 1):
            probability = poisson_pmf(team1_expected, team1_score) * poisson_pmf(team2_expected, team2_score)
            probabilities.append((team1_score, team2_score, probability))
    total = sum(probability for _, _, probability in probabilities)
    all_scorelines = sorted(
        [(team1_score, team2_score, probability / total) for team1_score, team2_score, probability in probabilities],
        key=lambda item: item[2],
        reverse=True,
    )
    team1_advances = advancing_team == team1
    compatible = [
        item
        for item in all_scorelines
        if (item[0] >= item[1] if team1_advances else item[1] >= item[0])
    ]
    top_scorelines = (compatible or all_scorelines)[:5]
    primary_team1, primary_team2, primary_probability = top_scorelines[0]
    return {
        "score": f"{primary_team1}-{primary_team2}",
        "score_probability_pct": round(primary_probability * 100, 1),
        "top_scorelines": [
            {"score": f"{team1_score}-{team2_score}", "probability_pct": round(probability * 100, 1)}
            for team1_score, team2_score, probability in top_scorelines
        ],
        "home_expected_goals": round(team1_expected, 2),
        "away_expected_goals": round(team2_expected, 2),
    }


def poisson_pmf(lam: float, goals: int) -> float:
    lam = max(float(lam), 0.05)
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def predict_scorelines(features: pd.DataFrame, score_models: tuple[Any, Any] | None, advancing_team: str, home_team: str) -> dict[str, Any] | None:
    goals = expected_goals(features, score_models)
    if goals is None:
        return None
    home_expected, away_expected = goals
    return scorelines_from_expected(home_expected, away_expected, advancing_team, home_team)


def fallback_projected_score(features: pd.DataFrame, predicted_winner: str, home_team: str) -> str:
    row = features.iloc[0]
    home_attack = row["home_recent_goals_for"] / 5
    away_attack = row["away_recent_goals_for"] / 5
    home_defence_allowed = row["home_recent_goals_against"] / 5
    away_defence_allowed = row["away_recent_goals_against"] / 5
    elo_adjustment = (row["home_elo"] - row["away_elo"]) / 400 * 0.25
    home_score = int(round(min(max(0.2, ((home_attack + away_defence_allowed) / 2) + elo_adjustment), 5)))
    away_score = int(round(min(max(0.2, ((away_attack + home_defence_allowed) / 2) - elo_adjustment), 5)))
    if predicted_winner == home_team and home_score <= away_score:
        home_score = min(5, away_score + 1)
    if predicted_winner != home_team and away_score <= home_score:
        away_score = min(5, home_score + 1)
    return f"{home_score}-{away_score}"


def predict_match(home_team: str, away_team: str, model_label: str | None = None, country: str = "") -> dict[str, Any]:
    if not home_team or not away_team or home_team == away_team:
        raise ValueError("Choose two different teams.")
    selected, model, encoder = model_bundle(model_label or default_model_label())
    score_models, score_metrics, score_dir = score_bundle()
    elo = read_csv(DATA_DIR / "elo_latest.csv")
    form = read_csv(DATA_DIR / "team_recent_form.csv")
    elo_lookup = team_lookup(elo, "country")
    form_lookup = team_lookup(form, "team")
    features = match_features(home_team, away_team, country, elo_lookup, form_lookup)
    reverse_features = match_features(away_team, home_team, country, elo_lookup, form_lookup)
    if features is None or reverse_features is None:
        raise ValueError("One of the selected teams does not have ELO or recent-form features.")

    forward = result_probabilities(model, encoder, features)
    reverse = result_probabilities(model, encoder, reverse_features)
    home_win = (forward["home_win"] + reverse["away_win"]) / 2
    away_win = (forward["away_win"] + reverse["home_win"]) / 2
    draw = (forward["draw"] + reverse["draw"]) / 2
    total = home_win + away_win + draw
    if total:
        home_win, away_win, draw = home_win / total, away_win / total, draw / total

    home_advance = home_win + draw / 2
    away_advance = away_win + draw / 2
    predicted = home_team if home_win >= away_win and home_win >= draw else away_team if away_win >= draw else "Draw"
    advancing_team = home_team if home_advance >= away_advance else away_team

    forward_goals = expected_goals(features, score_models)
    reverse_goals = expected_goals(reverse_features, score_models)
    scoreline = None
    if forward_goals and reverse_goals:
        team1_expected = (forward_goals[0] + reverse_goals[1]) / 2
        team2_expected = (forward_goals[1] + reverse_goals[0]) / 2
        scoreline = scorelines_from_expected(team1_expected, team2_expected, advancing_team, home_team)
    if scoreline is None:
        scoreline = {
            "score": fallback_projected_score(features, advancing_team, home_team),
            "score_probability_pct": None,
            "top_scorelines": [],
            "home_expected_goals": None,
            "away_expected_goals": None,
        }
    feature_row = features.iloc[0]
    feature_summary = {
        "elo_diff": round(float(feature_row["elo_diff"]), 1),
        "recent_win_diff": round(float(feature_row["recent_win_diff"]), 1),
        "recent_goal_diff_diff": round(float(feature_row["recent_goal_diff_diff"]), 1),
        "neutral": bool(feature_row["neutral"]),
    }
    explanation = [
        f"{home_team} ELO is {abs(feature_summary['elo_diff'])} points {'higher' if feature_summary['elo_diff'] >= 0 else 'lower'} than {away_team}.",
        f"Recent win-form difference is {feature_summary['recent_win_diff']:+.1f} over the last five-match form window.",
        f"Recent goal-difference edge is {feature_summary['recent_goal_diff_diff']:+.1f}.",
        "The manual predictor averages both team orders, so swapping Team 1 and Team 2 mirrors the result instead of changing the matchup.",
    ]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "model_label": selected["label"],
        "model_type": MODEL_KIND_LABELS.get(selected["kind"], selected["kind"]),
        "prediction": predicted,
        "projected_advancer": advancing_team,
        "projected_score": scoreline["score"],
        "score_probability_pct": scoreline["score_probability_pct"],
        "top_scorelines": scoreline["top_scorelines"],
        "expected_goals": {
            "home": scoreline["home_expected_goals"],
            "away": scoreline["away_expected_goals"],
        },
        "probabilities": {
            "home_win_pct": round(home_win * 100, 1),
            "draw_pct": round(draw * 100, 1),
            "away_win_pct": round(away_win * 100, 1),
            "home_advance_pct": round(home_advance * 100, 1),
            "away_advance_pct": round(away_advance * 100, 1),
        },
        "score_model": relative_path(score_dir),
        "score_model_top3_pct": clean_value(score_metrics.get("eval_metrics", {}).get("top3_scoreline_accuracy")),
        "order_invariant": True,
        "feature_summary": feature_summary,
        "explanation": explanation,
    }


def predict_knockout_match(row: pd.Series, model, encoder, elo_lookup: dict[str, pd.Series], form_lookup: dict[str, pd.Series], score_models: tuple[Any, Any] | None) -> dict[str, Any]:
    home_team = clean_team_name(row["home_team"])
    away_team = clean_team_name(row["away_team"])
    features = match_features(home_team, away_team, clean_team_name(row.get("country")), elo_lookup, form_lookup)
    if features is None:
        return {"score_type": "Unavailable", "score": "", "advancing_team": "", "home_advance_pct": None, "away_advance_pct": None, "draw_pct": None, "confidence_pct": None, "top_scorelines": []}
    probabilities = model.predict_proba(features[FEATURE_COLS])[0]
    by_label = dict(zip(encoder.classes_, probabilities))
    home_win = float(by_label.get("H", 0))
    away_win = float(by_label.get("A", 0))
    draw = float(by_label.get("D", 0))
    home_advance = home_win + draw / 2
    away_advance = away_win + draw / 2
    advancing_team = home_team if home_advance >= away_advance else away_team
    scoreline = predict_scorelines(features, score_models, advancing_team, home_team)
    if scoreline is None:
        scoreline = {"score": fallback_projected_score(features, advancing_team, home_team), "score_probability_pct": None, "top_scorelines": []}
    return {
        "score_type": "Projected",
        "score": scoreline["score"],
        "advancing_team": advancing_team,
        "home_advance_pct": round(home_advance * 100, 1),
        "away_advance_pct": round(away_advance * 100, 1),
        "draw_pct": round(draw * 100, 1),
        "confidence_pct": round(max(home_advance, away_advance) * 100, 1),
        "score_probability_pct": scoreline.get("score_probability_pct"),
        "top_scorelines": scoreline.get("top_scorelines", []),
    }


def build_round_of_32_outlook(results: pd.DataFrame, model, encoder, score_models: tuple[Any, Any] | None) -> pd.DataFrame:
    if results.empty or "stage" not in results:
        return pd.DataFrame()
    r32 = results[results["stage"].astype(str).str.casefold().eq("round-of-32")].copy()
    if r32.empty:
        return pd.DataFrame()
    elo_lookup = team_lookup(read_csv(DATA_DIR / "elo_latest.csv"), "country")
    form_lookup = team_lookup(read_csv(DATA_DIR / "team_recent_form.csv"), "team")
    rows = []
    for _, row in r32.sort_values("date").iterrows():
        home_team = clean_team_name(row["home_team"])
        away_team = clean_team_name(row["away_team"])
        output = {
            "date": clean_value(row.get("date")),
            "match": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "status": clean_team_name(row.get("status")),
            "venue": clean_team_name(row.get("venue")),
            "city": clean_team_name(row.get("city")),
        }
        if truthy(row.get("is_finished")) or truthy(row.get("is_live")):
            score = ""
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                score = f"{int(row['home_score'])}-{int(row['away_score'])}"
            advancing = ""
            if truthy(row.get("is_finished")):
                advancing = home_team if truthy(row.get("home_winner")) else away_team if truthy(row.get("away_winner")) else ""
            output.update(
                {
                    "score_type": "Actual" if truthy(row.get("is_finished")) else "Live",
                    "score": score,
                    "advancing_team": advancing,
                    "home_advance_pct": 100.0 if advancing == home_team else None,
                    "away_advance_pct": 100.0 if advancing == away_team else None,
                    "draw_pct": None,
                    "confidence_pct": 100.0 if advancing else None,
                    "score_probability_pct": None,
                    "top_scorelines": [],
                }
            )
        else:
            output.update(predict_knockout_match(row, model, encoder, elo_lookup, form_lookup, score_models))
        rows.append(output)
    return pd.DataFrame(rows)


def stage_score(row: pd.Series) -> str:
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        return f"{int(row['home_score'])}-{int(row['away_score'])}"
    return ""


def placeholder_ref(team: str) -> tuple[str, int, str] | None:
    text = str(team or "").strip()
    patterns = [
        (r"^Round of 32 (\d+) (Winner|Loser)$", "round-of-32"),
        (r"^Round of 16 (\d+) (Winner|Loser)$", "round-of-16"),
        (r"^Quarterfinal (\d+) (Winner|Loser)$", "quarterfinals"),
        (r"^Semifinal (\d+) (Winner|Loser)$", "semifinals"),
    ]
    for pattern, stage in patterns:
        match = re.match(pattern, text)
        if match:
            return stage, int(match.group(1)), match.group(2).casefold()
    return None


def resolve_bracket_team(team: str, resolved: dict[tuple[str, int, str], str]) -> str:
    reference = placeholder_ref(team)
    if not reference:
        return team
    return resolved.get(reference, team)


def live_leader(row: pd.Series, home: str, away: str) -> str:
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return ""
    home_score = float(row.get("home_score"))
    away_score = float(row.get("away_score"))
    if home_score > away_score:
        return home
    if away_score > home_score:
        return away
    return ""


def bracket(
    results: pd.DataFrame,
    outlook: pd.DataFrame,
    model=None,
    encoder=None,
    score_models: tuple[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    if results.empty or "stage" not in results:
        return []
    knockout = results[results["stage"].isin(KNOCKOUT_STAGE_ORDER)].copy()
    r32_predictions = outlook.set_index("match").to_dict("index") if not outlook.empty else {}
    elo_lookup = team_lookup(read_csv(DATA_DIR / "elo_latest.csv"), "country")
    form_lookup = team_lookup(read_csv(DATA_DIR / "team_recent_form.csv"), "team")
    resolved: dict[tuple[str, int, str], str] = {}
    stages = []
    for stage in KNOCKOUT_STAGE_ORDER:
        stage_rows = knockout[knockout["stage"] == stage].sort_values("date")
        matches = []
        for match_index, (_, row) in enumerate(stage_rows.iterrows(), start=1):
            raw_home = clean_team_name(row["home_team"])
            raw_away = clean_team_name(row["away_team"])
            home = resolve_bracket_team(raw_home, resolved)
            away = resolve_bracket_team(raw_away, resolved)
            key = f"{home} vs {away}"
            score = stage_score(row)
            winner = ""
            loser = ""
            status = clean_team_name(row.get("status"))
            mode = "pending"
            top_scorelines = []
            home_advance_pct = None
            away_advance_pct = None
            confidence_pct = None
            score_probability_pct = None
            if truthy(row.get("is_finished")):
                mode = "actual"
                winner = home if truthy(row.get("home_winner")) else away if truthy(row.get("away_winner")) else ""
                if winner:
                    loser = away if winner == home else home
                    home_advance_pct = 100.0 if winner == home else 0.0
                    away_advance_pct = 100.0 if winner == away else 0.0
                    confidence_pct = 100.0
            elif truthy(row.get("is_live")):
                mode = "live"
                winner = live_leader(row, home, away)
                if winner:
                    loser = away if winner == home else home
                    home_advance_pct = 100.0 if winner == home else 0.0
                    away_advance_pct = 100.0 if winner == away else 0.0
                    confidence_pct = 100.0
                else:
                    simulated_row = row.copy()
                    simulated_row["home_team"] = home
                    simulated_row["away_team"] = away
                    pred = predict_knockout_match(simulated_row, model, encoder, elo_lookup, form_lookup, score_models) if model and encoder else {}
                    winner = pred.get("advancing_team") or ""
                    loser = away if winner == home else home if winner == away else ""
                    home_advance_pct = pred.get("home_advance_pct")
                    away_advance_pct = pred.get("away_advance_pct")
                    confidence_pct = pred.get("confidence_pct")
            elif key in r32_predictions:
                pred = r32_predictions[key]
                mode = "projected"
                score = pred.get("score") or ""
                winner = pred.get("advancing_team") or ""
                loser = away if winner == home else home if winner == away else ""
                status = "Projected"
                top_scorelines = pred.get("top_scorelines") or []
                home_advance_pct = pred.get("home_advance_pct")
                away_advance_pct = pred.get("away_advance_pct")
                confidence_pct = pred.get("confidence_pct")
                score_probability_pct = pred.get("score_probability_pct")
            elif (
                model
                and encoder
                and home
                and away
                and not is_placeholder_team(home)
                and not is_placeholder_team(away)
            ):
                simulated_row = row.copy()
                simulated_row["home_team"] = home
                simulated_row["away_team"] = away
                pred = predict_knockout_match(simulated_row, model, encoder, elo_lookup, form_lookup, score_models)
                if pred.get("advancing_team"):
                    mode = "projected"
                    score = pred.get("score") or ""
                    winner = pred.get("advancing_team") or ""
                    loser = away if winner == home else home if winner == away else ""
                    status = "Live simulation"
                    top_scorelines = pred.get("top_scorelines") or []
                    home_advance_pct = pred.get("home_advance_pct")
                    away_advance_pct = pred.get("away_advance_pct")
                    confidence_pct = pred.get("confidence_pct")
                    score_probability_pct = pred.get("score_probability_pct")
            if winner:
                resolved[(stage, match_index, "winner")] = winner
            if loser:
                resolved[(stage, match_index, "loser")] = loser
            matches.append(
                {
                    "date": clean_value(row.get("date")),
                    "home_team": home,
                    "away_team": away,
                    "raw_home_team": raw_home,
                    "raw_away_team": raw_away,
                    "score": score,
                    "winner": winner,
                    "status": status,
                    "mode": mode,
                    "simulation_note": "Live bracket simulation" if mode == "projected" else "",
                    "top_scorelines": top_scorelines,
                    "home_advance_pct": clean_value(home_advance_pct),
                    "away_advance_pct": clean_value(away_advance_pct),
                    "confidence_pct": clean_value(confidence_pct),
                    "score_probability_pct": clean_value(score_probability_pct),
                }
            )
        stages.append({"stage": stage, "label": KNOCKOUT_STAGE_LABELS[stage], "matches": matches})
    return stages


def simulation_favorite(simulation: pd.DataFrame) -> dict[str, Any]:
    if simulation.empty or "win_pct" not in simulation:
        return {"team": "N/A", "value": None}
    row = simulation.sort_values("win_pct", ascending=False).iloc[0]
    return {"team": row["team"], "value": round(float(row["win_pct"]), 2)}


def biggest_simulation_mover(simulation: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    if simulation.empty or baseline.empty or "win_pct" not in simulation or "win_pct" not in baseline:
        return {"team": "N/A", "delta": None}
    compare = simulation[["team", "win_pct"]].merge(
        baseline[["team", "win_pct"]].rename(columns={"win_pct": "baseline_win_pct"}),
        on="team",
        how="left",
    ).dropna(subset=["baseline_win_pct"])
    if compare.empty:
        return {"team": "N/A", "delta": None}
    compare["delta"] = compare["win_pct"] - compare["baseline_win_pct"]
    row = compare.iloc[compare["delta"].abs().argmax()]
    return {"team": row["team"], "delta": round(float(row["delta"]), 2)}


def next_knockout_pick(outlook: pd.DataFrame) -> dict[str, Any]:
    if outlook.empty:
        return {"team": "N/A", "detail": "No knockout outlook"}
    scheduled = outlook[outlook["score_type"] == "Projected"].copy()
    if scheduled.empty:
        return {"team": "N/A", "detail": "No scheduled Round of 32 matches"}
    row = scheduled.sort_values("date").iloc[0]
    detail = row["match"]
    if row.get("score"):
        detail = f"{detail} | {row['score']}"
    return {"team": row.get("advancing_team", "N/A"), "detail": detail}


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    out["score"] = out.apply(
        lambda row: ""
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"])
        else f"{int(row['home_score'])}-{int(row['away_score'])}",
        axis=1,
    )
    return out


def live_match_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "is_live" not in results:
        return pd.DataFrame()
    live = format_results(results[results["is_live"]].copy())
    wanted = [
        "date",
        "stage",
        "home_team",
        "away_team",
        "score",
        "status",
        "display_clock",
        "period",
        "venue",
        "city",
    ]
    return live[[column for column in wanted if column in live]].sort_values("date")


def upcoming_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "date" not in predictions:
        return predictions
    out = predictions.copy()
    dates = pd.to_datetime(out["date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return out[dates.isna() | (dates >= now)].copy()


def make_espn_scheduled_prediction_frame(results: pd.DataFrame, model, encoder) -> pd.DataFrame:
    if results.empty or "date" not in results:
        return pd.DataFrame()
    scheduled = results.copy()
    dates = pd.to_datetime(scheduled["date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    scheduled = scheduled[
        scheduled.get("is_scheduled", False).astype(bool)
        & (dates >= now)
    ].copy()
    if scheduled.empty:
        return pd.DataFrame()

    elo_lookup = team_lookup(read_csv(DATA_DIR / "elo_latest.csv"), "country")
    form_lookup = team_lookup(read_csv(DATA_DIR / "team_recent_form.csv"), "team")
    rows = []
    for _, row in scheduled.sort_values("date").iterrows():
        team1 = clean_team_name(row.get("home_team"))
        team2 = clean_team_name(row.get("away_team"))
        if not team1 or not team2 or is_placeholder_team(team1) or is_placeholder_team(team2):
            continue
        features = match_features(team1, team2, clean_team_name(row.get("country")), elo_lookup, form_lookup)
        if features is None:
            continue
        probabilities = result_probabilities(model, encoder, features)
        team1_win = probabilities["home_win"] * 100
        draw = probabilities["draw"] * 100
        team2_win = probabilities["away_win"] * 100
        choices = {
            team1: team1_win,
            "Draw": draw,
            team2: team2_win,
        }
        prediction = max(choices, key=choices.get)
        rows.append(
            {
                "date": clean_value(row.get("date")),
                "stage": clean_team_name(row.get("stage")),
                "team1": team1,
                "team2": team2,
                "prediction": prediction,
                "team1_win_pct": round(team1_win, 1),
                "draw_pct": round(draw, 1),
                "team2_win_pct": round(team2_win, 1),
                "confidence_pct": round(max(team1_win, draw, team2_win), 1),
                "venue": clean_team_name(row.get("venue")),
                "city": clean_team_name(row.get("city")),
                "country": clean_team_name(row.get("country")),
                "source": "espn_scheduled",
            }
        )
    return pd.DataFrame(rows)


def resolve_features_path(metrics: dict[str, Any], selected_kind: str) -> Path:
    raw = metrics.get("features_path")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = BASE_DIR / path
        if path.exists():
            return path
    if selected_kind == "live_result":
        for candidate in [MODEL_DIR / "training_features_live_result.csv", MODEL_DIR / "training_features_live_only.csv"]:
            if candidate.exists():
                return candidate
    return MODEL_DIR / "training_features_augmented.csv"


def dashboard(
    model_label: str | None,
    team: str = "All teams",
    metric: str = "win_pct",
    sort: str = "desc",
    top: int = 12,
    simulate_bracket: bool = False,
) -> dict[str, Any]:
    selected, model, encoder = model_bundle(model_label or default_model_label())
    score_models, score_metrics, score_dir = score_bundle()
    results = read_csv(DATA_DIR / "worldcup_2026_live_results.csv")
    fixtures = read_csv(DATA_DIR / "upcoming_match_features.csv")
    simulation = read_csv(selected["simulation"]) if selected["simulation"] else pd.DataFrame()
    baseline = read_csv(MODEL_DIR / "simulation_results.csv")
    metrics = read_json(selected["metrics"])
    quality = data_quality_report()
    features_path = resolve_features_path(metrics, selected["kind"])
    features = read_csv(features_path)
    fixture_predictions = upcoming_prediction_rows(make_prediction_frame(fixtures, model, encoder))
    espn_predictions = make_espn_scheduled_prediction_frame(results, model, encoder)
    predictions = pd.concat([espn_predictions, fixture_predictions], ignore_index=True)
    if not predictions.empty:
        predictions = predictions.drop_duplicates(subset=["date", "team1", "team2"], keep="first")
    outlook = build_round_of_32_outlook(results, model, encoder, score_models)
    formatted_results = format_results(results)
    if team and team != "All teams":
        predictions = predictions[(predictions.get("team1") == team) | (predictions.get("team2") == team)]
        formatted_results = formatted_results[(formatted_results.get("home_team") == team) | (formatted_results.get("away_team") == team)]
        outlook = outlook[(outlook.get("home_team") == team) | (outlook.get("away_team") == team)]
    ranking = simulation.copy()
    if not ranking.empty and not baseline.empty and "team" in baseline and "win_pct" in baseline:
        ranking = ranking.merge(
            baseline[["team", "win_pct"]].rename(columns={"win_pct": "baseline_win_pct"}),
            on="team",
            how="left",
        )
        ranking["win_delta_pct"] = (ranking["win_pct"] - ranking["baseline_win_pct"]).round(2)
    if not ranking.empty and metric in ranking:
        ranking = ranking.sort_values(metric, ascending=(sort == "asc")).head(max(1, min(int(top), 48)))
    stage_cols = ["r32_pct", "r16_pct", "quarter_pct", "semi_pct", "final_pct", "win_pct"]
    stage_series = []
    if not ranking.empty:
        for _, row in ranking.head(8).iterrows():
            for stage in stage_cols:
                if stage in row:
                    stage_series.append({"team": row["team"], "stage": stage.replace("_pct", ""), "pct": round(float(row[stage]), 2)})
    status_counts = {}
    if not results.empty:
        for key in ["is_finished", "is_live", "is_scheduled"]:
            if key in results:
                status_counts[key] = int(results[key].sum())
    result_mix = {}
    if not features.empty and "target" in features:
        result_mix = features["target"].value_counts().to_dict()
    importance = metrics.get("feature_importance", {})
    importance_rows = [
        {"feature": name, "importance": round(float(value), 4)}
        for name, value in sorted(importance.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "selected_model": {
            "label": selected["label"],
            "run_id": selected["run_id"],
            "kind": selected["kind"],
            "kind_label": MODEL_KIND_LABELS.get(selected["kind"], selected["kind"]),
            "model_path": relative_path(selected["model"]),
            "simulation_path": relative_path(selected["simulation"]),
            "features_path": relative_path(features_path),
            "score_model_path": relative_path(score_dir),
        },
        "snapshot": latest_fetch_label(results),
        "metrics": {
            "finished_matches": status_counts.get("is_finished", 0),
            "live_matches": status_counts.get("is_live", 0),
            "scheduled_matches": status_counts.get("is_scheduled", 0),
            "teams_simulated": int(len(simulation)),
            "training_rows": clean_value(metrics.get("training_rows")),
            "eval_accuracy": clean_value(metrics.get("eval_accuracy")),
            "in_sample_accuracy": clean_value(metrics.get("in_sample_accuracy")),
            "score_top3_accuracy": clean_value(score_metrics.get("eval_metrics", {}).get("top3_scoreline_accuracy")),
        },
        "glance": {
            "favorite": simulation_favorite(simulation),
            "biggest_mover": biggest_simulation_mover(simulation, baseline),
            "next_knockout_pick": next_knockout_pick(outlook),
        },
        "simulation": records(ranking),
        "stage_series": stage_series,
        "round_of_32": records(outlook),
        "live_matches": records(live_match_rows(results)),
        "bracket": bracket(
            results,
            outlook if simulate_bracket else pd.DataFrame(),
            model if simulate_bracket else None,
            encoder if simulate_bracket else None,
            score_models if simulate_bracket else None,
        ),
        "bracket_mode": "simulation" if simulate_bracket else "live",
        "predictions": records(predictions.sort_values("date") if "date" in predictions else predictions, limit=80),
        "results": records(formatted_results.sort_values("date") if "date" in formatted_results else formatted_results, limit=104),
        "data_health": {
            "espn_rows": int(len(results)),
            "fixture_rows": int(len(fixtures)),
            "feature_rows": int(len(features)),
            "result_mix": result_mix,
            "source_snapshot": latest_fetch_label(results),
            "quality_status": quality["status"],
            "quality_summary": quality["summary"],
        },
        "model_registry": model_registry(),
        "model_diagnostics": {
            "feature_importance": importance_rows[:12],
            "warning": metrics.get("warning"),
            "eval_report": metrics.get("eval_report", {}),
        },
    }


def options_payload() -> dict[str, Any]:
    options = artifact_options()
    return {
        "models": [
            {
                "label": label,
                "kind": value["kind"],
                "kind_label": MODEL_KIND_LABELS.get(value["kind"], value["kind"]),
                "run_id": value["run_id"],
            }
            for label, value in options.items()
        ],
        "default_model": default_model_label(),
        "teams": ["All teams", *all_teams()],
        "predictor_teams": predictor_teams(),
        "team_codes": team_code_map(),
        "ranking_metrics": [
            {"label": "Win tournament", "value": "win_pct"},
            {"label": "Reach final", "value": "final_pct"},
            {"label": "Reach semi-final", "value": "semi_pct"},
            {"label": "Reach quarter-final", "value": "quarter_pct"},
            {"label": "Reach round of 16", "value": "r16_pct"},
            {"label": "Reach round of 32", "value": "r32_pct"},
        ],
    }


def backtest(model_label: str | None, source: str = "selected", limit: int = 25) -> dict[str, Any]:
    label = model_label or default_model_label()
    if source == "live":
        live_labels = [item["label"] for item in options_payload()["models"] if item["kind"] == "live_result"]
        if live_labels:
            label = live_labels[0]
    elif source == "augmented":
        augmented_labels = [item["label"] for item in options_payload()["models"] if item["kind"] in {"augmented", "historical_live"}]
        if augmented_labels:
            label = augmented_labels[0]
    selected, model, encoder = model_bundle(label)
    metrics = read_json(selected["metrics"])
    features_path = resolve_features_path(metrics, selected["kind"])
    df = read_csv(features_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("No feature file is available for this model.")
    df = df.dropna(subset=[*FEATURE_COLS, "target"]).copy()
    proba = model.predict_proba(df[FEATURE_COLS])
    pred_labels = encoder.inverse_transform(proba.argmax(axis=1))
    df["prediction"] = pred_labels
    df["confidence_pct"] = (proba.max(axis=1) * 100).round(1)
    df["correct"] = df["prediction"] == df["target"]
    accuracy = float(df["correct"].mean()) if len(df) else 0.0
    confusion = (
        df.groupby(["target", "prediction"]).size().reset_index(name="matches").sort_values(["target", "matches"], ascending=[True, False])
    )
    sample_cols = ["date", "home_team", "away_team", "target", "prediction", "confidence_pct", "correct"]
    sample = df.sort_values("date", ascending=False)[sample_cols].head(max(1, min(limit, 100))).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    return {
        "model_label": selected["label"],
        "model_type": MODEL_KIND_LABELS.get(selected["kind"], selected["kind"]),
        "source": source,
        "features_path": relative_path(features_path),
        "rows": int(len(df)),
        "accuracy_pct": round(accuracy * 100, 1),
        "eval_accuracy_pct": round(float(metrics["eval_accuracy"]) * 100, 1) if metrics.get("eval_accuracy") is not None else None,
        "note": "This retrospective check scores saved matches against the selected final model. Use eval accuracy for the cleaner train/test estimate.",
        "confusion": records(confusion),
        "sample": records(sample),
    }
