from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import live_store


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "src" / "modeling"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
SIMULATION_RUNS_DIR = MODEL_DIR / "simulation_runs"
SCORE_ARTIFACTS_DIR = MODEL_DIR / "score_artifacts"


def file_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": relative_path(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        if exists
        else None,
    }


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if path.name == "worldcup_2026_live_results.csv" and live_store.is_enabled():
        try:
            db_results = live_store.load_live_results()
            if not db_results.empty:
                return db_results
        except Exception:
            pass
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def status_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(not item["passed"] and item["severity"] == "error" for item in checks):
        return "fail"
    if any(not item["passed"] for item in checks):
        return "warn"
    return "pass"


def validate_live_results(results: pd.DataFrame) -> list[dict[str, Any]]:
    required = {
        "date",
        "stage",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "is_finished",
        "is_live",
        "is_scheduled",
    }
    checks = [
        check(
            "live_results_present",
            not results.empty,
            f"{len(results)} ESPN rows loaded.",
        ),
        check(
            "live_results_schema",
            required.issubset(results.columns),
            f"Required columns missing: {sorted(required.difference(results.columns))}",
        ),
    ]
    if results.empty or not required.issubset(results.columns):
        return checks

    complete_key_cols = ["date", "home_team", "away_team"]
    duplicate_count = int(results.duplicated(subset=complete_key_cols).sum())
    checks.append(
        check(
            "no_duplicate_matches",
            duplicate_count == 0,
            f"{duplicate_count} duplicate rows by date/home/away.",
        )
    )

    score_cols = results[["home_score", "away_score"]]
    negative_scores = int((score_cols.fillna(0) < 0).any(axis=1).sum())
    checks.append(check("no_negative_scores", negative_scores == 0, f"{negative_scores} rows have negative scores."))

    status_sum = results[["is_finished", "is_live", "is_scheduled"]].fillna(False).astype(bool).sum(axis=1)
    bad_status_rows = int((status_sum != 1).sum())
    checks.append(
        check(
            "single_match_status",
            bad_status_rows == 0,
            f"{bad_status_rows} rows are not exactly one of finished/live/scheduled.",
        )
    )

    scheduled = results["is_scheduled"].fillna(False).astype(bool)
    scheduled_with_score = int(results.loc[scheduled, ["home_score", "away_score"]].notna().any(axis=1).sum())
    checks.append(
        check(
            "scheduled_matches_unscored",
            scheduled_with_score == 0,
            f"{scheduled_with_score} scheduled rows already have a score.",
            severity="warning",
        )
    )

    placeholder_teams = int(
        results["home_team"].astype(str).str.contains("Winner|Loser|Round of", case=False, regex=True).sum()
        + results["away_team"].astype(str).str.contains("Winner|Loser|Round of", case=False, regex=True).sum()
    )
    checks.append(
        check(
            "placeholder_teams_only_expected",
            placeholder_teams <= 48,
            f"{placeholder_teams} placeholder team references found.",
            severity="warning",
        )
    )
    return checks


def validate_training_features(features: pd.DataFrame, name: str) -> list[dict[str, Any]]:
    required = {
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "elo_diff",
        "neutral",
        "target",
    }
    checks = [
        check(f"{name}_present", not features.empty, f"{len(features)} feature rows loaded."),
        check(
            f"{name}_schema",
            required.issubset(features.columns),
            f"Required columns missing: {sorted(required.difference(features.columns))}",
        ),
    ]
    if features.empty or not required.issubset(features.columns):
        return checks

    required_na = int(features[list(required)].isna().any(axis=1).sum())
    checks.append(check(f"{name}_required_values", required_na == 0, f"{required_na} rows have missing required values."))

    bad_targets = sorted(set(features["target"].dropna()) - {"H", "D", "A"})
    checks.append(check(f"{name}_target_values", not bad_targets, f"Unexpected target labels: {bad_targets}."))

    neutral_values = sorted(set(features["neutral"].dropna().astype(int)))
    checks.append(
        check(
            f"{name}_neutral_binary",
            set(neutral_values).issubset({0, 1}),
            f"Neutral column values: {neutral_values}.",
        )
    )
    return checks


def validate_artifacts() -> list[dict[str, Any]]:
    expected = [
        ARTIFACTS_DIR / "historical_live" / "model.pkl",
        ARTIFACTS_DIR / "historical_live" / "label_encoder.pkl",
        ARTIFACTS_DIR / "historical_live" / "metrics.json",
        ARTIFACTS_DIR / "live_result" / "model.pkl",
        ARTIFACTS_DIR / "live_result" / "label_encoder.pkl",
        ARTIFACTS_DIR / "live_result" / "metrics.json",
        SCORE_ARTIFACTS_DIR / "exact_score" / "home_goal_model.pkl",
        SCORE_ARTIFACTS_DIR / "exact_score" / "away_goal_model.pkl",
        SCORE_ARTIFACTS_DIR / "exact_score" / "metrics.json",
        SIMULATION_RUNS_DIR / "simulation_results_historical_live.csv",
        SIMULATION_RUNS_DIR / "simulation_results_live_result.csv",
    ]
    return [
        check(f"artifact_exists:{relative_path(path)}", path.exists(), f"{relative_path(path)} exists={path.exists()}.")
        for path in expected
    ]


def data_quality_report() -> dict[str, Any]:
    live_results = read_csv(DATA_DIR / "worldcup_2026_live_results.csv")
    historical_live = read_csv(MODEL_DIR / "training_features_augmented.csv")
    live_result = read_csv(MODEL_DIR / "training_features_live_result.csv")

    sections = {
        "live_results": validate_live_results(live_results),
        "historical_live_features": validate_training_features(historical_live, "historical_live_features"),
        "live_result_features": validate_training_features(live_result, "live_result_features"),
        "artifacts": validate_artifacts(),
    }
    all_checks = [item for checks in sections.values() for item in checks]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status_from_checks(all_checks),
        "summary": {
            "checks": len(all_checks),
            "passed": sum(1 for item in all_checks if item["passed"]),
            "warnings": sum(1 for item in all_checks if not item["passed"] and item["severity"] == "warning"),
            "errors": sum(1 for item in all_checks if not item["passed"] and item["severity"] == "error"),
        },
        "files": {
            "live_results": file_info(DATA_DIR / "worldcup_2026_live_results.csv"),
            "historical_live_features": file_info(MODEL_DIR / "training_features_augmented.csv"),
            "live_result_features": file_info(MODEL_DIR / "training_features_live_result.csv"),
        },
        "storage": {
            "live_results_source": "postgresql" if live_store.is_enabled() else "csv",
            "database_configured": live_store.is_enabled(),
        },
        "sections": sections,
    }
