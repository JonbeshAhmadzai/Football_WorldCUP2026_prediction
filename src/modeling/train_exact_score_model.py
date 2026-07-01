from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from data_prep import CUTOFF_YEAR, NAME_MAP
from eval_model_training import FEATURE_COLS


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = Path(__file__).parent
MATCHES_PATH = PROCESSED_DIR / "matches_augmented.csv"
ELO_PATH = RAW_DIR / "elo_ratings_wc2026.csv"
FEATURES_PATH = MODEL_DIR / "training_features_exact_score_augmented.csv"
SCORE_ARTIFACTS_DIR = MODEL_DIR / "score_artifacts"
EVAL_SPLIT_YEAR = 2018
MAX_SCORE = 6

XGB_GOAL_PARAMS = dict(
    objective="count:poisson",
    n_estimators=350,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.9,
    eval_metric="poisson-nloglik",
    random_state=42,
)


def summarize_last_matches(last_matches):
    if not last_matches:
        return {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
        }

    wins = sum(1 for match in last_matches if match["result"] == "W")
    draws = sum(1 for match in last_matches if match["result"] == "D")
    losses = sum(1 for match in last_matches if match["result"] == "L")
    goals_for = sum(match["goals_for"] for match in last_matches)
    goals_against = sum(match["goals_against"] for match in last_matches)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
    }


def target_from_score(home_score, away_score):
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def build_recent_score_features(matches):
    matches = matches.sort_values("date").reset_index(drop=True)
    team_history = {}
    rows = []

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        team_history.setdefault(home, [])
        team_history.setdefault(away, [])

        home_stats = summarize_last_matches(team_history[home][-5:])
        away_stats = summarize_last_matches(team_history[away][-5:])
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        rows.append(
            {
                "date": match["date"],
                "home_team": home,
                "away_team": away,
                "home_recent_wins": home_stats["wins"],
                "home_recent_draws": home_stats["draws"],
                "home_recent_losses": home_stats["losses"],
                "home_recent_goals_for": home_stats["goals_for"],
                "home_recent_goals_against": home_stats["goals_against"],
                "home_recent_goal_diff": home_stats["goal_diff"],
                "away_recent_wins": away_stats["wins"],
                "away_recent_draws": away_stats["draws"],
                "away_recent_losses": away_stats["losses"],
                "away_recent_goals_for": away_stats["goals_for"],
                "away_recent_goals_against": away_stats["goals_against"],
                "away_recent_goal_diff": away_stats["goal_diff"],
                "recent_win_diff": home_stats["wins"] - away_stats["wins"],
                "recent_goal_diff_diff": home_stats["goal_diff"] - away_stats["goal_diff"],
                "neutral": match["neutral"],
                "tournament": match["tournament"],
                "target": target_from_score(home_score, away_score),
                "home_score": home_score,
                "away_score": away_score,
            }
        )

        home_result = target_from_score(home_score, away_score)
        away_result = target_from_score(away_score, home_score)
        team_history[home].append(
            {
                "result": "W" if home_result == "H" else "L" if home_result == "A" else "D",
                "goals_for": home_score,
                "goals_against": away_score,
            }
        )
        team_history[away].append(
            {
                "result": "W" if away_result == "H" else "L" if away_result == "A" else "D",
                "goals_for": away_score,
                "goals_against": home_score,
            }
        )

    return pd.DataFrame(rows)


def build_score_training_features(output_path=FEATURES_PATH):
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH)

    matches = matches[
        matches["home_score"].notna()
        & matches["away_score"].notna()
        & matches["home_team"].notna()
        & matches["away_team"].notna()
    ].copy()
    matches["home_team"] = matches["home_team"].replace(NAME_MAP)
    matches["away_team"] = matches["away_team"].replace(NAME_MAP)
    matches = matches[matches["date"].dt.year >= CUTOFF_YEAR]

    elo_teams = set(elo["country"].unique())
    matches = matches[
        matches["home_team"].isin(elo_teams) & matches["away_team"].isin(elo_teams)
    ].copy()

    features = build_recent_score_features(matches)
    features["year"] = features["date"].dt.year
    elo_lookup = elo[["year", "country", "rating"]].rename(columns={"rating": "elo"})
    features = features.merge(
        elo_lookup.rename(columns={"country": "home_team", "elo": "home_elo"}),
        on=["year", "home_team"],
        how="left",
    )
    features = features.merge(
        elo_lookup.rename(columns={"country": "away_team", "elo": "away_elo"}),
        on=["year", "away_team"],
        how="left",
    )
    features = features.dropna(subset=["home_elo", "away_elo"]).copy()
    features["elo_diff"] = features["home_elo"] - features["away_elo"]
    features["neutral"] = features["neutral"].astype(int)

    output_cols = [
        "date",
        "home_team",
        "away_team",
        *FEATURE_COLS,
        "target",
        "home_score",
        "away_score",
    ]
    features = features[output_cols].reset_index(drop=True)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} score-training rows to {output_path}")
    return features


def train_goal_model(df, target_col):
    model = xgb.XGBRegressor(**XGB_GOAL_PARAMS)
    model.fit(df[FEATURE_COLS], df[target_col])
    return model


def poisson_pmf(lam, goal):
    lam = max(float(lam), 0.05)
    return math.exp(-lam) * (lam**goal) / math.factorial(goal)


def scoreline_probabilities(home_lambda, away_lambda, max_score=MAX_SCORE):
    rows = []
    for home_score in range(max_score + 1):
        for away_score in range(max_score + 1):
            probability = poisson_pmf(home_lambda, home_score) * poisson_pmf(
                away_lambda, away_score
            )
            rows.append((home_score, away_score, probability))

    total = sum(probability for _, _, probability in rows)
    return [
        (home_score, away_score, probability / total)
        for home_score, away_score, probability in rows
    ]


def scoreline_metrics(home_model, away_model, df):
    home_lambda = home_model.predict(df[FEATURE_COLS]).clip(0.05, MAX_SCORE)
    away_lambda = away_model.predict(df[FEATURE_COLS]).clip(0.05, MAX_SCORE)
    rounded_home = home_lambda.round().clip(0, MAX_SCORE).astype(int)
    rounded_away = away_lambda.round().clip(0, MAX_SCORE).astype(int)

    exact_match = (
        (rounded_home == df["home_score"].to_numpy())
        & (rounded_away == df["away_score"].to_numpy())
    ).mean()

    top3_hits = []
    for index, row in df.reset_index(drop=True).iterrows():
        probabilities = sorted(
            scoreline_probabilities(home_lambda[index], away_lambda[index]),
            key=lambda item: item[2],
            reverse=True,
        )
        actual = (int(row["home_score"]), int(row["away_score"]))
        top3_hits.append(any((home, away) == actual for home, away, _ in probabilities[:3]))

    return {
        "home_goal_mae": mean_absolute_error(df["home_score"], home_lambda),
        "away_goal_mae": mean_absolute_error(df["away_score"], away_lambda),
        "rounded_exact_score_accuracy": float(exact_match),
        "top3_scoreline_accuracy": float(sum(top3_hits) / len(top3_hits)),
    }


def save_artifacts(home_model, away_model, metrics, run_id):
    run_dir = SCORE_ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "home_goal_model.pkl", "wb") as f:
        pickle.dump(home_model, f)
    with open(run_dir / "away_goal_model.pkl", "wb") as f:
        pickle.dump(away_model, f)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved exact score artifact -> {run_dir}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = "exact_score"

    df = build_score_training_features()
    train_df = df[df["date"].dt.year < EVAL_SPLIT_YEAR]
    test_df = df[df["date"].dt.year >= EVAL_SPLIT_YEAR]

    print(f"Train: {len(train_df)} rows (pre-{EVAL_SPLIT_YEAR})")
    print(f"Test:  {len(test_df)} rows ({EVAL_SPLIT_YEAR}+)")

    eval_home_model = train_goal_model(train_df, "home_score")
    eval_away_model = train_goal_model(train_df, "away_score")
    eval_metrics = scoreline_metrics(eval_home_model, eval_away_model, test_df)

    final_home_model = train_goal_model(df, "home_score")
    final_away_model = train_goal_model(df, "away_score")
    in_sample_metrics = scoreline_metrics(final_home_model, final_away_model, df)

    metrics = {
        "run_id": run_id,
        "last_trained_at": timestamp,
        "artifact_label": "exact_score_augmented",
        "features_path": str(FEATURES_PATH),
        "training_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "temporal_split_year": EVAL_SPLIT_YEAR,
        "max_score_grid": MAX_SCORE,
        "eval_metrics": eval_metrics,
        "in_sample_metrics": in_sample_metrics,
        "feature_columns": FEATURE_COLS,
        "xgb_goal_params": XGB_GOAL_PARAMS,
        "note": "Goal models estimate expected goals; scoreline probabilities use an independent Poisson grid.",
    }
    save_artifacts(final_home_model, final_away_model, metrics, run_id)


if __name__ == "__main__":
    main()
