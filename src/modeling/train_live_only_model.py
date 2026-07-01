from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from eval_model_training import FEATURE_COLS, save_training_artifacts


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = Path(__file__).parent
LIVE_RESULTS_PATH = PROCESSED_DIR / "worldcup_2026_live_results.csv"
ELO_PATH = PROCESSED_DIR / "elo_latest.csv"
FORM_PATH = PROCESSED_DIR / "team_recent_form.csv"
OUTPUT_PATH = MODEL_DIR / "training_features_live_result.csv"

NAME_MAP = {
    "USA": "United States",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Congo DR": "DR Congo",
}

DEFAULT_FORM = {
    "wins": 2,
    "draws": 1,
    "losses": 2,
    "goals_for": 5,
    "goals_against": 5,
    "goal_diff": 0,
}

LIVE_ONLY_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=80,
    max_depth=2,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42,
)


def normalize_team(team):
    return NAME_MAP.get(team, team)


def target_from_score(row):
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def build_form_lookup(form_df):
    lookup = {}
    for _, row in form_df.iterrows():
        team = normalize_team(row["team"])
        lookup[team] = {
            "wins": row["last5_wins"],
            "draws": row["last5_draws"],
            "losses": row["last5_losses"],
            "goals_for": row["last5_goals_for"],
            "goals_against": row["last5_goals_against"],
            "goal_diff": row["last5_goal_difference"],
        }
    return lookup


def form_values(form_lookup, team):
    return form_lookup.get(team, DEFAULT_FORM)


def build_live_only_features(output_path=OUTPUT_PATH):
    results = pd.read_csv(LIVE_RESULTS_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH)
    form = pd.read_csv(FORM_PATH)

    finished = results[
        results["is_finished"]
        & results["home_score"].notna()
        & results["away_score"].notna()
    ].copy()

    finished["home_team"] = finished["home_team"].map(normalize_team)
    finished["away_team"] = finished["away_team"].map(normalize_team)
    finished["target"] = finished.apply(target_from_score, axis=1)
    finished["neutral"] = (
        finished["country"].fillna("").str.casefold()
        != finished["home_team"].fillna("").str.casefold()
    ).astype(int)

    elo_lookup = dict(zip(elo["country"].map(normalize_team), elo["rating"]))
    form_lookup = build_form_lookup(form)

    rows = []
    for _, row in finished.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = elo_lookup.get(home)
        away_elo = elo_lookup.get(away)
        if pd.isna(home_elo) or pd.isna(away_elo):
            continue

        home_form = form_values(form_lookup, home)
        away_form = form_values(form_lookup, away)
        rows.append(
            {
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_recent_wins": home_form["wins"],
                "home_recent_draws": home_form["draws"],
                "home_recent_losses": home_form["losses"],
                "home_recent_goals_for": home_form["goals_for"],
                "home_recent_goals_against": home_form["goals_against"],
                "home_recent_goal_diff": home_form["goal_diff"],
                "away_recent_wins": away_form["wins"],
                "away_recent_draws": away_form["draws"],
                "away_recent_losses": away_form["losses"],
                "away_recent_goals_for": away_form["goals_for"],
                "away_recent_goals_against": away_form["goals_against"],
                "away_recent_goal_diff": away_form["goal_diff"],
                "recent_win_diff": home_form["wins"] - away_form["wins"],
                "recent_goal_diff_diff": home_form["goal_diff"] - away_form["goal_diff"],
                "neutral": row["neutral"],
                "target": row["target"],
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} live result rows to {output_path}")
    print(features["target"].value_counts().to_string())
    return features


def train_model(df, params=None):
    params = params or LIVE_ONLY_PARAMS
    model = xgb.XGBClassifier(**params)
    model.fit(df[FEATURE_COLS], df["label"])
    return model


def evaluate(model, df, le):
    pred = model.predict(df[FEATURE_COLS])
    acc = accuracy_score(df["label"], pred)
    report = classification_report(
        df["label"],
        pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    print(f"Accuracy: {acc:.3f} ({len(df)} samples)")
    print(classification_report(df["label"], pred, target_names=le.classes_, zero_division=0))
    return acc, report


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = "live_result"

    df = build_live_only_features()
    if len(df) < 20:
        raise ValueError("Not enough finished live matches to train a live result model.")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["target"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    print(f"Train: {len(train_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    eval_model = train_model(train_df)
    eval_accuracy, eval_report = evaluate(eval_model, test_df, le)

    final_model = train_model(df)
    in_sample_accuracy, in_sample_report = evaluate(final_model, df, le)
    importance = pd.Series(
        final_model.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    metrics = {
        "run_id": run_id,
        "last_trained_at": timestamp,
        "features_path": str(OUTPUT_PATH),
        "artifact_label": "live_result",
        "training_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "eval_accuracy": eval_accuracy,
        "in_sample_accuracy": in_sample_accuracy,
        "eval_report": eval_report,
        "in_sample_report": in_sample_report,
        "feature_importance": importance.to_dict(),
        "feature_columns": FEATURE_COLS,
        "xgb_params": LIVE_ONLY_PARAMS,
        "warning": "Live result model is trained on current tournament matches only and is expected to be unstable.",
    }

    save_training_artifacts(final_model, le, metrics, run_id, update_latest=False)


if __name__ == "__main__":
    main()
