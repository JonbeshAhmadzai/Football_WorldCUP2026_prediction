import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = Path(__file__).parent / "training_features.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_recent_wins", "home_recent_draws", "home_recent_losses",
    "home_recent_goals_for", "home_recent_goals_against", "home_recent_goal_diff",
    "away_recent_wins", "away_recent_draws", "away_recent_losses",
    "away_recent_goals_for", "away_recent_goals_against", "away_recent_goal_diff",
    "recent_win_diff", "recent_goal_diff_diff",
    "neutral",
]

EVAL_SPLIT_YEAR = 2018

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)


def load_data(features_path=FEATURES_PATH):
    df = pd.read_csv(features_path, parse_dates=["date"])
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["target"])  # A=0, D=1, H=2
    return df, le


def temporal_split(df):
    train = df[df["date"].dt.year < EVAL_SPLIT_YEAR]
    test = df[df["date"].dt.year >= EVAL_SPLIT_YEAR]
    return train, test


def train_model(df, params=None):
    if params is None:
        params = XGB_PARAMS
    X = df[FEATURE_COLS]
    y = df["label"]
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def evaluate(model, df, le):
    X = df[FEATURE_COLS]
    y_true = df["label"]
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}  ({len(df)} samples)")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    report = classification_report(
        y_true,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    return acc, report


def feature_importance(model):
    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importance.to_string())
    return importance


def save_training_artifacts(model, le, metrics, run_id, update_latest=False):
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    versioned_model_path = run_dir / "model.pkl"
    versioned_encoder_path = run_dir / "label_encoder.pkl"
    versioned_metrics_path = run_dir / "metrics.json"

    if update_latest and MODEL_PATH.exists():
        previous_model_path = run_dir / "previous_latest_model.pkl"
        previous_model_path.write_bytes(MODEL_PATH.read_bytes())
        print(f"Backed up previous latest model → {previous_model_path}")
    if update_latest and ENCODER_PATH.exists():
        previous_encoder_path = run_dir / "previous_latest_label_encoder.pkl"
        previous_encoder_path.write_bytes(ENCODER_PATH.read_bytes())
        print(f"Backed up previous latest encoder → {previous_encoder_path}")

    with open(versioned_model_path, "wb") as f:
        pickle.dump(model, f)
    with open(versioned_encoder_path, "wb") as f:
        pickle.dump(le, f)
    versioned_metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model → {versioned_model_path}")
    print(f"Saved encoder → {versioned_encoder_path}")
    print(f"Saved metrics → {versioned_metrics_path}")

    if update_latest:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(le, f)
        print(f"Updated latest model → {MODEL_PATH}")
        print(f"Updated latest encoder → {ENCODER_PATH}")
    else:
        print("Latest model files were not updated.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURES_PATH,
        help="CSV feature file used for training.",
    )
    parser.add_argument(
        "--artifact-label",
        default="baseline",
        help="Label added to the versioned training artifact folder.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Also update src/modeling/model.pkl and label_encoder.pkl.",
    )
    parser.add_argument(
        "--overwrite-artifact",
        action="store_true",
        help="Write to src/modeling/artifacts/<artifact-label> instead of a timestamped folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.artifact_label if args.overwrite_artifact else f"{args.artifact_label}_{timestamp}"
    df, le = load_data(args.features_path)

    train_df, test_df = temporal_split(df)
    print(f"Train: {len(train_df)} rows (pre-{EVAL_SPLIT_YEAR})")
    print(f"Test:  {len(test_df)} rows ({EVAL_SPLIT_YEAR}+)\n")

    print("--- Eval model (pre-2018 train / post-2018 test) ---")
    eval_model = train_model(train_df)
    eval_accuracy, eval_report = evaluate(eval_model, test_df, le)
    importance = feature_importance(eval_model)

    print(f"\n--- Final model (all {len(df)} rows) ---")
    final_model = train_model(df)
    print("Retrained on full dataset.")

    # Sanity check: in-sample accuracy on full data
    print("\nIn-sample accuracy (not meaningful for eval, just a sanity check):")
    in_sample_accuracy, in_sample_report = evaluate(final_model, df, le)

    metrics = {
        "run_id": run_id,
        "last_trained_at": timestamp,
        "features_path": str(args.features_path),
        "artifact_label": args.artifact_label,
        "training_rows": len(df),
        "temporal_split_year": EVAL_SPLIT_YEAR,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "eval_accuracy": eval_accuracy,
        "in_sample_accuracy": in_sample_accuracy,
        "eval_report": eval_report,
        "in_sample_report": in_sample_report,
        "feature_importance": importance.to_dict(),
        "feature_columns": FEATURE_COLS,
        "xgb_params": XGB_PARAMS,
    }
    save_training_artifacts(final_model, le, metrics, run_id, args.update_latest)


if __name__ == "__main__":
    main()
