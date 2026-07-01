from __future__ import annotations

import os
import shlex
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = os.environ.get(
    "WORLDCUP_PYTHON",
    str(REPO_ROOT / ".venv" / "bin" / "python"),
)

Q_REPO_ROOT = shlex.quote(str(REPO_ROOT))
Q_PYTHON_BIN = shlex.quote(PYTHON_BIN)


def repo_command(command: str) -> str:
    return f"cd {Q_REPO_ROOT} && {command}"


default_args = {
    "owner": "football-prediction",
    "retries": 1,
}


with DAG(
    dag_id="worldcup_2026_augmented_pipeline",
    default_args=default_args,
    description="Scrape ESPN results and overwrite current historical-live and live-result models.",
    start_date=datetime(2026, 6, 11),
    schedule="@hourly",
    catchup=False,
    tags=["world-cup", "football", "historical-live-model", "live-result-model"],
) as dag:
    scrape_latest_results = BashOperator(
        task_id="scrape_latest_results",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/scraping/scrape_latest_results.py"
        ),
    )

    validate_data_after_scrape = BashOperator(
        task_id="validate_data_after_scrape",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} scripts/validate_project_data.py "
            "--output reports/data_quality_latest.json"
        ),
    )

    build_augmented_matches = BashOperator(
        task_id="build_augmented_matches",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/transform/build_augmented_matches.py"
        ),
    )

    build_training_dataset_augmented = BashOperator(
        task_id="build_training_dataset_augmented",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/transform/build_training_dataset_augmented.py"
        ),
    )

    load_database = BashOperator(
        task_id="load_database",
        bash_command=repo_command(f"{Q_PYTHON_BIN} src/load/load_to_database.py"),
    )

    build_features_augmented = BashOperator(
        task_id="build_features_augmented",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/data_prep_augmented.py"
        ),
    )

    train_augmented_model = BashOperator(
        task_id="train_augmented_model",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/eval_model_training.py "
            "--features-path src/modeling/training_features_augmented.csv "
            "--artifact-label historical_live "
            "--overwrite-artifact"
        ),
    )

    train_exact_score_model = BashOperator(
        task_id="train_exact_score_model",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/train_exact_score_model.py"
        ),
    )

    run_augmented_simulation = BashOperator(
        task_id="run_augmented_simulation",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/world_cup_montecarlo_simulation.py "
            "--model-path src/modeling/artifacts/historical_live/model.pkl "
            "--encoder-path src/modeling/artifacts/historical_live/label_encoder.pkl "
            "--artifact-label historical_live "
            "--overwrite-output"
        ),
    )

    train_live_result_model = BashOperator(
        task_id="train_live_result_model",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/train_live_only_model.py"
        ),
    )

    run_live_result_simulation = BashOperator(
        task_id="run_live_result_simulation",
        bash_command=repo_command(
            f"{Q_PYTHON_BIN} src/modeling/world_cup_montecarlo_simulation.py "
            "--model-path src/modeling/artifacts/live_result/model.pkl "
            "--encoder-path src/modeling/artifacts/live_result/label_encoder.pkl "
            "--artifact-label live_result "
            "--overwrite-output"
        ),
    )

    scrape_latest_results >> validate_data_after_scrape >> build_augmented_matches

    (
        build_augmented_matches
        >> build_training_dataset_augmented
        >> load_database
        >> build_features_augmented
        >> train_augmented_model
        >> run_augmented_simulation
    )

    build_features_augmented >> train_exact_score_model

    validate_data_after_scrape >> train_live_result_model >> run_live_result_simulation
