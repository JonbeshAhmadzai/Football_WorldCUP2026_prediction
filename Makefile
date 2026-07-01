PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
UVICORN ?= .venv/bin/uvicorn
AIRFLOW ?= .airflow-venv/bin/airflow
AIRFLOW_HOME ?= $(PWD)/airflow_home
AIRFLOW_DAGS_FOLDER ?= $(PWD)/src/scheduler

.PHONY: install install-airflow run-api build-ui test validate scrape train simulate pipeline airflow-list

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt
	cd app/frontend && npm install

install-airflow:
	python3 -m venv .airflow-venv
	.airflow-venv/bin/pip install -r requirements-airflow.txt

run-api:
	$(UVICORN) app.backend.main:app --host 127.0.0.1 --port 8501

build-ui:
	cd app/frontend && npm run build

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_project_data.py --output reports/data_quality_latest.json

scrape:
	$(PYTHON) src/scraping/scrape_latest_results.py

train:
	$(PYTHON) src/transform/build_augmented_matches.py
	$(PYTHON) src/transform/build_training_dataset_augmented.py
	$(PYTHON) src/modeling/data_prep_augmented.py
	$(PYTHON) src/modeling/eval_model_training.py --features-path src/modeling/training_features_augmented.csv --artifact-label historical_live --overwrite-artifact
	$(PYTHON) src/modeling/train_live_only_model.py
	$(PYTHON) src/modeling/train_exact_score_model.py

simulate:
	$(PYTHON) src/modeling/world_cup_montecarlo_simulation.py --model-path src/modeling/artifacts/historical_live/model.pkl --encoder-path src/modeling/artifacts/historical_live/label_encoder.pkl --artifact-label historical_live --overwrite-output
	$(PYTHON) src/modeling/world_cup_montecarlo_simulation.py --model-path src/modeling/artifacts/live_result/model.pkl --encoder-path src/modeling/artifacts/live_result/label_encoder.pkl --artifact-label live_result --overwrite-output

pipeline: scrape validate train simulate build-ui

airflow-list:
	AIRFLOW_HOME=$(AIRFLOW_HOME) AIRFLOW__CORE__DAGS_FOLDER=$(AIRFLOW_DAGS_FOLDER) $(AIRFLOW) dags list
