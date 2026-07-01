# FIFA World Cup 2026 Match Prediction Platform

An end-to-end football analytics project for predicting FIFA World Cup 2026 match outcomes, simulating tournament paths, tracking live tournament results, and presenting everything in a professional interactive dashboard.

The project combines historical international football data, current World Cup result scraping from ESPN, machine learning models, Monte Carlo tournament simulation, Airflow automation, and a React + FastAPI web application.

![Dashboard preview](presentations/assets/slide_1.png)

## Project Highlights

- **Interactive React dashboard** with live center, tournament probability ranking, knockout bracket, team path analysis, match predictor, and backtesting views.
- **FastAPI backend** serving predictions, model metadata, live refreshes, simulations, dashboard data, and quality checks.
- **Historical + live model** trained on historical match data plus scraped 2026 World Cup results.
- **Live result model** trained only on available 2026 World Cup result data.
- **Exact score model** estimating projected goals for matchups.
- **Monte Carlo tournament simulation** for round progression, finalist, and tournament winner probabilities.
- **ESPN scraping workflow** for finished, upcoming, and live match state snapshots.
- **Airflow DAG** to automate scraping, validation, training, and simulation refreshes.
- **Data quality checks and tests** for API health, model availability, prediction symmetry, and data readiness.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | React, Vite, Tailwind CSS, lucide-react |
| Backend | FastAPI, Uvicorn, Pydantic |
| Data | pandas, NumPy, SQLite, ESPN scoreboard data |
| Modeling | scikit-learn, XGBoost |
| Simulation | Monte Carlo tournament simulator |
| Automation | Apache Airflow |
| Testing | pytest, FastAPI TestClient |

## Repository Structure

```text
.
├── app/
│   ├── backend/                  # FastAPI app, services, quality checks
│   └── frontend/                 # React/Vite/Tailwind dashboard
├── data/
│   ├── raw/                      # Original datasets and ESPN scrape snapshots
│   ├── processed/                # Cleaned and feature-ready datasets
│   └── database/                 # SQLite database used by the project
├── src/
│   ├── scraping/                 # ESPN/latest result scraping scripts
│   ├── transform/                # Cleaning and feature engineering
│   ├── modeling/                 # Training, score model, simulations, artifacts
│   ├── scheduler/                # Airflow DAG
│   ├── load/                     # Database loading
│   └── pipeline/                 # Original pipeline entry point
├── scripts/                      # Validation and presentation helpers
├── tests/                        # Backend/data/model smoke tests
├── presentations/                # Project slide deck and generated visuals
├── reports/                      # Data quality reports
├── requirements.txt              # Main Python dependencies
├── requirements-airflow.txt      # Separate Airflow dependencies
└── Makefile                      # Common local commands
```

## Models

The app exposes two main prediction models plus one score projection model.

| Model | Artifact path | Purpose |
| --- | --- | --- |
| Historical + Live Model | `src/modeling/artifacts/historical_live/` | Main model trained with historical international data and completed 2026 World Cup live results. |
| Live Result Model | `src/modeling/artifacts/live_result/` | Separate model trained only on available 2026 World Cup result rows. Useful for current-tournament behavior, but less stable with limited data. |
| Exact Score Model | `src/modeling/score_artifacts/exact_score/` | Estimates likely goal counts for team-vs-team predictions. |

Simulation outputs are stored in:

```text
src/modeling/simulation_runs/
├── simulation_results_historical_live.csv
└── simulation_results_live_result.csv
```

## Data Sources

The project uses a mix of static historical datasets and live tournament snapshots.

Key files include:

- `data/raw/results.csv` - historical international match results.
- `data/raw/wc_2026_fixtures.csv` - World Cup 2026 fixture list.
- `data/raw/wc_2026_teams.csv` - participating teams.
- `data/raw/elo_ratings_wc2026.csv` - team Elo ratings.
- `data/raw/espn_worldcup_scoreboard.json` - latest ESPN scoreboard scrape.
- `data/processed/worldcup_2026_live_results.csv` - cleaned live/tournament result rows.
- `data/processed/matches_augmented.csv` - historical matches merged with available live World Cup results.
- `data/processed/training_dataset_augmented.csv` - training dataset for the historical + live model.

Only matches with available final results are merged into the training data. Upcoming or placeholder fixtures are used for prediction and simulation, not as training labels.

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:JonbeshAhmadzai/Football_WorldCUP2026_prediction.git
cd Football_WorldCUP2026_prediction
```

If you are already inside the cloned project, stay on the active branch:

```bash
git checkout jon-clean
```

### 2. Install Python and frontend dependencies

```bash
make install
```

This creates `.venv`, installs Python dependencies, and installs frontend dependencies inside `app/frontend`.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd app/frontend
npm install
cd ../..
```

### 3. Build the frontend

```bash
make build-ui
```

### 4. Run the application

```bash
make run-api
```

Open:

```text
http://127.0.0.1:8501
```

FastAPI serves both the API and the built React dashboard on port `8501`.

## Development Mode

For frontend development with hot reload, run the API and Vite separately.

Terminal 1:

```bash
make run-api
```

Terminal 2:

```bash
cd app/frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## Main Commands

| Command | Description |
| --- | --- |
| `make install` | Create Python venv, install Python dependencies, install frontend packages. |
| `make build-ui` | Build the React app into `app/frontend/dist`. |
| `make run-api` | Run FastAPI and serve the dashboard at `127.0.0.1:8501`. |
| `make scrape` | Scrape the latest ESPN World Cup scoreboard/results snapshot. |
| `make validate` | Run project data/model quality checks and write `reports/data_quality_latest.json`. |
| `make train` | Rebuild augmented data and train historical-live, live-result, and exact-score models. |
| `make simulate` | Run tournament simulations for both main model variants. |
| `make pipeline` | Run scrape, validate, train, simulate, and frontend build in sequence. |
| `make test` | Run the test suite. |

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Basic backend health check. |
| `/api/options` | GET | Teams, countries, metrics, and UI option lists. |
| `/api/models` | GET | Available model registry. |
| `/api/model-health` | GET | Model/data artifact status. |
| `/api/pipeline/status` | GET | Latest pipeline and artifact timestamps. |
| `/api/dashboard` | GET | Dashboard payload including rankings, live center, path data, and bracket data. |
| `/api/refresh-live` | POST | Refresh ESPN live/result snapshot. |
| `/api/predict-match` | POST | Predict a team-vs-team matchup. |
| `/api/backtest` | POST | Compare predictions against known completed matches. |

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8501/api/predict-match \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Argentina","away_team":"France","model_label":"historical_live"}'
```

## Airflow Automation

Airflow is intentionally installed in a separate environment because it has a large dependency tree.

### Install Airflow

```bash
make install-airflow
```

### List DAGs

```bash
make airflow-list
```

The main DAG is:

```text
worldcup_2026_augmented_pipeline
```

Location:

```text
src/scheduler/airflow_augmented_worldcup_dag.py
```

The DAG runs hourly and performs:

1. Scrape latest ESPN results.
2. Validate data after scrape.
3. Build augmented historical + live match data.
4. Rebuild training features.
5. Load the SQLite database.
6. Train the historical + live model.
7. Train the live result model.
8. Train the exact score model.
9. Run tournament simulations for both model variants.

The current design overwrites the active `historical_live` and `live_result` model artifacts so the app always reads the latest available models.

## Running the Full Pipeline

To refresh the data, retrain models, rerun simulations, and rebuild the frontend:

```bash
make pipeline
```

To run steps separately:

```bash
make scrape
make validate
make train
make simulate
make build-ui
```

## Deployment

The project is deployment-ready as a single Dockerized web service.

The Docker image:

1. Installs frontend dependencies.
2. Builds the React dashboard.
3. Installs Python dependencies.
4. Starts FastAPI with Uvicorn.
5. Serves the React build and API from the same deployed service.

### Deploy on Render

This repository includes:

```text
render.yaml
Dockerfile
.dockerignore
```

Recommended Render setup:

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from the repository.
3. Select the deployment branch.
4. Render will use `render.yaml` to create:
   - the Dockerized FastAPI + React web service
   - a PostgreSQL database
   - a scheduled cron scraper
5. The health check path is `/api/health`.

The app will be served by Render on its generated public URL.

The deployed app reads live match state from PostgreSQL through `DATABASE_URL`. If `DATABASE_URL` is not configured, the app falls back to local CSV/JSON files.

### Deploy with Docker Locally

```bash
docker build -t worldcup-2026-prediction-app .
docker run --rm -p 8501:8501 worldcup-2026-prediction-app
```

Open:

```text
http://127.0.0.1:8501
```

## Live Updates in Deployment

GitHub is mainly for source code. Live match updates are handled by deployment infrastructure, not by committing new data to GitHub.

The production-style live flow is:

```text
Render Cron Job
    -> ESPN scoreboard scraper
    -> PostgreSQL live tables
    -> FastAPI dashboard API
    -> React dashboard polling
```

The Render blueprint creates a cron service named:

```text
worldcup-2026-live-scraper
```

It runs every 5 minutes and executes:

```bash
python src/scraping/scrape_latest_results.py
```

When `DATABASE_URL` exists, the scraper writes:

- cleaned live match rows to `worldcup_live_results`
- raw ESPN snapshots to `espn_scoreboard_snapshots`

The FastAPI app reads `worldcup_live_results` first and falls back to `data/processed/worldcup_2026_live_results.csv` for local development.

This avoids GitHub spam commits and avoids redeploying the whole app just because a live score changed.

### Model Refresh in Deployment

Live score display and dashboard match status update through PostgreSQL without redeploying.

Model retraining is heavier than live-score refresh. The current deployment keeps model artifacts inside the Docker image and project files. For a fully production-grade retraining setup, use one of these:

- a persistent disk shared by the training worker and web service
- object storage for model artifacts
- a model registry service

For this project, the clean practical setup is:

- cron scraper every 5 minutes for live match updates
- model retraining locally or through Airflow when new final results are available
- redeploy after model artifacts change

This keeps the deployed dashboard live while avoiding unsafe automatic model overwrites in an ephemeral container.

## Testing and Quality Checks

Run tests:

```bash
make test
```

Run data quality validation:

```bash
make validate
```

The validation report is written to:

```text
reports/data_quality_latest.json
```

Current test coverage includes:

- Prediction symmetry for team order.
- Backend API smoke checks.
- Data/model artifact availability.
- Dashboard payload integrity.

## Dashboard Features

The React dashboard includes:

- **Live Center** - in-play, recently finished, and upcoming match watchlists from the latest ESPN scrape.
- **Tournament Probability Ranking** - sortable winner and round progression probability rankings.
- **Path to the Trophy** - selected-team path and progression outlook.
- **Knockout Bracket** - live bracket mode and full simulated tournament mode.
- **Match Predictor** - choose any two teams and compare model probabilities plus projected score.
- **Backtesting** - test model predictions against completed matches.
- **Model Registry** - inspect model type, artifact status, and latest simulation files.

## Environment Variables

The project includes `.env.example` for optional API tokens:

```text
FOOTBALL_DATA_API_TOKEN=replace_with_your_token
SPORTMONKS_API_TOKEN=replace_with_your_token
DATABASE_URL=postgresql://user:password@host:5432/database
```

The current ESPN scraping workflow does not require API tokens. `DATABASE_URL` is optional locally, but recommended in deployment.

Never commit `.env`. It is intentionally ignored by Git.

## Notes and Limitations

- Football predictions are probabilistic, not guarantees.
- The live-only model can react strongly to the current tournament, but it has much less training data than the historical + live model.
- Exact score prediction is especially uncertain; use projected scores as indicative estimates rather than exact forecasts.
- ESPN page structures can change, so scraping code may need maintenance if the source format changes.
- Knockout fixtures that are not officially known yet are filled through simulation mode using model projections.

## Presentation

A five-slide project presentation is included at:

```text
presentations/football_match_prediction_project_5_slides.pptx
```

Generated slide images are available in:

```text
presentations/assets/
```

## Git Remotes

This project was originally cloned from another repository. The local Git setup can keep two remotes:

- `origin` - this project repository.
- `upstream` - the original source repository, useful only if you want to pull future updates from the original project.

Normal pushes should go to:

```bash
git push origin jon-clean
```

## License

This repository is intended for educational and portfolio use as part of a BeCode football prediction project. Check source dataset licenses and third-party data provider terms before using the project commercially.
