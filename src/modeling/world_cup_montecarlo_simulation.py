"""
Monte Carlo simulation for FIFA World Cup 2026.

Simulates the full tournament N times and reports win/finalist/semi-final
probabilities for every team. Actual confirmed results are used where available;
remaining matches are predicted stochastically from the XGBoost model.

Group stage: 3-class prediction (H / D / A).
Knockout rounds: draw probability is split equally between the two teams.

Performance notes:
- Group stage match probabilities are precomputed once (features are fixed).
- Knockout match probabilities are cached by (team1, team2) pair.
- random.random() is used instead of np.random.choice to avoid array overhead.
"""

import pickle
import random
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent
BASE_DIR = MODEL_DIR.parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW = BASE_DIR / "data" / "raw"
SIMULATION_RUNS_DIR = MODEL_DIR / "simulation_runs"

# ---------------------------------------------------------------------------
# Feature columns — must match eval_model_training.py
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_recent_wins", "home_recent_draws", "home_recent_losses",
    "home_recent_goals_for", "home_recent_goals_against", "home_recent_goal_diff",
    "away_recent_wins", "away_recent_draws", "away_recent_losses",
    "away_recent_goals_for", "away_recent_goals_against", "away_recent_goal_diff",
    "recent_win_diff", "recent_goal_diff_diff",
    "neutral",
]

RESULTS_NAME_MAP = {"Czech Republic": "Czechia"}
LIVE_RESULTS_NAME_MAP = {
    "USA": "United States",
    "United States": "United States",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Czech Republic": "Czechia",
    "Congo DR": "DR Congo",
}

DEFAULT_FORM = {
    "wins": 2, "draws": 1, "losses": 2,
    "goals_for": 5, "goals_against": 5, "goal_diff": 0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all(model_path=MODEL_DIR / "model.pkl", encoder_path=MODEL_DIR / "label_encoder.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    upcoming = pd.read_csv(DATA_PROCESSED / "upcoming_match_features.csv", parse_dates=["date"])
    elo_df = pd.read_csv(DATA_PROCESSED / "elo_latest.csv")
    form_df = pd.read_csv(DATA_PROCESSED / "team_recent_form.csv")
    results_raw = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    live_results_path = DATA_PROCESSED / "worldcup_2026_live_results.csv"
    if live_results_path.exists():
        live_results = pd.read_csv(live_results_path, parse_dates=["date"])
    else:
        live_results = pd.DataFrame()

    return model, le, upcoming, elo_df, form_df, results_raw, live_results


def build_elo_lookup(elo_df):
    return dict(zip(elo_df["country"], elo_df["rating"]))


def build_form_lookup(form_df):
    lookup = {}
    for _, row in form_df.iterrows():
        normalized = {"Czech Republic": "Czechia"}.get(row["team"], row["team"])
        lookup[normalized] = {
            "wins": row["last5_wins"],
            "draws": row["last5_draws"],
            "losses": row["last5_losses"],
            "goals_for": row["last5_goals_for"],
            "goals_against": row["last5_goals_against"],
            "goal_diff": row["last5_goal_difference"],
        }
    return lookup


def normalize_live_team_name(team):
    return LIVE_RESULTS_NAME_MAP.get(team, team)


def is_truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def is_placeholder_team(team):
    if pd.isna(team):
        return True
    text = str(team)
    return any(
        token in text
        for token in [
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
    )


def add_result_to_lookup(lookup, home_team, away_team, home_score, away_score):
    if pd.isna(home_team) or pd.isna(away_team):
        return
    if pd.isna(home_score) or pd.isna(away_score):
        return

    home = normalize_live_team_name(home_team)
    away = normalize_live_team_name(away_team)
    hs = int(home_score)
    as_ = int(away_score)
    result = "H" if hs > as_ else ("A" if hs < as_ else "D")
    lookup[(home, away)] = result


def add_knockout_winner_to_lookup(lookup, row):
    home = normalize_live_team_name(row["home_team"])
    away = normalize_live_team_name(row["away_team"])
    if is_placeholder_team(home) or is_placeholder_team(away):
        return

    if is_truthy(row.get("home_winner")):
        lookup[(home, away)] = home
    elif is_truthy(row.get("away_winner")):
        lookup[(home, away)] = away


def build_actual_results(results_raw, live_results):
    """Confirmed WC 2026 results keyed by (team1_norm, team2_norm) -> 'H'/'D'/'A'."""
    lookup = {}
    knockout_winners = {}

    if not live_results.empty:
        live = live_results[
            live_results.get("is_finished", False)
            & live_results["home_score"].notna()
            & live_results["away_score"].notna()
        ].copy()
        for _, row in live.iterrows():
            add_result_to_lookup(
                lookup,
                row["home_team"],
                row["away_team"],
                row["home_score"],
                row["away_score"],
            )
            if row.get("stage") != "group-stage":
                add_knockout_winner_to_lookup(knockout_winners, row)

    if lookup:
        return lookup, knockout_winners

    wc = results_raw[
        (results_raw["date"].dt.year == 2026)
        & results_raw["tournament"].str.contains("FIFA World Cup", na=False)
        & results_raw["home_score"].notna()
    ].copy()
    wc["home_team"] = wc["home_team"].replace(RESULTS_NAME_MAP)
    wc["away_team"] = wc["away_team"].replace(RESULTS_NAME_MAP)

    for _, row in wc.iterrows():
        add_result_to_lookup(
            lookup,
            row["home_team"],
            row["away_team"],
            row["home_score"],
            row["away_score"],
        )
    return lookup, knockout_winners


def build_fixed_r32_matchups(live_results):
    """Use real ESPN Round-of-32 fixtures once all matchups are known."""
    if live_results.empty or "stage" not in live_results:
        return None

    r32 = live_results[live_results["stage"] == "round-of-32"].copy()
    if r32.empty:
        return None

    r32 = r32.sort_values("date")
    matchups = []
    for _, row in r32.iterrows():
        home = normalize_live_team_name(row["home_team"])
        away = normalize_live_team_name(row["away_team"])
        if is_placeholder_team(home) or is_placeholder_team(away):
            return None
        matchups.append((home, away))

    return matchups if len(matchups) == 16 else None


def build_group_stage_data(upcoming):
    """Returns groups dict and schedule list."""
    gs = upcoming[upcoming["stage"] == "Group Stage"].copy()

    groups = defaultdict(list)
    for _, row in gs.iterrows():
        g = row["group"]
        for t in (row["team1_normalized"], row["team2_normalized"]):
            if t not in groups[g]:
                groups[g].append(t)

    schedule = []
    for _, row in gs.iterrows():
        t1, t2 = row["team1_normalized"], row["team2_normalized"]
        features = np.array([
            row["team1_elo"], row["team2_elo"], row["elo_diff"],
            row["team1_last5_wins"], row["team1_last5_draws"], row["team1_last5_losses"],
            row["team1_last5_goals_for"], row["team1_last5_goals_against"],
            row["team1_last5_goal_difference"],
            row["team2_last5_wins"], row["team2_last5_draws"], row["team2_last5_losses"],
            row["team2_last5_goals_for"], row["team2_last5_goals_against"],
            row["team2_last5_goal_difference"],
            row["last5_win_difference"], row["last5_goal_diff_difference"],
            1,
        ], dtype=np.float32)
        schedule.append((row["group"], t1, t2, features))

    return dict(groups), schedule


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------
def batch_probs(model, le, feature_matrix):
    """
    feature_matrix: (N, n_features) float32 array.
    Returns (N, 3) array: columns are [p_team1_wins, p_draw, p_team2_wins].
    """
    proba = model.predict_proba(feature_matrix).astype(np.float64)
    classes = list(le.classes_)
    idx_H = classes.index("H")
    idx_D = classes.index("D")
    idx_A = classes.index("A")
    out = np.stack([proba[:, idx_H], proba[:, idx_D], proba[:, idx_A]], axis=1)
    out /= out.sum(axis=1, keepdims=True)
    return out


def build_ko_feature_vec(team1, team2, elo_lookup, form_lookup):
    e1 = elo_lookup.get(team1, 1500)
    e2 = elo_lookup.get(team2, 1500)
    f1 = form_lookup.get(team1, DEFAULT_FORM)
    f2 = form_lookup.get(team2, DEFAULT_FORM)
    return np.array([
        e1, e2, e1 - e2,
        f1["wins"], f1["draws"], f1["losses"],
        f1["goals_for"], f1["goals_against"], f1["goal_diff"],
        f2["wins"], f2["draws"], f2["losses"],
        f2["goals_for"], f2["goals_against"], f2["goal_diff"],
        f1["wins"] - f2["wins"],
        f1["goal_diff"] - f2["goal_diff"],
        1,
    ], dtype=np.float32)


def precompute_gs_probs(model, le, schedule, actual_results):
    """
    Batch-predict group stage match probabilities once.
    Returns list parallel to schedule: each entry is (p_h, p_d, p_a) or None if actual result exists.
    """
    indices_to_predict = []
    feature_rows = []
    for i, (_, t1, t2, features) in enumerate(schedule):
        actual = actual_results.get((t1, t2))
        if actual is None:
            rev = actual_results.get((t2, t1))
            if rev is None:
                indices_to_predict.append(i)
                feature_rows.append(features)

    probs_list = [None] * len(schedule)
    if feature_rows:
        matrix = np.stack(feature_rows, axis=0)
        proba_matrix = batch_probs(model, le, matrix)
        for j, i in enumerate(indices_to_predict):
            probs_list[i] = tuple(proba_matrix[j])

    return probs_list


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------
def simulate_group_stage(groups, schedule, gs_probs, actual_results, elo_lookup):
    """
    Simulate group stage using precomputed probabilities where no actual result exists.
    Returns (standings, points).
    """
    points = defaultdict(int)

    for (_, t1, t2, _), probs in zip(schedule, gs_probs):
        actual = actual_results.get((t1, t2))
        if actual is None:
            rev = actual_results.get((t2, t1))
            if rev is not None:
                actual = "A" if rev == "H" else ("H" if rev == "A" else "D")

        if actual is not None:
            result = actual
        else:
            p_h, p_d, p_a = probs
            r = random.random()
            if r < p_h:
                result = "H"
            elif r < p_h + p_d:
                result = "D"
            else:
                result = "A"

        if result == "H":
            points[t1] += 3
        elif result == "D":
            points[t1] += 1
            points[t2] += 1
        else:
            points[t2] += 3

    standings = {}
    for group, teams in groups.items():
        ranked = sorted(
            teams,
            key=lambda t: (points[t], elo_lookup.get(t, 1500)),
            reverse=True,
        )
        standings[group] = ranked

    return standings, points


def determine_qualifiers(standings, points, elo_lookup):
    qualifiers = {}
    third_place = []

    for group, ranked in standings.items():
        qualifiers[f"1{group}"] = ranked[0]
        qualifiers[f"2{group}"] = ranked[1]
        third_place.append(ranked[2])

    best_third = sorted(
        third_place,
        key=lambda t: (points[t], elo_lookup.get(t, 1500)),
        reverse=True,
    )[:8]
    random.shuffle(best_third)

    return qualifiers, best_third


def build_r32_bracket(qualifiers, best_third):
    q = qualifiers
    b = best_third
    return [
        (q["1B"], q["2A"]),
        (q["1A"], q["2B"]),
        (q["1C"], q["2D"]),
        (q["1D"], q["2C"]),
        (q["1E"], q["2F"]),
        (q["1F"], q["2E"]),
        (q["1G"], q["2H"]),
        (q["1H"], q["2G"]),
        (q["1I"], q["2J"]),
        (q["1J"], q["2I"]),
        (q["1K"], q["2L"]),
        (q["1L"], q["2K"]),
        (b[0], b[1]),
        (b[2], b[3]),
        (b[6], b[7]),
        (b[4], b[5]),
    ]


def actual_knockout_winner(team1, team2, actual_results, knockout_winners):
    winner = knockout_winners.get((team1, team2))
    if winner is not None:
        return winner

    winner = knockout_winners.get((team2, team1))
    if winner is not None:
        return winner

    result = actual_results.get((team1, team2))
    if result == "H":
        return team1
    if result == "A":
        return team2

    result = actual_results.get((team2, team1))
    if result == "H":
        return team2
    if result == "A":
        return team1

    return None


def simulate_ko_match(
    team1,
    team2,
    ko_prob_cache,
    model,
    le,
    elo_lookup,
    form_lookup,
    actual_results,
    knockout_winners,
):
    """Simulate a knockout match (no draw). Uses cached probabilities."""
    winner = actual_knockout_winner(team1, team2, actual_results, knockout_winners)
    if winner is not None:
        return winner

    key = (team1, team2)
    if key not in ko_prob_cache:
        features = build_ko_feature_vec(team1, team2, elo_lookup, form_lookup)
        proba = batch_probs(model, le, features.reshape(1, -1))[0]
        p_h, p_d, p_a = proba
        # split draw prob equally → p_t1 wins
        ko_prob_cache[key] = (p_h + p_d / 2) / (p_h + p_d / 2 + p_a + p_d / 2)

    return team1 if random.random() < ko_prob_cache[key] else team2


def simulate_knockout_bracket(
    r32_matchups,
    ko_prob_cache,
    model,
    le,
    elo_lookup,
    form_lookup,
    reach,
    actual_results,
    knockout_winners,
):
    current_round = r32_matchups
    next_stage_keys = ["r16", "quarter", "semi", "final"]

    for next_stage_key in next_stage_keys:
        next_round = []
        for t1, t2 in current_round:
            winner = simulate_ko_match(
                t1,
                t2,
                ko_prob_cache,
                model,
                le,
                elo_lookup,
                form_lookup,
                actual_results,
                knockout_winners,
            )
            reach[winner][next_stage_key] += 1
            next_round.append(winner)
        current_round = list(zip(next_round[::2], next_round[1::2]))

    assert len(current_round) == 1
    t1, t2 = current_round[0]
    champion = simulate_ko_match(
        t1,
        t2,
        ko_prob_cache,
        model,
        le,
        elo_lookup,
        form_lookup,
        actual_results,
        knockout_winners,
    )
    reach[champion]["winner"] += 1
    return champion


def simulate_one_tournament(
    groups, schedule, gs_probs, actual_results, knockout_winners,
    fixed_r32_matchups, ko_prob_cache, model, le, elo_lookup, form_lookup, reach
):
    if fixed_r32_matchups:
        r32 = fixed_r32_matchups
    else:
        standings, points = simulate_group_stage(groups, schedule, gs_probs, actual_results, elo_lookup)
        qualifiers, best_third = determine_qualifiers(standings, points, elo_lookup)
        r32 = build_r32_bracket(qualifiers, best_third)

    for t1, t2 in r32:
        reach[t1]["r32"] += 1
        reach[t2]["r32"] += 1

    return simulate_knockout_bracket(
        r32,
        ko_prob_cache,
        model,
        le,
        elo_lookup,
        form_lookup,
        reach,
        actual_results,
        knockout_winners,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / "model.pkl",
        help="Model pickle to use for simulation.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=MODEL_DIR / "label_encoder.pkl",
        help="Label encoder pickle to use for simulation.",
    )
    parser.add_argument(
        "--artifact-label",
        default="baseline",
        help="Label added to the versioned simulation file.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations to run.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Also update src/modeling/simulation_results.csv.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Write simulation_runs/simulation_results_<artifact-label>.csv instead of a timestamped file.",
    )
    return parser.parse_args()


def run(
    n_simulations=10_000,
    model_path=MODEL_DIR / "model.pkl",
    encoder_path=MODEL_DIR / "label_encoder.pkl",
    artifact_label="baseline",
    update_latest=False,
    overwrite_output=False,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = artifact_label if overwrite_output else f"{artifact_label}_{timestamp}"
    print("Loading data and model...")
    model, le, upcoming, elo_df, form_df, results_raw, live_results = load_all(
        model_path,
        encoder_path,
    )

    elo_lookup = build_elo_lookup(elo_df)
    form_lookup = build_form_lookup(form_df)
    actual_results, knockout_winners = build_actual_results(results_raw, live_results)
    print(f"Loaded {len(actual_results)} confirmed World Cup results.")
    if knockout_winners:
        print(f"Loaded {len(knockout_winners)} confirmed knockout winners.")
    groups, schedule = build_group_stage_data(upcoming)
    fixed_r32_matchups = build_fixed_r32_matchups(live_results)
    if fixed_r32_matchups:
        print("Using ESPN Round-of-32 bracket with real confirmed matchups.")

    print("Precomputing group stage probabilities...")
    gs_probs = precompute_gs_probs(model, le, schedule, actual_results)

    ko_prob_cache = {}
    all_teams = [t for teams in groups.values() for t in teams]
    reach = defaultdict(lambda: defaultdict(int))
    champions = defaultdict(int)

    print(f"Running {n_simulations:,} simulations...")
    for _ in range(n_simulations):
        champ = simulate_one_tournament(
            groups, schedule, gs_probs, actual_results, knockout_winners,
            fixed_r32_matchups,
            ko_prob_cache, model, le, elo_lookup, form_lookup, reach,
        )
        champions[champ] += 1

    rows = []
    for team in all_teams:
        r = reach[team]
        rows.append({
            "team": team,
            "group": next(g for g, teams in groups.items() if team in teams),
            "win_pct": round(champions[team] / n_simulations * 100, 2),
            "final_pct": round(r["final"] / n_simulations * 100, 2),
            "semi_pct": round(r["semi"] / n_simulations * 100, 2),
            "quarter_pct": round(r["quarter"] / n_simulations * 100, 2),
            "r16_pct": round(r["r16"] / n_simulations * 100, 2),
            "r32_pct": round(r["r32"] / n_simulations * 100, 2),
            "wins": champions[team],
        })

    df = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)
    df.index += 1

    SIMULATION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = MODEL_DIR / "simulation_results.csv"
    versioned_path = SIMULATION_RUNS_DIR / f"simulation_results_{run_id}.csv"

    if update_latest and latest_path.exists():
        previous_latest_path = SIMULATION_RUNS_DIR / f"previous_latest_simulation_results_{run_id}.csv"
        previous_latest_path.write_bytes(latest_path.read_bytes())
        print(f"\nBacked up previous latest results to {previous_latest_path}")

    df.to_csv(versioned_path)
    print(f"\nSaved simulation results to {versioned_path}")
    if update_latest:
        df.to_csv(latest_path)
        print(f"Updated latest results at {latest_path}\n")
    else:
        print("Latest simulation_results.csv was not updated.\n")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(df[["team", "group", "win_pct", "final_pct", "semi_pct", "quarter_pct", "r16_pct"]].to_string())

    bel = df[df["team"] == "Belgium"]
    if not bel.empty:
        rank = bel.index[0]
        print(f"\nBelgium: rank #{rank}, {bel['win_pct'].values[0]}% chance to win the tournament")

    return df


if __name__ == "__main__":
    args = parse_args()
    run(
        n_simulations=args.n_simulations,
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        artifact_label=args.artifact_label,
        update_latest=args.update_latest,
        overwrite_output=args.overwrite_output,
    )
