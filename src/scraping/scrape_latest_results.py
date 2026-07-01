import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.storage import live_store


BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
START_DATE = "20260611"
END_DATE = "20260719"
SCOREBOARD_LIMIT = 200
RAW_OUTPUT_PATH = Path("data/raw/espn_worldcup_scoreboard.json")
PROCESSED_OUTPUT_PATH = Path("data/processed/worldcup_2026_live_results.csv")
LOCAL_FIXTURES_PATH = Path("data/processed/fixtures_clean.csv")

TEAM_NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


def local_fixtures():
    if not LOCAL_FIXTURES_PATH.exists():
        return None

    fixtures = pd.read_csv(LOCAL_FIXTURES_PATH, parse_dates=["date"])
    return fixtures


def get_json(url, params=None):
    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_scoreboard():
    date_range = f"{START_DATE}-{END_DATE}"
    payload = get_json(
        BASE_URL,
        params={
            "dates": date_range,
            "limit": SCOREBOARD_LIMIT,
        },
    )
    return [{
        "date_range": date_range,
        "events_count": len(payload.get("events", [])),
        "payload": payload,
    }]


def normalize_team_name(name):
    if not name:
        return None
    return TEAM_NAME_MAP.get(name, name)


def competition(event):
    competitions = event.get("competitions") or []
    return competitions[0] if competitions else {}


def competitor_by_side(comp, side):
    for competitor in comp.get("competitors") or []:
        if competitor.get("homeAway") == side:
            return competitor
    return {}


def competitor_team_name(competitor):
    team = competitor.get("team") or {}
    return (
        team.get("displayName")
        or team.get("shortDisplayName")
        or team.get("name")
        or competitor.get("displayName")
    )


def competitor_score(competitor):
    score = competitor.get("score")
    if score in (None, ""):
        return None
    return int(score)


def parse_group(alt_game_note):
    if not alt_game_note:
        return None

    match = re.search(r"Group\s+([A-L])", alt_game_note)
    return match.group(1) if match else None


def status_flags(status_type):
    state = status_type.get("state")
    completed = bool(status_type.get("completed"))
    name = status_type.get("name")

    return {
        "is_finished": completed or state == "post",
        "is_live": state == "in",
        "is_scheduled": state == "pre",
        "status": status_type.get("description") or status_type.get("detail") or name,
    }


def clean_events(payloads):
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for page in payloads:
        for event in page["payload"].get("events", []):
            comp = competition(event)
            status = comp.get("status") or {}
            status_type = status.get("type") or {}
            flags = status_flags(status_type)
            home = competitor_by_side(comp, "home")
            away = competitor_by_side(comp, "away")
            raw_home_team = competitor_team_name(home)
            raw_away_team = competitor_team_name(away)
            home_score = competitor_score(home)
            away_score = competitor_score(away)
            if flags["is_scheduled"]:
                home_score = None
                away_score = None
            venue = comp.get("venue") or {}
            address = venue.get("address") or {}

            rows.append({
                "provider": "espn",
                "provider_match_id": event.get("id"),
                "date": event.get("date") or comp.get("date"),
                "stage": (event.get("season") or {}).get("slug"),
                "group": parse_group(comp.get("altGameNote")),
                "home_team": normalize_team_name(raw_home_team),
                "away_team": normalize_team_name(raw_away_team),
                "raw_home_team": raw_home_team,
                "raw_away_team": raw_away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_winner": home.get("winner"),
                "away_winner": away.get("winner"),
                "status": flags["status"],
                "status_state": status_type.get("state"),
                "status_detail": status_type.get("detail"),
                "display_clock": status.get("displayClock"),
                "period": status.get("period"),
                "is_finished": flags["is_finished"],
                "is_live": flags["is_live"],
                "is_scheduled": flags["is_scheduled"],
                "venue": venue.get("fullName"),
                "city": address.get("city"),
                "country": address.get("country"),
                "source_note": comp.get("altGameNote"),
                "fetched_at": fetched_at,
            })

    if not rows:
        return pd.DataFrame()

    cleaned = pd.DataFrame(rows)
    cleaned = cleaned.drop_duplicates(subset=["provider_match_id"]).sort_values("date")
    return cleaned.reset_index(drop=True)


def save_outputs(payloads, cleaned):
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "espn",
        "source_url": BASE_URL,
        "date_range": f"{START_DATE}-{END_DATE}",
        "events_count": int(sum(page["events_count"] for page in payloads)),
        "pages": payloads,
    }
    RAW_OUTPUT_PATH.write_text(json.dumps(raw_payload, indent=2))
    cleaned.to_csv(PROCESSED_OUTPUT_PATH, index=False)
    if live_store.is_enabled():
        live_store.save_scoreboard_snapshot(raw_payload)
        live_store.save_live_results(cleaned)


def local_fixture_teams(fixtures):
    if fixtures is None:
        return set()

    teams = set(fixtures["team1"].dropna()).union(set(fixtures["team2"].dropna()))
    return {normalize_team_name(team) for team in teams}


def print_summary(fixtures, cleaned):
    print(f"Saved raw JSON: {RAW_OUTPUT_PATH}")
    print(f"Saved cleaned CSV: {PROCESSED_OUTPUT_PATH}")
    if live_store.is_enabled():
        print("Synced live results to PostgreSQL via DATABASE_URL.")
    print(f"Rows: {len(cleaned)}")

    if cleaned.empty:
        print("\nNo ESPN World Cup events were returned for the tournament date range.")
        return

    status_counts = Counter(cleaned["status"].fillna("Unknown"))
    print("\nStatus counts:")
    for status, count in status_counts.most_common():
        print(f"- {status}: {count}")

    print("\nScore availability:")
    print(f"- rows with both scores: {cleaned[['home_score', 'away_score']].notna().all(axis=1).sum()}")
    print(f"- live rows: {cleaned['is_live'].sum()}")
    print(f"- finished rows: {cleaned['is_finished'].sum()}")
    print(f"- scheduled rows: {cleaned['is_scheduled'].sum()}")

    fixture_teams = local_fixture_teams(fixtures)
    scraped_teams = set(cleaned["home_team"].dropna()).union(set(cleaned["away_team"].dropna()))
    missing_from_local = sorted(scraped_teams - fixture_teams) if fixture_teams else []
    if fixtures is None:
        print("\nLocal fixtures file not found, so team-name validation was skipped.")
    elif missing_from_local:
        print("\nESPN team names not found in local fixtures after normalization:")
        for team in missing_from_local:
            print(f"- {team}")
    else:
        print("\nAll ESPN team names matched local fixtures after normalization.")


def main():
    fixtures = local_fixtures()
    payloads = fetch_scoreboard()
    cleaned = clean_events(payloads)
    save_outputs(payloads, cleaned)
    print_summary(fixtures, cleaned)


if __name__ == "__main__":
    main()
