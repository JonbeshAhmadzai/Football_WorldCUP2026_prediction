from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd


LIVE_RESULTS_TABLE = "worldcup_live_results"
SCOREBOARD_SNAPSHOTS_TABLE = "espn_scoreboard_snapshots"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def is_enabled() -> bool:
    return bool(database_url())


def _connect():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set, but psycopg is not installed. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return psycopg.connect(database_url())


def init_live_schema() -> None:
    if not is_enabled():
        return

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {LIVE_RESULTS_TABLE} (
                    provider TEXT NOT NULL,
                    provider_match_id TEXT PRIMARY KEY,
                    date TIMESTAMPTZ,
                    stage TEXT,
                    tournament_group TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    raw_home_team TEXT,
                    raw_away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    home_winner BOOLEAN,
                    away_winner BOOLEAN,
                    status TEXT,
                    status_state TEXT,
                    status_detail TEXT,
                    display_clock TEXT,
                    period INTEGER,
                    is_finished BOOLEAN NOT NULL DEFAULT FALSE,
                    is_live BOOLEAN NOT NULL DEFAULT FALSE,
                    is_scheduled BOOLEAN NOT NULL DEFAULT FALSE,
                    venue TEXT,
                    city TEXT,
                    country TEXT,
                    source_note TEXT,
                    fetched_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCOREBOARD_SNAPSHOTS_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    date_range TEXT NOT NULL,
                    events_count INTEGER NOT NULL DEFAULT 0,
                    payload JSONB NOT NULL,
                    fetched_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{LIVE_RESULTS_TABLE}_date "
                f"ON {LIVE_RESULTS_TABLE} (date)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{LIVE_RESULTS_TABLE}_status "
                f"ON {LIVE_RESULTS_TABLE} (is_live, is_finished, is_scheduled)"
            )
        conn.commit()


def _none_if_missing(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def _int_or_none(value: Any) -> int | None:
    value = _none_if_missing(value)
    if value is None or value == "":
        return None
    return int(value)


def _timestamp_or_none(value: Any) -> Any:
    value = _none_if_missing(value)
    if value is None or value == "":
        return None
    return pd.to_datetime(value, errors="coerce", utc=True).to_pydatetime()


def save_live_results(results: pd.DataFrame) -> int:
    if not is_enabled() or results.empty:
        return 0

    init_live_schema()
    columns = [
        "provider",
        "provider_match_id",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "raw_home_team",
        "raw_away_team",
        "home_score",
        "away_score",
        "home_winner",
        "away_winner",
        "status",
        "status_state",
        "status_detail",
        "display_clock",
        "period",
        "is_finished",
        "is_live",
        "is_scheduled",
        "venue",
        "city",
        "country",
        "source_note",
        "fetched_at",
    ]
    sql = f"""
        INSERT INTO {LIVE_RESULTS_TABLE} (
            provider, provider_match_id, date, stage, tournament_group,
            home_team, away_team, raw_home_team, raw_away_team,
            home_score, away_score, home_winner, away_winner,
            status, status_state, status_detail, display_clock, period,
            is_finished, is_live, is_scheduled, venue, city, country,
            source_note, fetched_at, updated_at
        )
        VALUES (
            %(provider)s, %(provider_match_id)s, %(date)s, %(stage)s, %(group)s,
            %(home_team)s, %(away_team)s, %(raw_home_team)s, %(raw_away_team)s,
            %(home_score)s, %(away_score)s, %(home_winner)s, %(away_winner)s,
            %(status)s, %(status_state)s, %(status_detail)s, %(display_clock)s,
            %(period)s, %(is_finished)s, %(is_live)s, %(is_scheduled)s,
            %(venue)s, %(city)s, %(country)s, %(source_note)s, %(fetched_at)s,
            NOW()
        )
        ON CONFLICT (provider_match_id) DO UPDATE SET
            provider = EXCLUDED.provider,
            date = EXCLUDED.date,
            stage = EXCLUDED.stage,
            tournament_group = EXCLUDED.tournament_group,
            home_team = EXCLUDED.home_team,
            away_team = EXCLUDED.away_team,
            raw_home_team = EXCLUDED.raw_home_team,
            raw_away_team = EXCLUDED.raw_away_team,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            home_winner = EXCLUDED.home_winner,
            away_winner = EXCLUDED.away_winner,
            status = EXCLUDED.status,
            status_state = EXCLUDED.status_state,
            status_detail = EXCLUDED.status_detail,
            display_clock = EXCLUDED.display_clock,
            period = EXCLUDED.period,
            is_finished = EXCLUDED.is_finished,
            is_live = EXCLUDED.is_live,
            is_scheduled = EXCLUDED.is_scheduled,
            venue = EXCLUDED.venue,
            city = EXCLUDED.city,
            country = EXCLUDED.country,
            source_note = EXCLUDED.source_note,
            fetched_at = EXCLUDED.fetched_at,
            updated_at = NOW()
    """

    payload = []
    for _, row in results.iterrows():
        item = {column: _none_if_missing(row.get(column)) for column in columns}
        item["provider_match_id"] = str(item["provider_match_id"])
        item["date"] = _timestamp_or_none(item["date"])
        item["fetched_at"] = _timestamp_or_none(item["fetched_at"])
        item["home_score"] = _int_or_none(item["home_score"])
        item["away_score"] = _int_or_none(item["away_score"])
        item["period"] = _int_or_none(item["period"])
        item["home_winner"] = None if item["home_winner"] is None else _bool(item["home_winner"])
        item["away_winner"] = None if item["away_winner"] is None else _bool(item["away_winner"])
        item["is_finished"] = _bool(item["is_finished"])
        item["is_live"] = _bool(item["is_live"])
        item["is_scheduled"] = _bool(item["is_scheduled"])
        payload.append(item)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, payload)
        conn.commit()
    return len(payload)


def save_scoreboard_snapshot(raw_payload: dict[str, Any]) -> None:
    if not is_enabled():
        return

    init_live_schema()
    fetched_at = _timestamp_or_none(raw_payload.get("fetched_at")) or datetime.now(timezone.utc)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCOREBOARD_SNAPSHOTS_TABLE}
                    (provider, source_url, date_range, events_count, payload, fetched_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    raw_payload.get("provider", "espn"),
                    raw_payload.get("source_url", ""),
                    raw_payload.get("date_range", ""),
                    int(raw_payload.get("events_count") or 0),
                    json.dumps(raw_payload),
                    fetched_at,
                ),
            )
        conn.commit()


def load_live_results() -> pd.DataFrame:
    if not is_enabled():
        return pd.DataFrame()

    init_live_schema()
    sql = f"""
        SELECT
            provider,
            provider_match_id,
            date,
            stage,
            tournament_group AS "group",
            home_team,
            away_team,
            raw_home_team,
            raw_away_team,
            home_score,
            away_score,
            home_winner,
            away_winner,
            status,
            status_state,
            status_detail,
            display_clock,
            period,
            is_finished,
            is_live,
            is_scheduled,
            venue,
            city,
            country,
            source_note,
            fetched_at
        FROM {LIVE_RESULTS_TABLE}
        ORDER BY date NULLS LAST, provider_match_id
    """
    with _connect() as conn:
        return pd.read_sql(sql, conn)
