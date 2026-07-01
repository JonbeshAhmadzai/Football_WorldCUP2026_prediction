from __future__ import annotations

from app.backend import services


def test_manual_prediction_is_order_invariant():
    forward = services.predict_match("Spain", "France", "Historical + ESPN live model")
    reverse = services.predict_match("France", "Spain", "Historical + ESPN live model")

    assert forward["order_invariant"] is True
    assert reverse["order_invariant"] is True
    assert forward["prediction"] == reverse["prediction"]
    assert forward["projected_advancer"] == reverse["projected_advancer"]
    assert forward["probabilities"]["home_win_pct"] == reverse["probabilities"]["away_win_pct"]
    assert forward["probabilities"]["away_win_pct"] == reverse["probabilities"]["home_win_pct"]
    assert forward["probabilities"]["draw_pct"] == reverse["probabilities"]["draw_pct"]

    mirrored_reverse_scorelines = [
        {"score": "-".join(row["score"].split("-")[::-1]), "probability_pct": row["probability_pct"]}
        for row in reverse["top_scorelines"]
    ]
    assert forward["top_scorelines"][:3] == mirrored_reverse_scorelines[:3]


def test_blank_host_country_builds_neutral_features():
    elo_lookup = services.team_lookup(services.read_csv(services.DATA_DIR / "elo_latest.csv"), "country")
    form_lookup = services.team_lookup(services.read_csv(services.DATA_DIR / "team_recent_form.csv"), "team")

    neutral = services.match_features("Spain", "France", "", elo_lookup, form_lookup)
    spain_host = services.match_features("Spain", "France", "Spain", elo_lookup, form_lookup)

    assert neutral is not None
    assert spain_host is not None
    assert int(neutral.iloc[0]["neutral"]) == 1
    assert int(spain_host.iloc[0]["neutral"]) == 0
