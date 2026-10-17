"""AARIS — seed the fleet with varied realistic health states for demos/tests.

ALL seeded maintenance context is SYNTHETIC demonstration data (see
ground/aircraft_profiles.json + ground/make_profiles.py). The RUL values
below are C-MAPSS-model-shaped numbers, not real operational records.

Design (deterministic; 90 airframes = 6 aircraft models x 15, see
make_profiles.py):
  * 54 maintenance candidates (18 CRITICAL, 18 HIGH, 18 MEDIUM) -> SCHEDULED
  * 24 healthy LOW-risk aircraft (incl. AC-017 @ 112.76)         -> DEFERRED (healthy)
  * 12 aircraft with no report yet (the last two of each model block:
    AC-014/015, AC-029/030, AC-044/045, AC-059/060,
    AC-074/075, AC-089/090)                                      -> DEFERRED (awaiting telemetry)
  * every model block carries the same band shape, so each of the 6 models
    shows 3 CRITICAL / 3 HIGH / 3 MEDIUM / 4 LOW / 2 unreported in GET /models
  * no-spare aircraft include short-deadline CRITICAL/HIGH ones (spare logic)
  * PLUS: each reporting aircraft gets the synthetic degradation-trend
    history from aircraft_profiles.json ingested as OLDER reports, so the
    dashboard can plot a real RUL trend. The CURRENT report (latest
    timestamp) stays exactly the SEED value below, so the scheduler still
    sees the designed fleet state.

The SEED list is DERIVED from aircraft_profiles.json ("reports.current" per
aircraft) — one source of truth, so the seed and the profiles cannot drift.

Usage:
  # seed the live fleet.db (same code path as the HTTP API):
  ../.venv/Scripts/python.exe seed_demo.py

  # or import payload_for() from smoke tests to seed an isolated DB.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROFILES_JSON = os.path.join(HERE, "aircraft_profiles.json")


def load_current_rul() -> list[tuple[str, float]]:
    """[(aircraft_id, current RUL)] from aircraft_profiles.json, roster order.

    Aircraft whose profile has no current value (the deliberate telemetry gap)
    are skipped, and trend-only points are not part of the seed."""
    if not os.path.exists(PROFILES_JSON):
        raise RuntimeError(
            f"{PROFILES_JSON} not found — run make_profiles.py from ground/ first.")
    with open(PROFILES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for a in data.get("aircraft", []):
        current = (a.get("reports") or {}).get("current")
        if current is None:
            continue
        out.append((a["aircraft_id"], float(current)))
    return out


# aircraft_id -> predicted RUL (cycles); derived from the generated profiles.
SEED = load_current_rul()

TOP_ISSUES = [
    {"sensor": "sensor_15", "impact_cycles": -18.2,
     "note": "reduces predicted RUL (degradation driver)"},
]


def risk_band(rul: float) -> str:
    if rul >= 100:
        return "LOW"
    if rul >= 60:
        return "MEDIUM"
    if rul >= 30:
        return "HIGH"
    return "CRITICAL"


def payload_for(aircraft_id: str, rul: float, timestamp: str, engine_id: int = 1,
                action: str = "demo seed") -> dict:
    """Build a schema-valid edge-style report for the ground station."""
    health = round(max(0.0, min(100.0, rul / 125.0 * 100.0)), 2)
    risk = risk_band(rul)
    return {
        "aircraft_id": aircraft_id,
        "engine_id": engine_id,
        "timestamp": timestamp,
        "rul_cycles": round(rul, 2),
        "engine_health": health,
        "risk": risk,
        "top_issues": TOP_ISSUES if risk != "LOW" else [],
        "recommended_action": action,
        "model": {"name": "C-MAPSS Ridge RUL", "type": "Ridge", "alpha": 10.0,
                  "sequence_length": 30, "feature_count": 24, "max_rul": 125.0,
                  "artifact": "ridge_model.pkl"},
        "input_window_shape": [30, 24],
    }


def load_report_histories() -> dict[str, list[dict]]:
    """aircraft_id -> synthetic trend points from aircraft_profiles.json (may be {})."""
    if not os.path.exists(PROFILES_JSON):
        return {}
    with open(PROFILES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {a["aircraft_id"]: (a.get("reports") or {}).get("history", [])
            for a in data.get("aircraft", [])}


def _shift_ts(timestamp: str, days_ago: int) -> str:
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) - timedelta(days=days_ago)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_via_testclient(client, timestamp: str) -> int:
    """Ingest all seed reports through an open TestClient. Returns #201s.

    Since the security layer, ingestion needs a role that may ingest for any
    aircraft: we log in with the `admin` demo user (POST /auth/login) and send
    the short-lived JWT on every request. Edge device tokens are bound to a
    single aircraft and would 403 here."""
    login = client.post("/auth/login", data={"username": "admin",
                                             "password": "admin123"})
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    histories = load_report_histories()
    hist_ok = 0
    for aid, rul in SEED:
        # synthetic older trend points first (strictly older than `timestamp`,
        # so the latest-state guard keeps the current SEED value authoritative)
        for point in sorted(histories.get(aid, []), key=lambda p: -p["days_ago"]):
            past_ts = _shift_ts(timestamp, int(point["days_ago"]))
            r = client.post("/ingest_report",
                            json=payload_for(aid, float(point["rul_cycles"]), past_ts,
                                             action="synthetic trend point"),
                            headers=headers)
            assert r.status_code == 201, (aid, "history", r.status_code, r.text)
            hist_ok += 1
    ok = 0
    for aid, rul in SEED:
        r = client.post("/ingest_report", json=payload_for(aid, rul, timestamp),
                        headers=headers)
        assert r.status_code == 201, (aid, r.status_code, r.text)
        ok += 1
    print(f"[seed] {hist_ok} synthetic history reports + {ok} current reports ingested")
    return ok


if __name__ == "__main__":
    from datetime import datetime, timezone

    from fastapi.testclient import TestClient

    import ground_station as g

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Seeding live DB at {g.DB_PATH} with timestamp {ts}")
    with TestClient(g.app) as c:
        n = seed_via_testclient(c, ts)
        summary = c.get("/health").json()
    print(f"ingested {n} reports | {json.dumps(summary)}")
    print("Done. Run POST /schedule to generate a plan.")
