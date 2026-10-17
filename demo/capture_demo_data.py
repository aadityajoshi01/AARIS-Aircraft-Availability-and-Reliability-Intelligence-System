"""AARIS — capture dashboard/demo_data.json from the current ground/fleet.db.

Reads GET /fleet, GET /fleet/{id} (all roster aircraft), GET /models,
GET /schedule/latest and GET /alerts through an in-process TestClient
(commander service account) and writes the DEMO MODE snapshot the dashboard
falls back to.

The snapshot therefore contains the SAME synthetic demonstration data the
live station serves (see ground/aircraft_profiles.json provenance note —
no real IAF operational or maintenance data anywhere).

Run (from demo/):  ../.venv/Scripts/python.exe capture_demo_data.py
"""

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ground"))

import ground_station as g  # noqa: E402  (in-process only; port not touched)
from fastapi.testclient import TestClient  # noqa: E402

OUT = os.path.join(ROOT, "dashboard", "demo_data.json")


def main() -> None:
    with TestClient(g.app) as c:
        login = c.post("/auth/login", data={"username": "commander",
                                            "password": "commander123"})
        assert login.status_code == 200, login.text
        h = {"Authorization": f"Bearer {login.json()['access_token']}"}

        fleet = c.get("/fleet", headers=h).json()
        ids = [a["aircraft_id"] for a in fleet["fleet"]]
        aircraft = {aid: c.get(f"/fleet/{aid}", headers=h).json() for aid in ids}
        models = c.get("/models", headers=h).json()
        schedule = c.get("/schedule/latest", headers=h).json()
        alerts = c.get("/alerts", headers=h).json()

    if not (schedule.get("available") and schedule["plan"].get("solver_status") == "OPTIMAL"):
        raise SystemExit("refusing to capture: no OPTIMAL plan on the station "
                         "(POST /schedule first)")
    if not alerts.get("alerts"):
        raise SystemExit("refusing to capture: no ACTIVE alerts on the station, so "
                         "DEMO MODE could not demonstrate the critical-alert path")
    if not models.get("models"):
        raise SystemExit("refusing to capture: GET /models returned no aircraft-model "
                         "profiles, so DEMO MODE could not demonstrate the Aircraft "
                         "Models section (is aircraft_profiles.json loaded?)")

    doc = {
        "description": "Offline fallback captured from the live AARIS ground station "
                       "(GET /fleet, /fleet/{id}, /models, /schedule/latest, /alerts) on "
                       "the seeded synthetic demo fleet. ALL data is synthetic "
                       "demonstration data (NASA C-MAPSS-shaped RUL reports + fictional "
                       "fleet profiles from ground/aircraft_profiles.json); no real IAF "
                       "operational or maintenance data is used.",
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ground_station TestClient over live ground/fleet.db (read-only)",
        "fleet": fleet,
        "aircraft": aircraft,
        "models": models,
        "schedule": schedule,
        "alerts": alerts,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    plan = schedule["plan"]
    print(f"[capture] wrote {OUT}: {fleet['count']} aircraft, "
          f"{sum(a['report_count'] for a in aircraft.values())} reports, "
          f"{models['count']} aircraft models "
          f"({models['aircraft_total']} profiled airframes), "
          f"plan {plan['scheduled_count']}/{plan['deferred_count']} "
          f"({plan['solver_status']}), {alerts['count']} active alerts")


if __name__ == "__main__":
    main()
