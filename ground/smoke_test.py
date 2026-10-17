"""AARIS ground station — end-to-end smoke test (security-aware).

Drives the real edge node (`../edge/edge_node.py`) to produce an actual AC-017
report, ingests it through the authenticated API, verifies the security layer
(JWT users + static edge-device tokens, deny-by-default RBAC, audit trail),
then seeds a varied fleet and exercises the CP-SAT scheduler: capacity,
deadlines, spare lead times, deferrals, persistence across a service restart,
and determinism.

Runs against an isolated temp DB so it never disturbs the live `fleet.db`.

Run (from ground/):  ../.venv/Scripts/python.exe smoke_test.py
"""

import json
import os
import sqlite3
import sys

from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.join(os.path.dirname(HERE), "edge")
sys.path.insert(0, EDGE_DIR)   # so we can import the real edge node
sys.path.insert(0, HERE)

import edge_node                    # noqa: E402  (real report producer)
import ground_station as g          # noqa: E402
import security as sec              # noqa: E402
import seed_demo                    # noqa: E402

TMP_DB = os.path.join(HERE, "_smoke_test.db")
g.DB_PATH = TMP_DB                  # redirect ground to the temp DB
if os.path.exists(TMP_DB):
    os.remove(TMP_DB)

EDGE_SAMPLE = os.path.join(EDGE_DIR, "models", "ridge_model", "sample_input.json")
SEED_TS = "2026-12-02T00:00:00Z"   # newer than every timestamp used above

# the device token actually registered in this process (env EDGE_AC017_TOKEN
# or its dev default) — never hardcode a second copy
DEV_TOKEN = next(t for t, aid in sec.DEVICE_TOKENS.items() if aid == "AC-017")

# Expected demo-fleet baseline, DERIVED from the generated roster + seed instead
# of being hardcoded twice: 90 airframes = 6 aircraft models x 15, of which 78
# report and 12 (the last two of each model block) wait for telemetry.
ROSTER = [a["aircraft_id"] for a in
          json.load(open(os.path.join(HERE, "fleet_config.json"), encoding="utf-8"))["fleet"]]
CRITICAL_IDS = sorted(aid for aid, rul in seed_demo.SEED
                      if seed_demo.risk_band(rul) == "CRITICAL")
UNREPORTED_IDS = sorted(set(ROSTER) - {aid for aid, _ in seed_demo.SEED})
assert len(ROSTER) == 90 and len(CRITICAL_IDS) == 18 and len(UNREPORTED_IDS) == 12


def H(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def login(c: TestClient, username: str, password: str) -> str:
    r = c.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, (username, r.text)
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    return body["access_token"]


def audit_rows() -> list[dict]:
    """Direct read of the isolated DB's audit_logs (test-only access path)."""
    conn = sqlite3.connect(TMP_DB)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_logs ORDER BY id")]
    finally:
        conn.close()


def find(rows: list[dict], action: str, outcome: str,
         actor: str | None = None) -> list[dict]:
    return [r for r in rows if r["action"] == action and r["outcome"] == outcome
            and (actor is None or r["actor_id"] == actor)]


def get_real_edge_report() -> dict:
    """Produce the exact report the edge node emits for AC-017."""
    sample = json.load(open(EDGE_SAMPLE))
    sample["aircraft_id"] = "AC-017"
    sample["engine_id"] = 1
    sample["timestamp"] = "2026-10-11T13:07:00Z"
    with TestClient(edge_node.app) as ec:
        r = ec.post("/predict", json=sample)
        assert r.status_code == 200, r.text
        return r.json()


def check_plan(plan: dict) -> None:
    """Shared invariants for every produced plan."""
    assert plan["horizon_days"] == 30 and plan["max_daily_maintenance"] == 8
    ids = [t["aircraft_id"] for t in plan["tasks"]]
    assert len(ids) == 90 and len(set(ids)) == 90, "each aircraft exactly once"
    assert plan["scheduled_count"] + plan["deferred_count"] == 90
    assert all(v <= 8 for v in plan["daily_load"].values()), plan["daily_load"]
    assert sum(plan["daily_load"].values()) == plan["scheduled_count"]
    for t in plan["tasks"]:
        assert t["status"] in ("SCHEDULED", "DEFERRED")
        if t["status"] == "SCHEDULED":
            assert 1 <= t["scheduled_day"] <= t["maintenance_deadline_day"] <= 30, t


def main() -> int:
    edge_report = get_real_edge_report()
    print(f"[edge] produced AC-017 report: rul={edge_report['rul_cycles']} "
          f"health={edge_report['engine_health']} risk={edge_report['risk']}")
    # additive uncertainty field must not break ingestion (ground ignores extras)
    assert "uncertainty" in edge_report and edge_report["uncertainty"]["available"] is True

    with TestClient(g.app) as c:  # ---- phase A: ingest API + RBAC + scheduler ----
        # ---- auth: demo users, bad credentials ----
        bad_login = c.post("/auth/login",
                           data={"username": "maintainer", "password": "nope"})
        assert bad_login.status_code == 401
        t_maint = login(c, "maintainer", "maintainer123")
        t_cmd = login(c, "commander", "commander123")
        t_admin = login(c, "admin", "admin123")
        assert sec.decode_access_token(t_maint)["role"] == "maintenance_officer"
        assert sec.decode_access_token(t_cmd)["role"] == "commander"
        assert sec.decode_access_token(t_admin)["role"] == "admin"
        print("[0] POST /auth/login -> JWTs for maintainer/commander/admin; "
              "bad password -> 401")

        # ---- DoD: ingest the real edge report (device token) -> 201 ----
        ing = c.post("/ingest_report", json=edge_report, headers=H(DEV_TOKEN))
        assert ing.status_code == 201, ing.text
        body = ing.json()
        assert body["state_updated"] is True
        print(f"[1] POST /ingest_report (edge device token) -> 201, "
              f"report_id={body['report_id']}")

        fleet = c.get("/fleet", headers=H(t_maint)).json()
        ac17 = next(a for a in fleet["fleet"] if a["aircraft_id"] == "AC-017")
        assert (ac17["rul_cycles"], ac17["engine_health"], ac17["risk"]) == (112.76, 90.21, "LOW")
        assert ac17["latest_report_at"] == "2026-10-11T13:07:00Z"
        print(f"[2] GET /fleet -> {json.dumps({k: ac17[k] for k in ('aircraft_id','rul_cycles','engine_health','risk','latest_report_at')})}")

        detail = c.get("/fleet/AC-017", headers=H(t_maint)).json()
        assert detail["report_count"] == 1 and len(detail["history"]) == 1
        print(f"[3] GET /fleet/AC-017 -> report_count={detail['report_count']}")

        newer = dict(edge_report, timestamp="2026-12-01T00:00:00Z",
                     rul_cycles=40.0, engine_health=32.0, risk="HIGH")
        nb = c.post("/ingest_report", json=newer, headers=H(DEV_TOKEN)).json()
        assert nb["state_updated"] is True
        older = dict(edge_report, timestamp="2026-11-01T00:00:00Z",
                     rul_cycles=90.0, engine_health=72.0, risk="MEDIUM")
        ob = c.post("/ingest_report", json=older, headers=H(DEV_TOKEN)).json()
        assert ob["state_updated"] is False and ob["aircraft"]["risk"] == "HIGH"
        print("[4] timestamp guard: newer overwrote, older stored as history only")

        bad = {"missing": dict(edge_report, aircraft_id=""), "bad_risk": dict(edge_report, risk="MODERATE"),
               "rul_range": dict(edge_report, rul_cycles=999.0),
               "unknown": dict(edge_report, aircraft_id="ZZ-999"),
               "not_date": dict(edge_report, timestamp="not-a-date")}
        for name, payload in bad.items():
            # admin JWT so the failure is the payload, not the credentials
            assert c.post("/ingest_report", json=payload,
                          headers=H(t_admin)).status_code == 422, name
        assert c.get("/fleet/AC-999", headers=H(t_maint)).status_code == 404
        print(f"[5] 422/404 guards OK for: {', '.join(bad)}")

        # ---- seed the varied demo fleet (timestamp newer than all above) ----
        n = seed_demo.seed_via_testclient(c, SEED_TS)
        print(f"[6] seeded {n} demo reports (seed logs in as admin)")

        # ---- security layer: the 8 spec'd checks ----
        # (1) unauthenticated -> 401 on every protected endpoint
        assert c.get("/fleet").status_code == 401
        assert c.get("/fleet/AC-017").status_code == 401
        assert c.post("/schedule", json={}).status_code == 401
        assert c.get("/schedule/latest").status_code == 401
        assert c.post("/ingest_report", json=edge_report).status_code == 401
        print("[S1] unauthenticated /fleet, /schedule, /ingest_report -> 401")

        # (2) invalid credentials -> 401 (garbage token, expired token)
        assert c.get("/fleet", headers=H("not-a-jwt")).status_code == 401
        expired = sec.create_access_token("maintainer", "maintenance_officer",
                                          ttl_minutes=-5)
        assert c.get("/fleet", headers=H(expired)).status_code == 401
        assert c.post("/ingest_report", json=edge_report,
                      headers=H("garbage.garbage.garbage")).status_code == 401
        print("[S2] garbage + expired tokens -> 401")

        # (3) commander cannot schedule -> 403 (but can view)
        assert c.post("/schedule", json={}, headers=H(t_cmd)).status_code == 403
        assert c.post("/ingest_report", json=edge_report,
                      headers=H(t_cmd)).status_code == 403
        assert c.get("/fleet", headers=H(t_cmd)).status_code == 200
        print("[S3] commander: POST /schedule & /ingest_report -> 403, GET /fleet -> 200")

        # (4) maintenance_officer can view fleet and trigger scheduling
        assert c.get("/fleet", headers=H(t_maint)).status_code == 200
        mresp = c.post("/schedule", json={}, headers=H(t_maint))
        assert mresp.status_code == 200, mresp.text
        check_plan(mresp.json())
        # cross-role sanity: device token may not read the fleet
        assert c.get("/fleet", headers=H(DEV_TOKEN)).status_code == 403
        print("[S4] maintenance_officer: /fleet 200 + POST /schedule 200; "
              "device token /fleet -> 403")

        # (5+7) edge token ingests AC-017 -> 201 (older ts: history-only)
        ok = c.post("/ingest_report", json=edge_report, headers=H(DEV_TOKEN))
        assert ok.status_code == 201 and ok.json()["state_updated"] is False
        print("[S5] edge token: AC-017 ingest -> 201 (valid ingestion keeps working)")

        # (6) edge token on another aircraft -> 403
        wrong = dict(edge_report, aircraft_id="AC-002")
        assert c.post("/ingest_report", json=wrong, headers=H(DEV_TOKEN)).status_code == 403
        unknown_dev = dict(edge_report, aircraft_id="ZZ-999")
        assert c.post("/ingest_report", json=unknown_dev,
                      headers=H(DEV_TOKEN)).status_code == 403
        print("[S6] edge token: AC-002/ZZ-999 ingest -> 403 (device bound to AC-017)")

        # ---- scheduler: POST /schedule (maintainer) ----
        rs = c.post("/schedule", json={}, headers=H(t_maint))
        assert rs.status_code == 200, rs.text
        plan = rs.json()
        check_plan(plan)
        assert plan["solver_status"] == "OPTIMAL"
        assert plan["scheduled_count"] == 54 and plan["deferred_count"] == 36
        tasks = {t["aircraft_id"]: t for t in plan["tasks"]}
        # CRITICAL: scheduled day-1 for the day-1 deadline; all within deadline
        assert tasks["AC-001"]["scheduled_day"] == 1 and tasks["AC-001"]["maintenance_deadline_day"] == 1
        crit = [t for t in plan["tasks"] if t["risk"] == "CRITICAL"]
        assert len(crit) == 18 and all(t["status"] == "SCHEDULED" for t in crit)
        # urgency ordering: the lowest-RUL aircraft sit in the earliest days
        assert tasks["AC-001"]["scheduled_day"] <= tasks["AC-003"]["scheduled_day"]
        # spare logic: AC-003 no spare, deadline 5 -> hard lead time honored (day >= 5)
        ac003 = tasks["AC-003"]
        assert not ac003["spare_available"] and ac003["scheduled_day"] >= 5
        # healthy + telemetry-less aircraft deferred with reasons
        assert tasks["AC-017"]["status"] == "DEFERRED" and "Healthy" in tasks["AC-017"]["reason"]
        for aid in UNREPORTED_IDS:
            assert tasks[aid]["status"] == "DEFERRED" and "No health data" in tasks[aid]["reason"]
        print(f"[7] /schedule -> 200 OPTIMAL; scheduled=54 deferred=36; "
              f"max daily load={max(plan['daily_load'].values())}; "
              f"{len(UNREPORTED_IDS)} airframes deferred for telemetry; "
              f"AC-003(no spare) day={ac003['scheduled_day']}; AC-001 day=1 deadline=1")

        # determinism: identical solve on identical state
        plan2 = c.post("/schedule", json={}, headers=H(t_maint)).json()
        strip = lambda p: {k: v for k, v in p.items() if k not in ("generated_at", "plan_id")}
        assert strip(plan) == strip(plan2)
        print("[8] re-solve identical -> deterministic plan confirmed")

        # /schedule body validation
        assert c.post("/schedule", json={"horizon_days": 0},
                      headers=H(t_maint)).status_code == 422
        assert c.post("/schedule", json={"max_daily_maintenance": 99},
                      headers=H(t_maint)).status_code == 422
        print("[9] /schedule parameter validation -> 422")

        # persistence: latest == persisted (minus plan_id); commander may read it
        latest = c.get("/schedule/latest", headers=H(t_cmd)).json()
        assert latest["available"] is True
        assert strip(latest["plan"]) == strip(plan)
        assert c.get("/schedule/latest", headers=H(DEV_TOKEN)).status_code == 403
        print("[10] GET /schedule/latest returns the persisted plan "
               "(commander OK, device 403)")

        hh = c.get("/health").json()  # health stays open for liveness checks
        assert hh["status"] == "ok" and hh["aircraft_count"] == 90
        print(f"[11] GET /health -> {hh['aircraft_count']} aircraft, "
              f"{hh['report_count']} reports (no auth required)")

        # ---- model-level profiles: GET /models (6 types x 15 airframes) ----
        # Checked here, before the alert section mutates any latest state, so the
        # risk mix is compared against the pure seeded fleet.
        assert c.get("/models").status_code == 401
        assert c.get("/models", headers=H(DEV_TOKEN)).status_code == 403
        models = c.get("/models", headers=H(t_cmd)).json()
        assert models["count"] == 6 and models["aircraft_total"] == 90
        assert models["unprofiled_aircraft"] == 0 and models["synthetic"] is True
        assert sum(m["critical"] for m in models["models"]) == len(CRITICAL_IDS)
        for m in models["models"]:
            assert m["aircraft_count"] == 15, m["model_code"]
            assert m["risk_mix"]["CRITICAL"] == 3 and m["critical"] == 3, m["model_code"]
            assert m["ready_count"] == 12 and m["reporting_count"] == 13
            assert m["scheduled_in_plan"] == 9 and m["deferred_in_plan"] == 6
            assert m["role"] and m["engines"] and m["image"], m["model_code"]
            assert all(a["plan_status"] in ("SCHEDULED", "DEFERRED") for a in m["aircraft"])
        by_code = {m["model_code"]: m for m in models["models"]}
        su30 = by_code["SU30MKI"]
        assert "AC-017" in su30["aircraft_ids"] and len(su30["aircraft_ids"]) == 15
        assert su30["aircraft_ids"] == [f"AC-{i:03d}" for i in range(16, 31)]
        assert by_code["RAFALE-EH"]["aircraft_ids"] == [f"AC-{i:03d}" for i in range(1, 16)]
        print(f"[M1] GET /models -> {models['count']} aircraft models x 15 airframes, "
              f"3 CRITICAL per model, 9 scheduled in the plan each "
              f"(device token 403, no creds 401)")

        # ---- critical alerts: auto-flag, one ACTIVE per aircraft, RBAC, ack ----
        # Uses an OLDER timestamp on purpose: the aircraft's latest state must NOT
        # move (so the 54/36 plan above stays valid) while the CRITICAL report still
        # raises its urgent alert.
        assert c.get("/alerts").status_code == 401
        assert c.get("/alerts", headers=H(DEV_TOKEN)).status_code == 403
        seeded = c.get("/alerts", headers=H(t_cmd)).json()
        assert seeded["status_filter"] == "ACTIVE"
        assert sorted(a["aircraft_id"] for a in seeded["alerts"]) == CRITICAL_IDS, seeded["alerts"]
        assert all(a["severity"] == "CRITICAL" for a in seeded["alerts"])
        print(f"[A1] GET /alerts: no creds 401, device token 403; commander sees "
              f"{seeded['count']} seeded CRITICAL alerts (one per CRITICAL airframe)")

        crit_1 = dict(edge_report, risk="CRITICAL", rul_cycles=21.0,
                      engine_health=16.8, timestamp="2026-11-25T00:00:00Z")
        r1 = c.post("/ingest_report", json=crit_1, headers=H(DEV_TOKEN))
        assert r1.status_code == 201, r1.text
        b1 = r1.json()
        assert b1["state_updated"] is False and b1["alert_created"] is True
        assert isinstance(b1["alert_id"], int)
        ac017 = c.get("/fleet/AC-017", headers=H(t_cmd)).json()["aircraft"]
        assert ac017["rul_cycles"] == 112.76 and ac017["risk"] == "LOW", ac017
        print(f"[A2] CRITICAL ingest -> 201 alert_created=True id={b1['alert_id']} "
              f"while latest state untouched (old ts history-only)")

        # a second CRITICAL reading REFRESHES that alert instead of stacking one
        crit_2 = dict(crit_1, rul_cycles=18.5, engine_health=14.8,
                      timestamp="2026-11-26T00:00:00Z")
        b2 = c.post("/ingest_report", json=crit_2, headers=H(DEV_TOKEN)).json()
        assert b2["alert_created"] is False and b2["alert_id"] == b1["alert_id"]
        after = c.get("/alerts", headers=H(t_cmd)).json()
        assert after["count"] == 19, after["count"]
        ids = [a["id"] for a in after["alerts"]]
        assert len(ids) == len(set(ids)), "duplicate alert ids"
        assert len({a["aircraft_id"] for a in after["alerts"]}) == 19, "one per aircraft"
        refreshed = next(a for a in after["alerts"] if a["aircraft_id"] == "AC-017")
        assert refreshed["rul_cycles"] == 18.5, refreshed
        print("[A3] repeat CRITICAL -> same alert id refreshed (18.5 cycles), "
              "count 19, one ACTIVE alert per aircraft")

        ack = c.post(f"/alerts/{b1['alert_id']}/ack", headers=H(t_cmd)).json()
        assert ack["acknowledged"] is True and ack["already"] is False
        assert ack["alert"]["acknowledged_by"] == "commander"
        again = c.post(f"/alerts/{b1['alert_id']}/ack", headers=H(t_cmd)).json()
        assert again["acknowledged"] is False and again["already"] is True
        act = c.get("/alerts", headers=H(t_cmd)).json()
        assert act["count"] == 18 and all(a["id"] != b1["alert_id"] for a in act["alerts"])
        allok = c.get("/alerts", params={"status": "ALL"}, headers=H(t_cmd)).json()
        assert allok["count"] == 19
        assert c.post("/alerts/999999/ack", headers=H(t_cmd)).status_code == 404
        assert c.post(f"/alerts/{b1['alert_id']}/ack", headers=H(DEV_TOKEN)).status_code == 403
        assert c.get("/alerts", params={"limit": 0}, headers=H(t_cmd)).status_code == 422
        print("[A4] ack -> acknowledged (by commander); repeat ack is a no-op; "
              "ACTIVE 18 / ALL 19; unknown id 404; device token 403; bad limit 422")

        # ---- (spec A5) one device token per aircraft + auto-clear on recovery ----
        # The live fleet feed gives every aircraft its own token, so the binding
        # must hold for a craft other than AC-017 too, and an alert must close
        # when that aircraft's own latest reading recovers out of CRITICAL.
        bound = {}
        for tok, aid in sec.DEVICE_TOKENS.items():
            bound.setdefault(aid, tok)
        assert len(bound) == 90 and len(sec.DEVICE_TOKENS) == 90, bound
        tok2 = bound["AC-004"]
        assert tok2 != DEV_TOKEN, "each aircraft needs its own token"

        crit_ac4 = dict(edge_report, aircraft_id="AC-004", risk="CRITICAL",
                        rul_cycles=20.0, engine_health=16.0,
                        timestamp="2026-12-03T00:00:00Z")
        r2 = c.post("/ingest_report", json=crit_ac4, headers=H(tok2))
        assert r2.status_code == 201, r2.text
        b5 = r2.json()
        # AC-004 already had a seeded ACTIVE alert -> refreshed, not stacked
        assert b5["alert_created"] is False and isinstance(b5["alert_id"], int)
        assert b5["state_updated"] is True and b5["aircraft"]["risk"] == "CRITICAL"
        assert b5["aircraft"]["rul_cycles"] == 20.0, b5["aircraft"]
        # AC-017's edge device may not report for AC-004
        assert c.post("/ingest_report", json=crit_ac4,
                      headers=H(DEV_TOKEN)).status_code == 403
        active = c.get("/alerts", headers=H(t_cmd)).json()
        assert active["count"] == 18, active["count"]
        ac4_alerts = [a for a in active["alerts"] if a["aircraft_id"] == "AC-004"]
        assert len(ac4_alerts) == 1 and ac4_alerts[0]["id"] == b5["alert_id"]
        print("[A5a] AC-004 ingested by AC-004's own token -> 201, latest state "
              "CRITICAL/20.0; AC-017's token on AC-004 -> 403; still one ACTIVE "
              "alert for AC-004 (refreshed, not stacked)")

        # a strictly-newer non-CRITICAL reading from the same aircraft clears it
        ok_ac4 = dict(crit_ac4, risk="LOW", rul_cycles=118.0, engine_health=94.4,
                      timestamp="2026-12-04T00:00:00Z")
        b6 = c.post("/ingest_report", json=ok_ac4, headers=H(tok2)).json()
        assert b6["alert_cleared"] == 1 and b6["alert_id"] is None, b6
        active = c.get("/alerts", headers=H(t_cmd)).json()
        assert "AC-004" not in [a["aircraft_id"] for a in active["alerts"]]
        assert active["count"] == 17, active["count"]
        cleared = c.get("/alerts", params={"status": "CLEARED"},
                       headers=H(t_cmd)).json()
        mine = [a for a in cleared["alerts"] if a["id"] == b5["alert_id"]]
        assert len(mine) == 1 and mine[0]["status"] == "CLEARED" and mine[0]["cleared_at"]
        # a cleared alert is history: acking it is a no-op
        ack_cleared = c.post(f"/alerts/{b5['alert_id']}/ack", headers=H(t_cmd)).json()
        assert ack_cleared["acknowledged"] is False and ack_cleared["already"] is True
        # an OUT-OF-ORDER (older) healthy reading must not close an open alert:
        # AC-003 still has its seeded ACTIVE alert and its latest state is CRITICAL.
        tok3 = bound["AC-003"]
        stale_ok = dict(ok_ac4, aircraft_id="AC-003",
                        timestamp="2026-11-30T00:00:00Z")
        b7 = c.post("/ingest_report", json=stale_ok, headers=H(tok3)).json()
        assert b7["state_updated"] is False and b7["alert_cleared"] == 0, b7
        still = c.get("/alerts", headers=H(t_cmd)).json()
        assert "AC-003" in [a["aircraft_id"] for a in still["alerts"]]
        assert still["count"] == 17, still["count"]
        print("[A5b] newer LOW reading -> alert CLEARED (ACTIVE 18->17, cleared_at "
              "set, ack on cleared = no-op); older LOW reading for AC-003 stored "
              "as history only and left its ACTIVE alert open")

        # ---- (spec 8) audit trail: events recorded, no secrets logged ----
        rows = audit_rows()
        expected = [
            ("login", "success", "maintainer"), ("login", "denied", "maintainer"),
            ("auth", "denied", "anonymous"), ("access", "denied", "commander"),
            ("access", "denied", "edge-AC-017"),
            ("report_ingest", "success", "edge-AC-017"),
            ("report_ingest", "denied", "edge-AC-017"),
            ("report_ingest", "rejected", "admin"),
            ("report_ingest", "rejected", "anonymous"),   # validation_422 handler
            ("fleet_view", "success", "maintainer"),
            ("aircraft_view", "success", "maintainer"),
            ("scheduling", "success", "maintainer"),
            ("schedule_view", "success", "commander"),
            ("models_view", "success", "commander"),
            ("alerts_view", "success", "commander"),
            ("alert_ack", "success", "commander"),
            ("report_ingest", "success", "edge-AC-004"),
            ("report_ingest", "denied", "edge-AC-017"),
        ]
        for action, outcome, actor in expected:
            assert find(rows, action, outcome, actor), (action, outcome, actor)
        assert any('"alert_created": true' in (r["details_json"] or "")
                   for r in rows), "ingest audit must record alert_created"
        assert any('"alert_cleared": 1' in (r["details_json"] or "")
                   for r in rows), "ingest audit must record alert_cleared"
        print(f"[12] audit_logs: {len(rows)} events; all expected actions/outcomes "
              f"present (auth denials, ingest, scheduling, views)")

        dump = json.dumps(rows)
        for secret in (DEV_TOKEN, tok2, tok3, "maintainer123", "commander123",
                       "admin123"):
            assert secret not in dump, f"secret '{secret}' leaked into audit log"
        assert all('"rul_cycles"' not in (r["details_json"] or "") for r in rows), \
            "full raw report payload must never be logged"
        print("[13] audit hygiene: no tokens/passwords/raw payloads logged")

    # ---- phase B: simulated service restart (fresh lifespan on same DB file) ----
    with TestClient(g.app) as c2:
        latest2 = c2.get("/schedule/latest", headers=H(t_maint)).json()
        assert latest2["available"] is True and strip(latest2["plan"]) == strip(plan)
        fleet2 = c2.get("/fleet", headers=H(t_cmd)).json()
        assert fleet2["count"] == 90
        print("[14] after restart: /schedule/latest returns the same persisted plan")

    print("\nPASS: ground station + security + CP-SAT scheduler verified "
          "end-to-end (login, RBAC 401/403 matrix, per-aircraft device-token "
          "binding, ingest, guards, critical alerts auto-flag + ack + "
          "auto-clear on recovery, plan, capacity, "
          "deadlines, spares, audit trail, persistence, determinism).")
    return 0


if __name__ == "__main__":
    code = main()
    if os.path.exists(TMP_DB):
        os.remove(TMP_DB)
    raise SystemExit(code)
