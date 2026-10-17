"""AARIS end-to-end pipeline verification.

Run against live services (start them with demo/run_demo.ps1 or start.bat):

    .venv\\Scripts\\python.exe demo\\verify_pipeline.py

Checks (each independent):
  1. edge    GET  /health            - Ridge artifacts loaded
  2. edge    POST /predict           - real sample_input.json
  3. edge    prediction values       - ~112.76 cycles / ~90.21 % / LOW
  4. ground  GET  /health            - 90 aircraft in roster
  4b. ground POST  /auth/login       - commander demo user -> JWT
  4c. ground GET  /fleet             - unauthenticated -> 401 (deny-by-default)
  5. ground  GET  /fleet             - AC-017 latest state matches (Bearer JWT)
  6. ground  GET  /schedule/latest   - 90 tasks, 54 scheduled, 36 deferred (JWT)
  7. ground  plan capacity           - no day exceeds max_daily_maintenance
  8. ground  GET  /alerts            - ACTIVE alerts mirror exactly the CRITICAL
                                        aircraft, one per aircraft (JWT); the
                                        rest are CLEARED (aircraft recovered)
  9. ground  GET  /alerts            - unauthenticated -> 401 (deny-by-default)
 10. ground  GET  /models            - 6 aircraft models x 15 airframes, and each
                                        model's risk mix matches /fleet (JWT)
 11. ground  GET  /models            - unauthenticated -> 401 (deny-by-default)

Prints a PASS/FAIL summary; exits nonzero if anything fails.
Read-only except for the stateless edge /predict call and the /auth/login form
post - it never writes to the database or to any file.
"""

import argparse
import json
import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_INPUT = os.path.join(ROOT, "edge", "models", "ridge_model", "sample_input.json")

TIMEOUT = 10
TOL = 0.01  # cycles/percent tolerance for "approximately"

# Frozen baseline for the seeded synthetic demo fleet: 6 aircraft models x 15
# airframes (ground/make_profiles.py), each model carrying the same health shape
# (3 CRITICAL / 3 HIGH / 3 MEDIUM / 4 LOW / 2 airframes with no telemetry yet),
# so 18 / 18 / 18 / 24 / 12 across 90 airframes and 54 maintenance candidates.
EXPECTED = {
    "rul_cycles": 112.76,
    "engine_health": 90.21,
    "risk": "LOW",
    "aircraft_count": 90,
    "tasks": 90,
    "scheduled": 54,
    "deferred": 36,
    "models": 6,
    "per_model": 15,
    "critical_per_model": 3,
}

results = []  # (name, ok, detail)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    return ok


def short_err(exc: Exception) -> str:
    msg = str(exc.__cause__ or exc)
    return f"{exc.__class__.__name__}: {msg[:110]}"


def get_json(url: str, headers: dict | None = None):
    """Return (data, error_message). Never raises."""
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
    except requests.exceptions.RequestException as exc:
        return None, f"request failed ({short_err(exc)})"
    if r.status_code == 401:
        return None, "HTTP 401 (missing/invalid credentials)"
    if r.status_code == 403:
        return None, "HTTP 403 (role not permitted)"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "malformed JSON response"


def post_json(url: str, payload):
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return None, f"request failed ({short_err(exc)})"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        return r.json(), None
    except ValueError:
        return None, "malformed JSON response"


def close_to(value, expected: float) -> bool:
    return isinstance(value, (int, float)) and abs(value - expected) <= TOL


def main() -> int:
    ap = argparse.ArgumentParser(description="AARIS pipeline smoke check")
    ap.add_argument("--edge", default="http://127.0.0.1:8000")
    ap.add_argument("--ground", default="http://127.0.0.1:8001")
    args = ap.parse_args()

    hint = "Is the stack running? Start it with demo/run_demo.ps1 (or start.bat)."

    # --- 1) edge health ------------------------------------------------------
    health, err = get_json(args.edge + "/health")
    if not check("1. edge /health reachable and ok", err is None and health and health.get("status") == "ok",
                 err or f"status={health.get('status')}"):
        print("     " + hint)

    # --- 2) POST the real sample window --------------------------------------
    prediction, err = None, None
    if os.path.exists(SAMPLE_INPUT):
        with open(SAMPLE_INPUT, "r", encoding="utf-8") as f:
            sample = json.load(f)
        prediction, err = post_json(args.edge + "/predict", sample)
    else:
        err = f"sample_input.json not found at {SAMPLE_INPUT}"
    check("2. edge /predict accepts real sample_input.json", prediction is not None,
          err or f"engine={prediction.get('engine_id')} aircraft={prediction.get('aircraft_id')}")

    # --- 3) prediction values -------------------------------------------------
    if prediction:
        ok = (close_to(prediction.get("rul_cycles"), EXPECTED["rul_cycles"])
              and close_to(prediction.get("engine_health"), EXPECTED["engine_health"])
              and prediction.get("risk") == EXPECTED["risk"])
        check("3. prediction ~112.76 cycles / ~90.21 % / LOW", ok,
              f"got rul_cycles={prediction.get('rul_cycles')} "
              f"engine_health={prediction.get('engine_health')} risk={prediction.get('risk')}")
    else:
        check("3. prediction ~112.76 cycles / ~90.21 % / LOW", False, "skipped: no prediction")

    # --- 4) ground health ------------------------------------------------------
    ghealth, err = get_json(args.ground + "/health")
    if not check("4. ground /health reachable, 90 aircraft",
                 err is None and ghealth and ghealth.get("aircraft_count") == EXPECTED["aircraft_count"],
                 err or f"aircraft_count={ghealth.get('aircraft_count')}, "
                        f"report_count={ghealth.get('report_count')}"):
        print("     " + hint)

    # --- 4b) auth: commander demo user -> JWT for the protected reads ---------
    gh: dict = {}
    login, err = None, None
    try:
        resp = requests.post(args.ground + "/auth/login",
                             data={"username": "commander", "password": "commander123"},
                             timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        err = f"request failed ({short_err(exc)})"
    else:
        if resp.status_code != 200:
            err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        else:
            login = resp.json()
    if err is None and login and login.get("role") == "commander":
        gh = {"Authorization": f"Bearer {login['access_token']}"}
    check("4b. ground /auth/login (commander demo user) -> JWT", err is None and bool(gh),
          err or "got a usable token")

    # --- 4c) deny-by-default: unauthenticated /fleet must be refused -----------
    deny_ok, deny_detail = False, "no response"
    try:
        r = requests.get(args.ground + "/fleet", timeout=TIMEOUT)
        deny_ok, deny_detail = r.status_code == 401, f"HTTP {r.status_code}"
    except requests.exceptions.RequestException as exc:
        deny_detail = short_err(exc)
    check("4c. ground /fleet without credentials -> 401", deny_ok, deny_detail)

    # --- 5) fleet contains AC-017 with expected latest state --------------------
    fleet, err = get_json(args.ground + "/fleet", headers=gh)
    if err or not fleet:
        check("5. ground /fleet AC-017 state", False, err or "no fleet payload")
    else:
        ac17 = next((a for a in fleet.get("fleet", []) if a["aircraft_id"] == "AC-017"), None)
        if ac17 is None:
            check("5. ground /fleet AC-017 state", False, "AC-017 missing from fleet")
        else:
            ok = (close_to(ac17.get("rul_cycles"), EXPECTED["rul_cycles"])
                  and close_to(ac17.get("engine_health"), EXPECTED["engine_health"])
                  and ac17.get("risk") == EXPECTED["risk"])
            check("5. ground /fleet AC-017 state", ok,
                  f"AC-017: rul={ac17.get('rul_cycles')} health={ac17.get('engine_health')} "
                  f"risk={ac17.get('risk')} at {ac17.get('latest_report_at')}")

    # --- 6) latest schedule: 90 tasks / 54 / 36 ---------------------------------
    latest, err = get_json(args.ground + "/schedule/latest", headers=gh)
    plan = None
    if err or not latest or not latest.get("available"):
        check("6. /schedule/latest has 90 tasks, 54 scheduled, 36 deferred", False,
              err or "no plan persisted yet - POST /schedule once (run_demo.ps1 -Reset does this)")
    else:
        plan = latest["plan"]
        tasks = plan.get("tasks", [])
        n_sched = sum(1 for t in tasks if t.get("status") == "SCHEDULED")
        n_defer = sum(1 for t in tasks if t.get("status") == "DEFERRED")
        ok = (len(tasks) == EXPECTED["tasks"]
              and n_sched == EXPECTED["scheduled"] == plan.get("scheduled_count")
              and n_defer == EXPECTED["deferred"] == plan.get("deferred_count"))
        check("6. /schedule/latest has 90 tasks, 54 scheduled, 36 deferred", ok,
              f"tasks={len(tasks)} scheduled={n_sched}/{plan.get('scheduled_count')} "
              f"deferred={n_defer}/{plan.get('deferred_count')} solver={plan.get('solver_status')}")

    # --- 7) capacity: no day above max_daily_maintenance ------------------------
    if plan:
        cap = plan.get("max_daily_maintenance", 8)
        load = {int(d): n for d, n in (plan.get("daily_load") or {}).items()}
        over = {d: n for d, n in load.items() if n > cap}
        horizon = plan.get("horizon_days", 30)
        bad_days = [d for d in load if not 1 <= d <= horizon]
        total_load = sum(load.values())
        ok = (not over and not bad_days and total_load == plan.get("scheduled_count"))
        check("7. plan capacity respected (no day > max_daily)", ok,
              f"peak={max(load.values()) if load else 0}/{cap} per day, "
              f"{len(load)} active days, sum(daily_load)={total_load}")
    else:
        check("7. plan capacity respected (no day > max_daily)", False,
              "skipped: no plan")

    # --- 8) critical alerts mirror the CRITICAL fleet state (read-only) ---------
    alerts, err = get_json(args.ground + "/alerts", headers=gh)
    if err or alerts is None:
        check("8. /alerts match the CRITICAL aircraft (one ACTIVE each)", False, err or "no payload")
    else:
        crit = sorted(a["aircraft_id"] for a in (fleet or {}).get("fleet", [])
                      if a.get("risk") == "CRITICAL")
        alarmed = sorted({a["aircraft_id"] for a in alerts.get("alerts", [])
                          if a.get("status") == "ACTIVE"})
        ids = [a.get("id") for a in alerts.get("alerts", [])]
        ok = (alarmed == crit and len(ids) == len(set(ids))
              and all(a.get("severity") == "CRITICAL" for a in alerts.get("alerts", [])))
        # a CRITICAL that has since recovered must be CLEARED, not left ACTIVE
        cleared, cerr = get_json(args.ground + "/alerts?status=CLEARED&limit=500",
                                 headers=gh)
        if cerr or cleared is None:
            ok, cleared_detail = False, cerr or "no payload"
        else:
            rows = cleared.get("alerts", [])
            ok = ok and all(a.get("status") == "CLEARED" and a.get("cleared_at")
                            for a in rows)
            ok = ok and not (set(ids) & {a.get("id") for a in rows})
            cleared_detail = f"{len(rows)} cleared"
        check("8. /alerts match the CRITICAL aircraft (one ACTIVE each)", ok,
              f"alerts={alarmed} vs CRITICAL={crit}, {cleared_detail}")

    # --- 9) deny-by-default: unauthenticated /alerts must be refused ------------
    deny_ok, deny_detail = False, "no response"
    try:
        r = requests.get(args.ground + "/alerts", timeout=TIMEOUT)
        deny_ok, deny_detail = r.status_code == 401, f"HTTP {r.status_code}"
    except requests.exceptions.RequestException as exc:
        deny_detail = short_err(exc)
    check("9. ground /alerts without credentials -> 401", deny_ok, deny_detail)

    # --- 10) per-aircraft-MODEL profiles agree with the live fleet ---------------
    models, err = get_json(args.ground + "/models", headers=gh)
    name10 = ("10. ground /models: 6 model profiles x 15 airframes, aggregates "
              "match /fleet")
    if err or not models:
        check(name10, False, err or "no payload")
    else:
        rows = models.get("models", [])
        state = {a["aircraft_id"]: a for a in (fleet or {}).get("fleet", [])}
        ok = (models.get("count") == EXPECTED["models"] == len(rows)
              and models.get("aircraft_total") == EXPECTED["aircraft_count"]
              and models.get("unprofiled_aircraft") == 0
              and models.get("synthetic") is True)
        brief = []
        for m in rows:
            ids = m.get("aircraft_ids") or []
            live_mix = {band: sum(1 for i in ids if state.get(i, {}).get("risk") == band)
                        for band in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
            unreported = sum(1 for i in ids
                             if state.get(i, {}).get("rul_cycles") is None)
            ok = ok and (
                m.get("aircraft_count") == EXPECTED["per_model"] == len(ids)
                and (m.get("risk_mix") or {}) == live_mix
                and m.get("critical") == live_mix["CRITICAL"] == EXPECTED["critical_per_model"]
                and m.get("ready_count") == len(ids) - live_mix["CRITICAL"]
                and m.get("unreported") == unreported
                and m.get("reporting_count") == len(ids) - unreported
                and bool(m.get("image")) and bool(m.get("role")) and bool(m.get("engines"))
            )
            brief.append(f"{m.get('model_code')} {live_mix['CRITICAL']}C/"
                         f"{live_mix['HIGH']}H/{live_mix['MEDIUM']}M/{live_mix['LOW']}L "
                         f"avg {m.get('avg_rul_cycles')}")
        # the verified edge fixture must stay inside the Su-30MKI block it is seeded for
        su30 = next((m for m in rows if m.get("model_code") == "SU30MKI"), None)
        ok = ok and su30 is not None and "AC-017" in (su30.get("aircraft_ids") or [])
        check(name10, ok, "; ".join(brief))

    # --- 11) deny-by-default: unauthenticated /models must be refused ------------
    deny_ok, deny_detail = False, "no response"
    try:
        r = requests.get(args.ground + "/models", timeout=TIMEOUT)
        deny_ok, deny_detail = r.status_code == 401, f"HTTP {r.status_code}"
    except requests.exceptions.RequestException as exc:
        deny_detail = short_err(exc)
    check("11. ground /models without credentials -> 401", deny_ok, deny_detail)

    # --- summary ----------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("-" * 60)
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL")
    if n_fail:
        print("FAILED checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("AARIS pipeline verified end-to-end: edge -> ground -> CP-SAT plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
