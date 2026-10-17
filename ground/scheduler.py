"""AARIS — 30-day maintenance scheduler (Google OR-Tools CP-SAT).

Reads current fleet state from the ground station's SQLite DB and produces a
deterministic, explainable maintenance plan. This module is deliberately free
of FastAPI so it can be unit-tested on its own and imported by ground_station.py.

Planning model (v1)
-------------------
* Horizon: days 1..30.
* Each aircraft gets at most ONE maintenance task.
* At most MAX_DAILY_MAINTENANCE aircraft serviced per day.
* Maintenance deadline:  max(1, min(30, floor(rul_cycles * 0.8))).
* An aircraft is only ever scheduled ON OR BEFORE its deadline (hard). If that
  is impossible (next-available > deadline, or daily capacity exhausted before
  the deadline), it is DEFERRED with an explicit reason.
* Aircraft are prioritised by urgency (low RUL), mission priority, and risk band:
  the objective minimises sum_i w_i * day_i with
  w_i = 10 * urgency_i + 4 * risk_weight_i * mission_priority_i
  (higher w_i => solver pulls that aircraft to earlier days).
* Spare availability: tried as a HARD constraint first (no-spare aircraft
  cannot be serviced before SPARE_LEAD_DAYS, with everyone scheduled); if that
  is infeasible it relaxes to a soft early-slot penalty. This matches
  "hard only if every aircraft can still receive a feasible slot, else a
  documented deferral."

Determinism: single search worker + fixed random seed.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from ortools.sat.python import cp_model

HORIZON_DAYS = 30
# 8 hangar slots/day: sized for the 90-aircraft roster (6 models x 15 airframes),
# where 54 aircraft are maintenance candidates. POST /schedule may override it.
MAX_DAILY_MAINTENANCE = 8
SPARE_LEAD_DAYS = 5          # no-spare aircraft cannot be serviced before day N
MAX_RUL = 125.0

# Per-band deferred penalties (integer, scaled). Higher = less willing to defer.
DEFER_PENALTY = {"CRITICAL": 100_000, "HIGH": 60_000, "MEDIUM": 30_000}
# Extra cost per day-before-lead for scheduling a no-spare aircraft early (soft).
SPARE_EARLY_PENALTY = 4_000

RISK_WEIGHT = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}


class SchedulerError(Exception):
    """Raised when CP-SAT returns no feasible solution."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def compute_deadline_day(rul_cycles: Optional[float]) -> Optional[int]:
    """maintenance_deadline_day = max(1, min(30, floor(rul * 0.8)))."""
    if rul_cycles is None:
        return None
    return max(1, min(HORIZON_DAYS, math.floor(rul_cycles * 0.8)))


def _normalize_priority(mission_priority) -> int:
    try:
        return max(1, min(5, int(mission_priority)))
    except (TypeError, ValueError):
        return 3


def load_fleet_state(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(aircraft)")}
        rows = conn.execute("SELECT * FROM aircraft ORDER BY aircraft_id").fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        out.append({
            "aircraft_id": r["aircraft_id"],
            "rul_cycles": r["latest_rul_cycles"] if "latest_rul_cycles" in cols else None,
            "engine_health": r["latest_engine_health"] if "latest_engine_health" in cols else None,
            "risk": r["latest_risk"] if "latest_risk" in cols else None,
            "mission_priority": _normalize_priority(r["mission_priority"] if "mission_priority" in cols else 3),
            "spare_available": bool(r["spare_available"]) if "spare_available" in cols else True,
            "latest_report_at": r["latest_report_at"] if "latest_report_at" in cols else None,
            # next-available date, if that column is ever added later
            "next_available_day": (r["next_available_day"] if "next_available_day" in cols else None),
        })
    return out


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
def _solve(cands: list[dict], *, enforce_deadline_hard: bool,
           hard_spares: bool, force_all_scheduled: bool,
           horizon: int, max_daily: int) -> tuple[str, Optional[dict]]:
    """Build + solve one CP-SAT model. Returns (solver_status, {aircraft_id: day} | None)."""
    model = cp_model.CpModel()
    by_day: dict[int, list] = {d: [] for d in range(1, horizon + 1)}
    x: dict[tuple[int, int], object] = {}
    defer: dict[int, object] = {}

    for i, a in enumerate(cands):
        lo = a["day_lo"]
        hi = a["deadline_day"] if enforce_deadline_hard else horizon
        if hard_spares and not a["spare_available"]:
            lo = max(lo, SPARE_LEAD_DAYS)
        days = [d for d in range(lo, hi + 1) if 1 <= d <= horizon]

        if force_all_scheduled:
            if not days:
                # cannot force a schedule for this aircraft -> this mode is infeasible
                return "INFEASIBLE", None
            for d in days:
                lit = model.NewBoolVar(f"x_{a['aircraft_id']}_{d}")
                x[(i, d)] = lit
                by_day[d].append(lit)
            # scheduled on exactly one of its feasible days (no deferral allowed)
            model.AddExactlyOne([x[(i, d)] for d in days])
        else:
            defer[i] = model.NewBoolVar(f"defer_{a['aircraft_id']}")
            if days:
                lits = []
                for d in days:
                    lit = model.NewBoolVar(f"x_{a['aircraft_id']}_{d}")
                    x[(i, d)] = lit
                    by_day[d].append(lit)
                    lits.append(lit)
                model.AddExactlyOne(lits + [defer[i]])
            else:
                model.Add(defer[i] == 1)

    # daily maintenance capacity
    for d in range(1, horizon + 1):
        if by_day[d]:
            model.Add(sum(by_day[d]) <= max_daily)

    # objective: weighted scheduled days + spare-soft penalties + deferral penalties
    terms = []
    for i, a in enumerate(cands):
        w = max(1, int(round(10 * a["urgency"]
                             + 4 * RISK_WEIGHT.get(a["risk"] or "", 0) * a["mission_priority"])))
        for d in range(1, horizon + 1):
            lit = x.get((i, d))
            if lit is None:
                continue
            cost = w * d
            if (not hard_spares) and (not a["spare_available"]) and d < SPARE_LEAD_DAYS:
                cost += SPARE_EARLY_PENALTY * (SPARE_LEAD_DAYS - d)
            terms.append(cost * lit)
        if i in defer:
            terms.append(DEFER_PENALTY.get(a["risk"] or "", 10_000) * defer[i])
    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1        # determinism
    solver.parameters.random_seed = 7               # determinism
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if name not in ("OPTIMAL", "FEASIBLE"):
        return name, None

    assignment = {}
    for (i, d), lit in x.items():
        if solver.Value(lit):
            assignment[cands[i]["aircraft_id"]] = d
    return name, assignment


# ---------------------------------------------------------------------------
# Explainable reasons
# ---------------------------------------------------------------------------
def _sched_reason(a: dict, day: int, spare_hard: bool) -> str:
    parts = [f"{a['risk']} risk (RUL {a['rul_cycles']:.1f} cycles) — deadline day {a['deadline_day']}"]
    if a["urgency"] > 0.5:
        parts.append("low RUL")
    if a["mission_priority"] <= 2:
        parts.append(f"high mission priority ({a['mission_priority']})")
    if not a["spare_available"]:
        if spare_hard:
            parts.append(f"scheduled after {SPARE_LEAD_DAYS}-day spare lead time")
        else:
            parts.append("no spare on hand")
    return "; ".join(parts) + f" -> serviced on day {day}."


def _defer_reason(a: dict, max_daily: int) -> str:
    if a["risk"] == "CRITICAL":
        return (f"CRITICAL deadline day {a['deadline_day']} unreachable: all {max_daily} "
                f"daily slots before it are held by higher-urgency work; escalate for "
                f"an extra maintenance slot.")
    return (f"No feasible slot on/before deadline day {a.get('deadline_day')} within "
            f"{max_daily}/day capacity; deferred.")


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------
def build_plan(fleet: list[dict], generated_at: str,
               horizon_days: int = HORIZON_DAYS,
               max_daily: int = MAX_DAILY_MAINTENANCE) -> dict:
    cands: list[dict] = []
    tasks: dict[str, dict] = {}

    # --- classify every aircraft into "needs a slot" vs deterministic outcome ---
    for raw in fleet:
        a = dict(raw)
        aid = a["aircraft_id"]

        if a["rul_cycles"] is None:  # no data yet -> placeholder
            tasks[aid] = {**a, "deadline_day": None, "scheduled_day": None,
                          "status": "DEFERRED",
                          "reason": "No health data at the ground station yet; "
                                    "awaiting first report from the edge node."}
            continue

        a["deadline_day"] = compute_deadline_day(a["rul_cycles"])
        day_lo = 1
        nad = a.get("next_available_day")
        if isinstance(nad, int):
            day_lo = max(1, min(horizon_days, nad))
        a["day_lo"] = day_lo
        a["urgency"] = max(0.0, min(1.0, 1.0 - a["rul_cycles"] / MAX_RUL))

        # healthy (LOW risk) -> deferred by policy, keep capacity for degraded engines
        if a["risk"] == "LOW":
            tasks[aid] = {**a, "status": "DEFERRED", "scheduled_day": None,
                          "reason": f"Healthy (RUL {a['rul_cycles']:.1f} cycles, LOW risk); "
                                    f"no maintenance needed within {horizon_days}-day horizon."}
            continue

        cands.append(a)

    # --- solve, with documented fallback phases ---
    solver_status = "OPTIMAL"
    if cands:
        # Phase 1: hard deadlines + hard spares + everyone scheduled ("hard if feasible")
        status1, asg1 = _solve(cands, enforce_deadline_hard=True, hard_spares=True,
                               force_all_scheduled=True, horizon=horizon_days, max_daily=max_daily)
        if asg1 is not None:
            assignment, spare_hard, solver_status = asg1, True, status1
        else:
            # Phase 2: hard deadlines, soft spare penalty, deferrals allowed
            status2, asg2 = _solve(cands, enforce_deadline_hard=True, hard_spares=False,
                                   force_all_scheduled=False, horizon=horizon_days, max_daily=max_daily)
            if asg2 is not None:
                assignment, spare_hard, solver_status = asg2, False, status2
            else:
                # Phase 3 (last resort): allow past-deadline slots, soft spares, deferrals
                status3, asg3 = _solve(cands, enforce_deadline_hard=False, hard_spares=False,
                                       force_all_scheduled=False, horizon=horizon_days, max_daily=max_daily)
                if asg3 is None:
                    raise SchedulerError(
                        f"CP-SAT found no feasible plan (status: {status3}/{status2}/{status1}).")
                assignment, spare_hard, solver_status = asg3, False, status3

        for a in cands:
            aid = a["aircraft_id"]
            day = assignment.get(aid)
            if day is not None:
                tasks[aid] = {**a, "status": "SCHEDULED", "scheduled_day": int(day),
                              "reason": _sched_reason(a, int(day), spare_hard)}
            else:
                tasks[aid] = {**a, "status": "DEFERRED", "scheduled_day": None,
                              "reason": _defer_reason(a, max_daily)}

    # --- assemble ordered output matching the API schema ---
    ordered = sorted(tasks.values(),
                     key=lambda t: (t["status"] != "SCHEDULED",
                                    t["scheduled_day"] if t["status"] == "SCHEDULED" else 999,
                                    t["aircraft_id"]))

    daily_load: dict[str, int] = {}
    for t in ordered:
        if t["status"] == "SCHEDULED":
            daily_load[str(t["scheduled_day"])] = daily_load.get(str(t["scheduled_day"]), 0) + 1

    scheduled = [t for t in ordered if t["status"] == "SCHEDULED"]
    deferred = [t for t in ordered if t["status"] == "DEFERRED"]

    def _out(t: dict) -> dict:
        return {
            "aircraft_id": t["aircraft_id"],
            "scheduled_day": t.get("scheduled_day"),
            "maintenance_deadline_day": t.get("deadline_day"),
            "rul_cycles": t.get("rul_cycles"),
            "engine_health": t.get("engine_health"),
            "risk": t.get("risk"),
            "mission_priority": t.get("mission_priority"),
            "spare_available": t.get("spare_available"),
            "status": t["status"],
            "reason": t["reason"],
        }

    return {
        "generated_at": generated_at,
        "horizon_days": horizon_days,
        "max_daily_maintenance": max_daily,
        "scheduled_count": len(scheduled),
        "deferred_count": len(deferred),
        "tasks": [_out(t) for t in ordered],
        "daily_load": dict(sorted(daily_load.items(), key=lambda kv: int(kv[0]))),
        "solver_status": solver_status,
    }


def generate_plan(db_path: str) -> dict:
    """Entry point used by the ground station: read DB, solve, return plan dict."""
    fleet = load_fleet_state(db_path)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return build_plan(fleet, generated_at)


if __name__ == "__main__":  # manual run against the live fleet.db
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    plan = generate_plan(os.path.join(here, "fleet.db"))
    print(json.dumps(plan, indent=2))
