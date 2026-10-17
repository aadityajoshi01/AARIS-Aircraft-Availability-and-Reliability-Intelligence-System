"""AARIS — Ground Station.

A standalone FastAPI service that receives JSON health reports emitted by
`edge/edge_node.py` and maintains fleet-wide state in SQLite (`ground/fleet.db`).

Responsibilities:
  * ingest edge reports (exact schema produced by edge /predict)
  * persist every report as history; keep only the newest report as each
    aircraft's latest state (guarded by report timestamp)
  * expose fleet status + per-aircraft history for ops dashboards
  * load SYNTHETIC demonstration aircraft profiles (type, maintenance
    metadata, issue history) from aircraft_profiles.json into the
    aircraft_profiles table and attach them to GET /fleet/{aircraft_id}
    (additive key "profile"; GET /fleet is unchanged). The same profiles are
    grouped per aircraft MODEL (6 models x 15 airframes) by GET /models.
  * (placeholder) scheduling endpoints — the OR-Tools CP-SAT solver lands next

This service does NOT touch anything under `edge/` or `results/ridge_model/`.

Run (from the `ground/` directory):
    uvicorn ground_station:app --reload --port 8001
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator

import scheduler  # CP-SAT maintenance planner (same directory; no FastAPI deps)
import security   # JWT users + static edge-device tokens + RBAC + audit sink
from security import Identity

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
GROUND_DIR = os.path.dirname(os.path.abspath(__file__))
FLEET_CONFIG_JSON = os.path.join(GROUND_DIR, "fleet_config.json")
# SYNTHETIC demonstration profiles (see ground/aircraft_profiles.json +
# ground/make_profiles.py). No real IAF operational or maintenance data.
AIRCRAFT_PROFILES_JSON = os.path.join(GROUND_DIR, "aircraft_profiles.json")
DB_PATH = os.path.join(GROUND_DIR, "fleet.db")

MAX_RUL = 125.0          # must match edge model.max_rul
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# Time helpers (UTC ISO-8601, second precision, Z-suffixed)
# ---------------------------------------------------------------------------
def _to_utc(dt: datetime) -> datetime:
    dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt(dt: datetime) -> str:
    return _to_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> datetime:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return _to_utc(datetime.fromisoformat(s))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# SQLite access
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS aircraft (
    aircraft_id              TEXT PRIMARY KEY,
    base                     TEXT,
    mission_priority         INTEGER,
    spare_available          INTEGER,
    latest_rul_cycles        REAL,
    latest_engine_health     REAL,
    latest_risk              TEXT,
    latest_top_issues        TEXT,
    latest_recommended_action TEXT,
    latest_report_at         TEXT,
    updated_at               TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    aircraft_id          TEXT NOT NULL REFERENCES aircraft(aircraft_id),
    timestamp            TEXT NOT NULL,
    engine_id            TEXT,
    rul_cycles           REAL NOT NULL,
    engine_health        REAL NOT NULL,
    risk                 TEXT NOT NULL,
    top_issues           TEXT,
    recommended_action   TEXT,
    model                TEXT,
    input_window_shape   TEXT,
    raw_payload          TEXT NOT NULL,
    received_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_aircraft_ts
    ON reports (aircraft_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS aircraft_profiles (
    aircraft_id   TEXT PRIMARY KEY,
    profile_json  TEXT NOT NULL,
    stored_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS maintenance_plans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at  TEXT NOT NULL,
    plan_json     TEXT NOT NULL,
    stored_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    actor_id      TEXT NOT NULL,
    actor_role    TEXT NOT NULL,
    action        TEXT NOT NULL,
    resource      TEXT,
    aircraft_id   TEXT,
    outcome       TEXT NOT NULL,
    details_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_ts
    ON audit_logs (timestamp DESC, id DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    aircraft_id      TEXT NOT NULL REFERENCES aircraft(aircraft_id),
    report_id        INTEGER,
    rul_cycles       REAL,
    engine_health    REAL,
    severity         TEXT NOT NULL,
    message          TEXT,
    status           TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at       TEXT NOT NULL,
    acknowledged_at  TEXT,
    acknowledged_by  TEXT,
    cleared_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_status_created
    ON alerts (status, created_at DESC, id DESC);
"""


def _load_profiles(conn: sqlite3.Connection) -> int:
    """Upsert the SYNTHETIC aircraft profiles into aircraft_profiles.

    The 'reports' block (synthetic trend used by seed_demo.py at ingest time)
    is not stored in the profile row — live report history comes from the
    reports table. Missing file is fine: profiles are an additive enrichment.
    """
    if not os.path.exists(AIRCRAFT_PROFILES_JSON):
        print(f"[ground] note: {AIRCRAFT_PROFILES_JSON} not found; "
              "GET /fleet/{id} will return profile=null.")
        return 0
    with open(AIRCRAFT_PROFILES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    roster = {r["aircraft_id"] for r in conn.execute("SELECT aircraft_id FROM aircraft")}
    loaded = 0
    for a in data.get("aircraft", []):
        aid = a.get("aircraft_id")
        if not aid:
            continue
        if aid not in roster:
            print(f"[ground] warning: profile for {aid} skipped (not in fleet_config roster)")
            continue
        profile = {k: v for k, v in a.items() if k != "reports"}
        reports = a.get("reports") or {}
        profile["reports_note"] = reports.get("note")
        profile["synthetic"] = True   # provenance flag: never real operational data
        conn.execute(
            """INSERT OR REPLACE INTO aircraft_profiles
                 (aircraft_id, profile_json, stored_at) VALUES (?,?,?)""",
            (aid, json.dumps(profile), _now()),
        )
        loaded += 1
    return loaded


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a fleet.db was first created.

    CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so an older
    alerts table needs the cleared_at column that the recovery transition uses.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
    if "cleared_at" not in cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN cleared_at TEXT")
        print("[ground] migrated alerts: added cleared_at.")


def _init_db_and_seed() -> None:
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        if not os.path.exists(FLEET_CONFIG_JSON):
            raise RuntimeError(f"fleet_config.json not found at {FLEET_CONFIG_JSON}")
        with open(FLEET_CONFIG_JSON) as f:
            cfg = json.load(f)
        fleet = cfg["fleet"] if isinstance(cfg, dict) and "fleet" in cfg else cfg
        if not isinstance(fleet, list) or not fleet:
            raise RuntimeError("fleet_config.json must contain a non-empty 'fleet' list")
        rows = []
        for a in fleet:
            aid = a.get("aircraft_id")
            if not aid:
                raise RuntimeError(f"fleet entry missing aircraft_id: {a!r}")
            rows.append((
                aid,
                a.get("base"),
                int(a.get("mission_priority", 0)),
                1 if bool(a.get("spare_available", False)) else 0,
            ))
        conn.executemany(
            """INSERT OR IGNORE INTO aircraft
                 (aircraft_id, base, mission_priority, spare_available)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        loaded = _load_profiles(conn)
        conn.commit()
        print(f"[ground] SQLite at {DB_PATH}; seeded/verified {len(rows)} aircraft; "
              f"loaded {loaded} synthetic profiles.")
        if security.USING_DEV_DEVICE_TOKENS:
            # Tokens are never printed — only the fact that they are defaults.
            print(f"[ground] WARNING: {len(security.DEVICE_TOKENS)} edge device "
                  "tokens are the guessable local-demo defaults "
                  "(edge-ac###-dev-token). Fine on localhost for the demo; set "
                  "EDGE_<AIRCRAFT>_TOKEN for anything real.")
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db_and_seed()
    yield
    print("[ground] shutting down")


app = FastAPI(
    title="AARIS Ground Station",
    description="Fleet state + report history for AARIS edge health reports.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Audit log writer (registered as the security module's sink).
# Never pass tokens, passwords, or raw report payloads into these events.
# ---------------------------------------------------------------------------
def _audit(actor_id: str, actor_role: str, action: str, resource: str,
           aircraft_id: Optional[str], outcome: str,
           details_json: Optional[str]) -> None:
    try:
        conn = _connect()
        try:
            conn.execute(
                """INSERT INTO audit_logs
                     (timestamp, actor_id, actor_role, action, resource,
                      aircraft_id, outcome, details_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (_now(), actor_id, actor_role, action, resource or None,
                 aircraft_id, outcome, details_json),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:  # audit failure must never break a request
        print(f"[ground] audit write failed: {exc}")


security.register_audit_sink(_audit)


def _sanitize_non_finite(obj: Any) -> Any:
    """Replace NaN/Inf with strings so 422 error bodies are JSON-serializable.

    FastAPI's default validation handler echoes the offending input back in the
    422 detail; if that input is NaN/Inf the default JSON encoder crashes with a
    500 instead of returning the 422.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/ingest_report":
        security.audit("report_ingest", "rejected", resource="/ingest_report",
                       details={"reason": "validation_422"})
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_non_finite(jsonable_encoder(exc.errors()))},
    )


# ---------------------------------------------------------------------------
# 0) POST /auth/login — OAuth2 password form -> short-lived JWT
# ---------------------------------------------------------------------------
@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()) -> dict:
    role = security.verify_password(form.username, form.password)
    if not role:
        security.audit("login", "denied", actor_id=form.username or "anonymous",
                       details={"reason": "bad_credentials"})
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = security.create_access_token(form.username, role)
    security.audit("login", "success", actor_id=form.username, actor_role=role,
                   resource="/auth/login")
    return {"access_token": token, "token_type": "bearer", "role": role}


# ---------------------------------------------------------------------------
# Pydantic request model — mirrors the exact edge /predict response
# ---------------------------------------------------------------------------
class EdgeReport(BaseModel):
    # required, validated core fields
    aircraft_id: str = Field(min_length=1)
    timestamp: datetime
    rul_cycles: float
    engine_health: float
    risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    # carried-through optional fields (present in edge output; tolerated if absent)
    engine_id: Optional[Union[int, str]] = None
    top_issues: list[dict] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    model: Optional[dict] = None
    input_window_shape: Optional[list[int]] = None

    @field_validator("aircraft_id")
    @classmethod
    def _strip_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("aircraft_id must not be blank")
        return v

    @field_validator("rul_cycles")
    @classmethod
    def _rul_range(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("rul_cycles must be finite (no NaN/Inf)")
        if v < 0 or v > MAX_RUL:
            raise ValueError(f"rul_cycles must be within [0, {MAX_RUL}]; got {v}")
        return v

    @field_validator("engine_health")
    @classmethod
    def _health_range(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("engine_health must be finite (no NaN/Inf)")
        if v < 0 or v > 100:
            raise ValueError(f"engine_health must be within [0, 100]; got {v}")
        return v


# ---------------------------------------------------------------------------
# State projection helpers
# ---------------------------------------------------------------------------
def _aircraft_state(row: sqlite3.Row) -> dict:
    """Normalized latest state for one aircraft."""
    try:
        top_issues = json.loads(row["latest_top_issues"]) if row["latest_top_issues"] else []
    except (TypeError, json.JSONDecodeError):
        top_issues = []
    return {
        "aircraft_id": row["aircraft_id"],
        "base": row["base"],
        "mission_priority": row["mission_priority"],
        "spare_available": bool(row["spare_available"]),
        "rul_cycles": row["latest_rul_cycles"],
        "engine_health": row["latest_engine_health"],
        "risk": row["latest_risk"],
        "top_issues": top_issues,
        "recommended_action": row["latest_recommended_action"],
        "latest_report_at": row["latest_report_at"],
    }


def _report_brief(row: sqlite3.Row) -> dict:
    try:
        top_issues = json.loads(row["top_issues"]) if row["top_issues"] else []
    except (TypeError, json.JSONDecodeError):
        top_issues = []
    return {
        "report_id": row["id"],
        "timestamp": row["timestamp"],
        "engine_id": row["engine_id"],
        "rul_cycles": row["rul_cycles"],
        "engine_health": row["engine_health"],
        "risk": row["risk"],
        "top_issues": top_issues,
        "recommended_action": row["recommended_action"],
        "received_at": row["received_at"],
    }


def _alert_brief(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "aircraft_id": row["aircraft_id"],
        "report_id": row["report_id"],
        "rul_cycles": row["rul_cycles"],
        "engine_health": row["engine_health"],
        "severity": row["severity"],
        "message": row["message"],
        "status": row["status"],
        "created_at": row["created_at"],
        "acknowledged_at": row["acknowledged_at"],
        "acknowledged_by": row["acknowledged_by"],
        "cleared_at": row["cleared_at"],
    }


# ---------------------------------------------------------------------------
# 1) POST /ingest_report
# ---------------------------------------------------------------------------
@app.post("/ingest_report", status_code=201)
def ingest_report(
    report: EdgeReport,
    identity: Identity = Depends(security.require_roles("edge_device", "admin")),
) -> dict:
    # An edge device may only report for ITS OWN assigned aircraft.
    if identity.kind == "device" and report.aircraft_id != identity.aircraft_id:
        security.audit("report_ingest", "denied", actor_id=identity.actor_id,
                       actor_role=identity.role, resource="/ingest_report",
                       aircraft_id=report.aircraft_id,
                       details={"assigned_aircraft": identity.aircraft_id,
                                "reason": "device_assigned_to_other_aircraft"})
        raise HTTPException(
            status_code=403,
            detail=(f"This edge device is assigned to {identity.aircraft_id}; "
                    f"reports for '{report.aircraft_id}' are not permitted."),
        )

    incoming_ts = _fmt(report.timestamp)

    conn = _connect()
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT * FROM aircraft WHERE aircraft_id = ?", (report.aircraft_id,)
        ).fetchone()
        if row is None:
            security.audit("report_ingest", "rejected", actor_id=identity.actor_id,
                           actor_role=identity.role, resource="/ingest_report",
                           aircraft_id=report.aircraft_id,
                           details={"reason": "not_in_roster"})
            raise HTTPException(
                status_code=422,
                detail=(
                    f"aircraft_id '{report.aircraft_id}' is not in the fleet roster. "
                    f"Known IDs come from fleet_config.json (e.g. AC-001 .. AC-090)."
                ),
            )

        # persist every report as history
        cur.execute(
            """INSERT INTO reports
                 (aircraft_id, timestamp, engine_id, rul_cycles, engine_health, risk,
                  top_issues, recommended_action, model, input_window_shape,
                  raw_payload, received_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                report.aircraft_id,
                incoming_ts,
                None if report.engine_id is None else str(report.engine_id),
                report.rul_cycles,
                report.engine_health,
                report.risk,
                json.dumps(report.top_issues),
                report.recommended_action,
                json.dumps(report.model),
                json.dumps(report.input_window_shape),
                report.model_dump_json(),
                _now(),
            ),
        )
        report_id = cur.lastrowid

        # update latest state ONLY if the incoming timestamp is strictly newer
        stored = row["latest_report_at"]
        is_newer = (stored is None) or (_parse(incoming_ts) > _parse(stored))
        if is_newer:
            cur.execute(
                """UPDATE aircraft SET
                     latest_rul_cycles=?, latest_engine_health=?, latest_risk=?,
                     latest_top_issues=?, latest_recommended_action=?,
                     latest_report_at=?, updated_at=?
                   WHERE aircraft_id=?""",
                (
                    report.rul_cycles,
                    report.engine_health,
                    report.risk,
                    json.dumps(report.top_issues),
                    report.recommended_action,
                    incoming_ts,
                    _now(),
                    report.aircraft_id,
                ),
            )

        # Urgent critical alert: any CRITICAL health report keeps ONE active
        # alert per aircraft — refreshed on each new critical reading, created
        # on the first — so repeated reports don't flood the operator's queue.
        alert_created = False
        alert_id = None
        alert_cleared = 0
        if report.risk == "CRITICAL":
            message = (f"CRITICAL health report for {report.aircraft_id}: "
                       f"RUL {report.rul_cycles} cycles "
                       f"(engine health {report.engine_health}%). Ground the aircraft.")
            existing = cur.execute(
                "SELECT id FROM alerts WHERE aircraft_id = ? AND status = 'ACTIVE' "
                "ORDER BY id LIMIT 1",
                (report.aircraft_id,),
            ).fetchone()
            if existing is not None:
                alert_id = existing["id"]
                cur.execute(
                    """UPDATE alerts
                       SET report_id=?, rul_cycles=?, engine_health=?, message=?,
                           created_at=?
                       WHERE id=?""",
                    (report_id, report.rul_cycles, report.engine_health,
                     message, _now(), alert_id),
                )
            else:
                cur.execute(
                    """INSERT INTO alerts
                         (aircraft_id, report_id, rul_cycles, engine_health, severity,
                          message, status, created_at)
                       VALUES (?,?,?,?, 'CRITICAL', ?, 'ACTIVE', ?)""",
                    (report.aircraft_id, report_id, report.rul_cycles,
                     report.engine_health, message, _now()),
                )
                alert_id = cur.lastrowid
                alert_created = True
        elif is_newer:
            # The aircraft's own latest reading now says it is no longer
            # critical, so its open alert is auto-cleared instead of piling up
            # in the operator's queue. Only a strictly-newer report can do this:
            # a stale out-of-order healthy reading must not close an open alert.
            cur.execute(
                """UPDATE alerts
                   SET status = 'CLEARED', cleared_at = ?
                   WHERE aircraft_id = ? AND status = 'ACTIVE'""",
                (_now(), report.aircraft_id),
            )
            alert_cleared = cur.rowcount

        conn.commit()

        latest = cur.execute(
            "SELECT * FROM aircraft WHERE aircraft_id = ?", (report.aircraft_id,)
        ).fetchone()

        security.audit("report_ingest", "success", actor_id=identity.actor_id,
                       actor_role=identity.role, resource="/ingest_report",
                       aircraft_id=report.aircraft_id,
                       details={"report_id": report_id, "state_updated": bool(is_newer),
                                "alert_created": alert_created,
                                "alert_cleared": alert_cleared})

        return {
            "report_id": report_id,
            "state_updated": bool(is_newer),
            "alert_created": alert_created,
            "alert_id": alert_id,
            "alert_cleared": alert_cleared,
            "aircraft": _aircraft_state(latest),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2) GET /fleet
# ---------------------------------------------------------------------------
@app.get("/fleet")
def get_fleet(
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM aircraft ORDER BY aircraft_id"
        ).fetchall()
    finally:
        conn.close()
    fleet = [_aircraft_state(r) for r in rows]
    security.audit("fleet_view", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/fleet",
                   details={"count": len(fleet)})
    return {"count": len(fleet), "fleet": fleet}


# ---------------------------------------------------------------------------
# 3) GET /fleet/{aircraft_id}
# ---------------------------------------------------------------------------
@app.get("/fleet/{aircraft_id}")
def get_aircraft(
    aircraft_id: str,
    history_limit: int = Query(20, ge=1, le=500),
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM aircraft WHERE aircraft_id = ?", (aircraft_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404,
                               detail=f"No aircraft with id '{aircraft_id}'.")
        hist = conn.execute(
            """SELECT * FROM reports WHERE aircraft_id = ?
               ORDER BY timestamp DESC, id DESC LIMIT ?""",
            (aircraft_id, history_limit),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM reports WHERE aircraft_id = ?", (aircraft_id,)
        ).fetchone()["n"]
        prow = conn.execute(
            "SELECT profile_json FROM aircraft_profiles WHERE aircraft_id = ?",
            (aircraft_id,),
        ).fetchone()
    finally:
        conn.close()

    profile = None
    if prow is not None:
        try:
            profile = json.loads(prow["profile_json"])
        except (TypeError, json.JSONDecodeError):
            profile = None

    security.audit("aircraft_view", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/fleet/{aircraft_id}",
                   aircraft_id=aircraft_id, details={"report_count": total})
    return {
        "aircraft": _aircraft_state(row),
        "report_count": total,
        "history": [_report_brief(r) for r in hist],
        # Additive (synthetic demo enrichment); None when profiles are absent.
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# 3a) GET /models — per-aircraft-MODEL profiles (synthetic demo enrichment)
# ---------------------------------------------------------------------------
def _model_aggregate(rows: list[dict], plan_tasks: dict) -> dict:
    """Aggregate one model's airframes into a model-level profile."""
    first = rows[0]
    ids = [r["aircraft_id"] for r in rows]
    reported = [r for r in rows if r["rul_cycles"] is not None]
    risk_mix = {band: sum(1 for r in rows if r["risk"] == band)
                for band in RISK_LEVELS}
    system_counts: dict[str, int] = {}
    open_issues = 0
    for r in rows:
        for issue in (r["profile"] or {}).get("issue_history", []) or []:
            system = issue.get("system") or "UNSCHEDULED"
            system_counts[system] = system_counts.get(system, 0) + 1
            if issue.get("status") == "OPEN":
                open_issues += 1

    return {
        "model_code": first["model_code"],
        "type": first["type"],
        "role": first["role"],
        "origin": first["origin"],
        "engines": first["engines"],
        "image": first["image"],
        "aircraft_count": len(rows),
        "aircraft_ids": ids,
        "reporting_count": len(reported),
        "unreported": len(rows) - len(reported),
        "risk_mix": risk_mix,
        "critical": risk_mix["CRITICAL"],
        # ready == flies today: not CRITICAL (the grounding band) — point estimate.
        "ready_count": len(rows) - risk_mix["CRITICAL"],
        "at_risk_count": risk_mix["CRITICAL"] + risk_mix["HIGH"],
        "avg_rul_cycles": (round(sum(r["rul_cycles"] for r in reported) / len(reported), 2)
                           if reported else None),
        "avg_engine_health": (round(sum(r["engine_health"] for r in reported) / len(reported), 2)
                              if reported else None),
        "lowest_rul_cycles": (min(r["rul_cycles"] for r in reported) if reported else None),
        "oldest_report_at": (min((r["latest_report_at"] for r in reported), default=None)),
        "latest_report_at": (max((r["latest_report_at"] for r in reported), default=None)),
        "bases": sorted({r["base"] for r in rows if r.get("base")}),
        "spares_missing": sum(1 for r in rows if not r["spare_available"]),
        "open_issues": open_issues,
        "issue_systems": [
            {"system": s, "count": c} for s, c in
            sorted(system_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
        ],
        "scheduled_in_plan": sum(1 for i in ids
                                 if plan_tasks.get(i, {}).get("status") == "SCHEDULED"),
        "deferred_in_plan": sum(1 for i in ids
                                if plan_tasks.get(i, {}).get("status") == "DEFERRED"),
        "aircraft": [
            {
                "aircraft_id": r["aircraft_id"],
                "base": r["base"],
                "mission_priority": r["mission_priority"],
                "spare_available": r["spare_available"],
                "rul_cycles": r["rul_cycles"],
                "engine_health": r["engine_health"],
                "risk": r["risk"],
                "latest_report_at": r["latest_report_at"],
                "plan_status": plan_tasks.get(r["aircraft_id"], {}).get("status"),
                "plan_day": plan_tasks.get(r["aircraft_id"], {}).get("scheduled_day"),
            }
            for r in rows
        ],
    }


@app.get("/models")
def get_models(
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    """Model-level profiles for the synthetic demo fleet.

    Groups every airframe that has a synthetic profile (aircraft_profiles) by
    its aircraft model and aggregates live state (risk mix, average RUL, spares)
    plus the last persisted maintenance plan. Read-only and additive: GET /fleet
    and the scheduler are untouched, and the scheduler never reads profiles.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT a.*, p.profile_json FROM aircraft a
               LEFT JOIN aircraft_profiles p ON p.aircraft_id = a.aircraft_id
               ORDER BY a.aircraft_id"""
        ).fetchall()
        plan_row = conn.execute(
            "SELECT plan_json, generated_at FROM maintenance_plans ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    plan_tasks: dict[str, dict] = {}
    plan_generated_at = None
    if plan_row is not None:
        plan_generated_at = plan_row["generated_at"]
        try:
            for t in (json.loads(plan_row["plan_json"]) or {}).get("tasks", []):
                plan_tasks[t["aircraft_id"]] = t
        except (TypeError, json.JSONDecodeError):
            plan_tasks = {}

    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    unprofiled = 0
    for row in rows:
        state = _aircraft_state(row)
        profile = None
        if row["profile_json"]:
            try:
                profile = json.loads(row["profile_json"])
            except (TypeError, json.JSONDecodeError):
                profile = None
        if not profile or not profile.get("type"):
            unprofiled += 1
            continue
        key = str(profile.get("model_code") or profile["type"])
        if key not in order:
            order.append(key)
        grouped.setdefault(key, []).append({**state,
                                            "type": profile.get("type"),
                                            "model_code": key,
                                            "role": profile.get("role"),
                                            "origin": profile.get("origin"),
                                            "engines": profile.get("engines"),
                                            "image": profile.get("image"),
                                            "profile": profile})

    models = [_model_aggregate(grouped[key], plan_tasks) for key in order]
    security.audit("models_view", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/models",
                   details={"models": len(models),
                            "aircraft": sum(m["aircraft_count"] for m in models),
                            "unprofiled": unprofiled})
    return {
        "count": len(models),
        "aircraft_total": sum(m["aircraft_count"] for m in models),
        "unprofiled_aircraft": unprofiled,
        "plan_generated_at": plan_generated_at,
        # provenance flag: identities/dates/issue logs are synthetic demo data.
        "synthetic": True,
        "note": "Synthetic demonstration aircraft-model profiles. The per-type "
                "photos are illustrative type images only. No real IAF "
                "operational or maintenance data is used.",
        "models": models,
    }


# ---------------------------------------------------------------------------
# 3b) GET /alerts  — critical health alerts (newest first)
# ---------------------------------------------------------------------------
@app.get("/alerts")
def get_alerts(
    status: Literal["ACTIVE", "ACKNOWLEDGED", "CLEARED", "ALL"] = Query("ACTIVE"),
    limit: int = Query(100, ge=1, le=500),
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    conn = _connect()
    try:
        if status == "ALL":
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
    finally:
        conn.close()
    alerts = [_alert_brief(r) for r in rows]
    security.audit("alerts_view", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/alerts",
                   details={"status": status, "count": len(alerts)})
    return {"count": len(alerts), "status_filter": status, "alerts": alerts}


# ---------------------------------------------------------------------------
# 3c) POST /alerts/{alert_id}/ack  — acknowledge one alert (operator action)
# ---------------------------------------------------------------------------
@app.post("/alerts/{alert_id}/ack")
def ack_alert(
    alert_id: int,
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404,
                               detail=f"No alert with id {alert_id}.")
        if row["status"] != "ACTIVE":
            return {"acknowledged": False, "already": True,
                    "alert": _alert_brief(row)}
        conn.execute(
            """UPDATE alerts
               SET status='ACKNOWLEDGED', acknowledged_at=?, acknowledged_by=?
               WHERE id=?""",
            (_now(), identity.actor_id, alert_id),
        )
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
    finally:
        conn.close()
    security.audit("alert_ack", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/alerts/{id}/ack",
                   aircraft_id=updated["aircraft_id"],
                   details={"alert_id": alert_id})
    return {"acknowledged": True, "already": False, "alert": _alert_brief(updated)}


# ---------------------------------------------------------------------------
# 4) POST /schedule  — CP-SAT 30-day maintenance plan (persisted)
# ---------------------------------------------------------------------------
class ScheduleRequest(BaseModel):
    horizon_days: int = Field(default=scheduler.HORIZON_DAYS, ge=1, le=365)
    max_daily_maintenance: int = Field(default=scheduler.MAX_DAILY_MAINTENANCE, ge=1, le=20)


@app.post("/schedule")
def schedule(
    req: Optional[ScheduleRequest] = None,
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "admin")),
) -> dict:
    """Solve the fleet maintenance plan with OR-Tools CP-SAT and persist it."""
    body = req or ScheduleRequest()
    try:
        fleet = scheduler.load_fleet_state(DB_PATH)
        plan = scheduler.build_plan(fleet, _now(),
                                    horizon_days=body.horizon_days,
                                    max_daily=body.max_daily_maintenance)
    except scheduler.SchedulerError as exc:
        security.audit("scheduling", "failed", actor_id=identity.actor_id,
                       actor_role=identity.role, resource="/schedule",
                       details={"reason": str(exc)[:200]})
        raise HTTPException(status_code=503,
                            detail=f"No feasible maintenance plan found: {exc}")

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO maintenance_plans (generated_at, plan_json, stored_at) VALUES (?,?,?)",
            (plan["generated_at"], json.dumps(plan), _now()),
        )
        conn.commit()
        # Stamp plan_id and re-persist so POST /schedule and GET /schedule/latest
        # return byte-identical plans (dashboard can compare them directly).
        plan["plan_id"] = cur.lastrowid
        cur.execute(
            "UPDATE maintenance_plans SET plan_json = ? WHERE id = ?",
            (json.dumps(plan), plan["plan_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    security.audit("scheduling", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/schedule",
                   details={"plan_id": plan["plan_id"],
                            "solver_status": plan.get("solver_status"),
                            "scheduled": plan.get("scheduled_count"),
                            "deferred": plan.get("deferred_count")})
    return plan


# ---------------------------------------------------------------------------
# 5) GET /schedule/latest  — last persisted plan
# ---------------------------------------------------------------------------
@app.get("/schedule/latest")
def schedule_latest(
    identity: Identity = Depends(
        security.require_roles("maintenance_officer", "commander", "admin")),
) -> dict:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT plan_json FROM maintenance_plans ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    security.audit("schedule_view", "success", actor_id=identity.actor_id,
                   actor_role=identity.role, resource="/schedule/latest",
                   details={"available": row is not None})
    if row is None:
        return {
            "available": False,
            "plan": {
                "schedule": [],
                "note": "No schedule generated yet — run POST /schedule "
                        "(OR-Tools CP-SAT solver).",
            },
        }
    plan = json.loads(row["plan_json"])
    return {"available": True, "plan": plan}


# ---------------------------------------------------------------------------
# 6) GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    db_ok = os.path.exists(DB_PATH)
    aircraft_count = report_count = None
    latest_report_at = None
    error = None
    try:
        conn = _connect()
        try:
            conn.execute("SELECT 1;")
            aircraft_count = conn.execute("SELECT COUNT(*) n FROM aircraft").fetchone()["n"]
            report_count = conn.execute("SELECT COUNT(*) n FROM reports").fetchone()["n"]
            r = conn.execute("SELECT MAX(timestamp) t FROM reports").fetchone()["t"]
            latest_report_at = r
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover
        error = str(exc)

    return {
        "status": "ok" if (db_ok and error is None) else "degraded",
        "database": {"path": DB_PATH, "exists": db_ok, "error": error},
        "aircraft_count": aircraft_count,
        "report_count": report_count,
        "latest_report_at": latest_report_at,
    }


if __name__ == "__main__":  # convenience: python ground_station.py
    import uvicorn

    # GROUND_HOST=0.0.0.0 makes the base station reachable from a Jetson on the
    # same network (demo/local only); default stays loopback.
    host = os.environ.get("GROUND_HOST", "127.0.0.1")
    port = int(os.environ.get("GROUND_PORT", "8001"))
    uvicorn.run("ground_station:app", host=host, port=port, reload=True)
