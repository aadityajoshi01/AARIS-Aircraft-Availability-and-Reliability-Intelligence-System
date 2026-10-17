"""AARIS — Aircraft Availability & Reliability Intelligence System.

Plain Streamlit UI over the ground station HTTP API:
  GET /fleet            - latest state for every aircraft
  GET /fleet/{id}       - one aircraft + report history
  GET /models           - per-aircraft-MODEL profiles (6 models x 15 airframes)
  GET /schedule/latest  - last persisted CP-SAT maintenance plan

The Fleet Readiness section answers five operator questions from live data:
available now / at risk / must enter maintenance (and by when) / daily load /
deferrals and reasons. Fleet Availability = not-CRITICAL and not scheduled
for maintenance today, over the full roster — computed every render.

The Aircraft Models section gives each of the 6 models its own profile (type
photo, role, engines, and the health of its 15 airframes: risk mix, average and
lowest RUL, ready count, spares gaps, plan split, issue-prone systems), and
every airframe row in it opens that aircraft's own profile.

Sections are switched with a nav bar (st.segmented_control) whose value lives
in session state, so clicking an aircraft anywhere — a fleet table row, a plan
row, a critical-alert row — opens that aircraft's profile in one click.

Since the ground-station security layer these endpoints are access-controlled.
LIVE mode signs in with an ID + password (POST /auth/login -> short-lived
commander JWT held only in the Streamlit session, never displayed or logged).
Demo users: commander/commander123, maintainer/maintainer123, admin/admin123.
An advanced sidebar field still accepts a pasted DASHBOARD_SERVICE_TOKEN JWT.

Run (repo root, DEMO MODE needs neither ground station nor credentials):
  .venv/Scripts/python.exe -m streamlit run dashboard/app.py

Click the DEMO MODE button in the sidebar to browse from
dashboard/demo_data.json when the ground station is not running.
"""

import functools
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup and constants
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 5
APP_NAME = "AARIS"
APP_TAGLINE = "Aircraft Availability & Reliability Intelligence System"
HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_FILE = os.path.join(HERE, "assets", "aaris_logo.png")
# Browser-tab icon; degrades gracefully if the PNG is missing.
_LOGO = LOGO_FILE if os.path.exists(LOGO_FILE) else None
# Wide banner (emblem + wordmark + tagline) shown at the top of the sidebar.
HEADING_LOGO_FILE = os.path.join(HERE, "assets", "aaris_heading.png")
_HEADING_LOGO = HEADING_LOGO_FILE if os.path.exists(HEADING_LOGO_FILE) else None


def _type_image(profile: dict):
    """Resolve a profile's per-type reference photo, or None if absent.

    profile["image"] is just a filename (e.g. "su30mki.jpg"); photos live under
    dashboard/assets/plane_types/. These depict the aircraft TYPE only — the
    synthetic banner above the section already states no real airframe data.
    """
    name = (profile or {}).get("image")
    if not name:
        return None
    path = os.path.join(HERE, "assets", "plane_types", os.path.basename(str(name)))
    return path if os.path.exists(path) else None

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon=_LOGO or "✈️",
    layout="wide",
)

DEMO_FILE = os.path.join(HERE, "demo_data.json")
# Conformal calibration sidecar (produced by experiments/conformal_ridge.py).
# Optional: the profile hides the uncertainty section if the file is absent.
CONFORMAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "conformal", "risk_calibration.json")
CONFORMAL_MAX_RUL = 125.0  # same cap as the frozen model's clip range
GROUNDING_THRESHOLD = 30.0  # CRITICAL band boundary (RUL < 30 => ground)

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
# Muted colors, used only for risk status.
RISK_COLORS = {
    "CRITICAL": "#c0504d",   # muted red
    "HIGH": "#e07b39",       # muted orange
    "MEDIUM": "#d9b310",     # muted yellow
    "LOW": "#6aa84f",        # muted green
}

# Nav bar sections. A state-driven control (not st.tabs) because st.tabs cannot
# be switched from code — and clicking an aircraft must switch to its profile.
NAV_FLEET, NAV_MODELS, NAV_DETAIL, NAV_PLAN, NAV_ABOUT = SECTIONS = [
    "Fleet Readiness", "Aircraft Models", "Aircraft Detail",
    "Maintenance Plan", "About"]


# ---------------------------------------------------------------------------
# Data access (API or bundled demo file). All errors surface as plain text.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_demo_file(path: str, mtime: float) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"_error": f"demo_data.json not found at {path}"}
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": f"demo_data.json could not be read: {exc}"}


def load_demo() -> dict:
    """Read the bundled demo_data.json, or return an error marker.

    Cached on the file's mtime: DEMO MODE reads this snapshot several times per
    render (fleet, models, schedule, alerts, and one aircraft detail) and the
    90-aircraft capture is large enough that re-parsing it on every rerun makes
    the offline path feel sluggish.
    """
    try:
        mtime = os.path.getmtime(DEMO_FILE)
    except OSError:
        mtime = -1.0
    return _read_demo_file(DEMO_FILE, mtime)


@st.cache_data(show_spinner=False)
def load_conformal() -> dict | None:
    """Read the conformal calibration sidecar once; None if absent/malformed."""
    try:
        with open(CONFORMAL_FILE, "r", encoding="utf-8") as f:
            cal = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(cal, dict) or "90" not in cal.get("halfwidths_cycles", {}):
        return None
    return cal


def _grounding_phrase(lo: float, hi: float) -> str:
    """Plain-language set-membership statement about the 30-cycle threshold."""
    if hi < GROUNDING_THRESHOLD:
        return ("Even the optimistic end of the interval is below 30 cycles — "
                "grounding this engine is not a borderline call.")
    if lo >= GROUNDING_THRESHOLD:
        return ("Even the pessimistic end of the interval stays above 30 cycles — "
                "the flying decision is stable across the interval.")
    return ("The interval spans the 30-cycle grounding threshold — the decision is "
            "borderline; treat the point-estimate risk band as a flag, not a guarantee.")


NO_TOKEN_MSG = (
    "LIVE mode requires a ground-station service token, and none was provided.\n\n"
    "Set DASHBOARD_SERVICE_TOKEN (environment variable or the sidebar field) to "
    "a commander JWT — create one with:\n"
    '  curl -X POST http://127.0.0.1:8001/auth/login -d "username=commander&password=commander123"\n\n'
    "Or click the DEMO MODE button in the sidebar to browse the bundled demo "
    "data without any API or credentials."
)


def api_get(base_url: str, path: str, token: str):
    """GET one endpoint with the dashboard service token.

    Returns (data, None) or (None, error_message). The token is placed only
    in the Authorization header — it is never shown in the UI or printed."""
    if not token:
        return None, NO_TOKEN_MSG
    url = base_url.rstrip("/") + path
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS,
                            headers={"Authorization": f"Bearer {token}"})
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the ground station at {base_url}. Start it with:\n"
            "  cd ground && ../.venv/Scripts/python.exe -m uvicorn "
            "ground_station:app --port 8001\n"
            "Or click the DEMO MODE button in the sidebar to browse the "
            "bundled demo data."
        )
    except requests.exceptions.Timeout:
        return None, f"Ground station timed out ({TIMEOUT_SECONDS}s) on GET {path}."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed on GET {path}: {exc}"
    if resp.status_code == 401:
        return None, ("Ground station rejected the dashboard credentials "
                      f"(HTTP 401 on GET {path}) — the token is absent, "
                      "malformed, or expired (commander JWTs last 60 minutes). "
                      "Mint a fresh one with POST /auth/login.")
    if resp.status_code == 403:
        return None, ("Dashboard service account does not have permission for "
                      f"this endpoint (HTTP 403 on GET {path}) — it needs the "
                      "commander (or admin) role.")
    if resp.status_code != 200:
        return None, f"GET {path} returned HTTP {resp.status_code}."
    try:
        return resp.json(), None
    except (ValueError, json.JSONDecodeError):
        return None, f"GET {path} returned malformed JSON (not a valid payload)."


def api_post(base_url: str, path: str, token: str | None = None, data: dict | None = None):
    """POST one endpoint (form-encoded `data`), with an optional bearer token.

    Returns (json_or_None, error_or_None). Used for login and alert ack; the
    token is only ever placed in the Authorization header."""
    url = base_url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.post(url, timeout=TIMEOUT_SECONDS, headers=headers, data=data)
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the ground station at {base_url}. Start it with:\n"
            "  cd ground && set GROUND_HOST=0.0.0.0 && ../.venv/Scripts/python.exe "
            "-m uvicorn ground_station:app --host 0.0.0.0 --port 8001"
        )
    except requests.exceptions.Timeout:
        return None, f"Ground station timed out ({TIMEOUT_SECONDS}s) on POST {path}."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed on POST {path}: {exc}"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except (ValueError, json.JSONDecodeError):
            detail = resp.text[:200]
        return None, f"POST {path} failed (HTTP {resp.status_code}): {detail}"
    try:
        return resp.json(), None
    except (ValueError, json.JSONDecodeError):
        return {}, None


def login(base_url: str, username: str, password: str):
    """Authenticate against POST /auth/login (OAuth2 password form).

    Returns (token, role, error). Local demo credentials only."""
    data, err = api_post(base_url, "/auth/login",
                         data={"username": username, "password": password})
    if err:
        return None, None, err
    token = (data or {}).get("access_token")
    if not token:
        return None, None, "Login succeeded but no token was returned."
    return token, (data or {}).get("role", ""), None


def fetch_alerts(base_url: str, token: str):
    """Active critical alerts as (list, error_or_None)."""
    data, err = api_get(base_url, "/alerts", token)
    if err:
        return [], err
    return (data or {}).get("alerts", []), None


def ack_alert(base_url: str, token: str, alert_id: int):
    """Acknowledge one alert; returns error_or_None."""
    _, err = api_post(base_url, f"/alerts/{alert_id}/ack", token=token)
    return err


def open_profile(aircraft_id: str) -> None:
    """Point the Aircraft Detail section at one aircraft and show it.

    Called from click handlers only: the nav bar and the aircraft selectbox are
    instantiated later in the run (or in another section entirely), and writing
    a widget's state after that widget has rendered raises
    StreamlitAPIException. detail_id is a plain key that survives a section
    switch, so a click wins over whatever was selected before."""
    st.session_state.detail_id = aircraft_id
    st.session_state.nav_section = NAV_DETAIL


def sync_detail_id() -> None:
    """on_change for the Aircraft Detail picker: keep detail_id on the choice."""
    st.session_state.detail_id = st.session_state.detail_aircraft


def open_profile_on_select(df_key: str, ids: list) -> None:
    """st.dataframe on_select callback: click a row -> open that aircraft.

    Indices are positions in the dataframe as built, so the id list is passed
    alongside the key. Both event kinds are accepted: clicking the row marker
    reports a row, clicking any cell (the aircraft_id column included) reports
    that cell — either way one click lands on the profile. No selection is a
    no-op."""
    selection = st.session_state[df_key]["selection"]
    row = selection["rows"][0] if selection["rows"] else (
        selection["cells"][0][0] if selection["cells"] else None
    )
    if row is not None and 0 <= row < len(ids):
        open_profile(ids[row])


def selectable(df: pd.DataFrame, df_key: str, alt: str):
    """Render a per-aircraft table whose rows open that aircraft's profile."""
    return st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        key=df_key,
        on_select=functools.partial(open_profile_on_select, df_key,
                                    list(df["aircraft_id"])),
        selection_mode=["single-row", "single-cell"],
        alt=alt,
    )


def alert_rows(active: list, on_ack, key_prefix: str = "ack") -> None:
    """Render one row per active alert with Acknowledge and Profile buttons.

    `on_ack(alert_id)` returns error_or_None: LIVE posts to the ground station,
    DEMO MODE records it in session state (the bundled file is read-only)."""
    for a in active:
        col_info, col_ack, col_profile = st.columns([4, 1, 1])
        with col_info:
            st.markdown(
                f"**{a['aircraft_id']}** · RUL {a.get('rul_cycles')} cycles · "
                f"health {a.get('engine_health')}%  \n"
                f"{a.get('message', '')}  \n"
                f"<span style='color:gray'>raised {a.get('created_at', '')} UTC</span>",
                unsafe_allow_html=True,
            )
        with col_ack:
            if st.button("Acknowledge", key=f"{key_prefix}_{a['id']}", width="stretch"):
                err = on_ack(a["id"])
                if err:
                    st.error(err)
                st.rerun()
        with col_profile:
            # The alert banner can be rendered inside the auto-refresh fragment,
            # so switching sections needs an explicit app-wide rerun.
            if st.button("Profile", key=f"{key_prefix}_profile_{a['id']}",
                         width="stretch"):
                open_profile(a["aircraft_id"])
                st.rerun()


def demo_alerts():
    """Bundled critical-alert snapshot for DEMO MODE, as (list, captured_at)."""
    d = load_demo()
    payload = d.get("alerts") or {}
    return list(payload.get("alerts") or []), d.get("captured_at_utc", "")


def fetch_fleet(base_url: str, demo: bool, token: str) -> dict:
    """Fleet list as {'fleet': [...], 'error': str|None}."""
    if demo:
        d = load_demo()
        return {"fleet": d.get("fleet", {}).get("fleet", []), "error": d.get("_error")}
    data, err = api_get(base_url, "/fleet", token)
    if err:
        return {"fleet": [], "error": err}
    return {"fleet": data.get("fleet", []), "error": None}


def fetch_models(base_url: str, demo: bool, token: str) -> dict:
    """Per-model profiles as {'models': [...], 'note': str|None, 'error': str|None}."""
    if demo:
        d = load_demo()
        if d.get("_error"):
            return {"models": [], "note": None, "plan_generated_at": None,
                    "error": d["_error"]}
        payload = d.get("models") or {}
        models = list(payload.get("models") or [])
        return {"models": models,
                "note": payload.get("note"),
                "plan_generated_at": payload.get("plan_generated_at"),
                "error": None if models else (
                    "demo_data.json has no /models snapshot — re-run "
                    "demo/capture_demo_data.py against the seeded station.")}
    data, err = api_get(base_url, "/models", token)
    if err:
        return {"models": [], "note": None, "plan_generated_at": None, "error": err}
    return {"models": list(data.get("models") or []),
            "note": data.get("note"),
            "plan_generated_at": data.get("plan_generated_at"),
            "error": None}


def fetch_aircraft(base_url: str, demo: bool, aircraft_id: str, token: str) -> dict:
    if demo:
        d = load_demo()
        detail = d.get("aircraft", {}).get(aircraft_id)
        if detail is None:
            return {"detail": None, "error": None, "missing_in_demo": True}
        return {"detail": detail, "error": None, "missing_in_demo": False}
    data, err = api_get(base_url, f"/fleet/{aircraft_id}", token)
    return {"detail": None if err else data, "error": err, "missing_in_demo": False}


def fetch_schedule(base_url: str, demo: bool, token: str) -> dict:
    if demo:
        d = load_demo()
        payload = d.get("schedule", {})
        return {"payload": payload, "error": d.get("_error")}
    data, err = api_get(base_url, "/schedule/latest", token)
    return {"payload": None if err else data, "error": err}


# ---------------------------------------------------------------------------
# Fleet Availability KPI — computed from live state, never hardcoded.
#   available = aircraft NOT in CRITICAL risk AND not scheduled for
#               maintenance today (UTC)
#   fleet availability = available / total aircraft on the roster
# Unreported aircraft (risk=None) count as not-CRITICAL by this formula;
# the caption surfaces their number so the operator sees what is inferred.
# ---------------------------------------------------------------------------
def compute_readiness(fleet: list, plan: dict, today=None) -> dict:
    """Derive readiness counts from the fleet list and the latest plan."""
    today = today or datetime.now(timezone.utc).date()
    total = len(fleet)

    # Which plan day is "today"? Plan days are 1-based offsets from the date
    # the plan was generated. A stale/young/unparseable plan means no aircraft
    # is 'in maintenance today' — the honest reading, not a zero-filled guess.
    scheduled_today = set()
    plan_today_index = None
    if plan and plan.get("generated_at"):
        try:
            base_day = datetime.strptime(plan["generated_at"], "%Y-%m-%dT%H:%M:%SZ").date()
            plan_today_index = (today - base_day).days + 1
            if 1 <= plan_today_index <= plan.get("horizon_days", 30):
                scheduled_today = {
                    t["aircraft_id"] for t in plan.get("tasks", [])
                    if t.get("status") == "SCHEDULED"
                    and t.get("scheduled_day") == plan_today_index
                }
        except (ValueError, TypeError):
            plan_today_index = None  # unparseable timestamp -> no 'today' mapping

    critical_ids = {a["aircraft_id"] for a in fleet if a.get("risk") == "CRITICAL"}
    available_ids = [
        a["aircraft_id"] for a in fleet
        if a["aircraft_id"] not in critical_ids and a["aircraft_id"] not in scheduled_today
    ]

    def task_rows(status, max_day=None):
        rows = []
        for t in (plan or {}).get("tasks", []):
            if t.get("status") != status:
                continue
            if max_day is not None and (t.get("scheduled_day") or 999) > max_day:
                continue
            rows.append(t)
        return sorted(rows, key=lambda t: (t.get("scheduled_day") or 999, t["aircraft_id"]))

    return {
        "total": total,
        "available": len(available_ids),
        "availability": (len(available_ids) / total) if total else None,
        "critical": len(critical_ids),
        "at_risk": sum(1 for a in fleet if a.get("risk") in ("CRITICAL", "HIGH")),
        "unreported": sum(1 for a in fleet if a.get("rul_cycles") is None),
        "scheduled_today": sorted(scheduled_today),
        "plan_today_index": plan_today_index,
        "upcoming_7d": task_rows("SCHEDULED", max_day=7),
        "deferred": task_rows("DEFERRED"),
    }


# ---------------------------------------------------------------------------
# Sidebar — brand logo, DEMO MODE button, settings
# ---------------------------------------------------------------------------
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "refresh_nonce" not in st.session_state:
    st.session_state.refresh_nonce = 0
# LIVE-mode auth: a JWT minted by POST /auth/login (id + password). Never shown.
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "auth_role" not in st.session_state:
    st.session_state.auth_role = None
if "auth_actor" not in st.session_state:
    st.session_state.auth_actor = None
# Live auto-refresh cadence (seconds) for the critical-alerts region + the last
# report timestamp we've already reflected, so a fresh report triggers an update.
REFRESH_SECONDS = 20
if "_last_seen_report" not in st.session_state:
    st.session_state._last_seen_report = None

with st.sidebar:
    # Wide AARIS banner as the nav-bar brand, then the prominent DEMO MODE
    # toggle right under it so the data-source switch is always one click away.
    if _HEADING_LOGO:
        st.image(_HEADING_LOGO, width="stretch")
    else:
        st.markdown(f"**{APP_NAME}** — {APP_TAGLINE}")
    _demo_on = st.session_state.demo_mode
    if st.button(
        "DEMO MODE: ON — go LIVE" if _demo_on
        else "DEMO MODE: OFF — use demo data",
        type="primary",
        width="stretch",
        help="ON: browse the bundled dashboard/demo_data.json — no ground "
             "station, API or login needed. OFF: LIVE mode — log in to the "
             "ground station over a local port.",
    ):
        st.session_state.demo_mode = not _demo_on
        st.rerun()
    st.caption(
        "Demo data (demo_data.json)" if _demo_on else "Live ground station"
    )
    st.divider()
    st.header("Settings")
    base_url = st.text_input(
        "Ground station URL", value="http://127.0.0.1:8001",
        help="Local base station. For a Jetson on the same network run it with "
             "GROUND_HOST=0.0.0.0 and point the reporter at this machine's IP.",
    )
    # Manual service-token fallback (kept for backward compatibility); normally
    # you log in with id + password instead.
    manual_token = ""
    if not _demo_on:
        if st.session_state.get("auth_token"):
            st.success(
                f"Signed in as **{st.session_state.auth_actor}** "
                f"({st.session_state.auth_role})"
            )
            if st.button("Log out", width="stretch"):
                st.session_state.auth_token = None
                st.session_state.auth_role = None
                st.session_state.auth_actor = None
                st.rerun()
        else:
            st.caption("Live mode needs a sign-in (see the form on the page). "
                       "Demo users: commander / maintainer.")
        with st.expander("Advanced: paste a service token instead"):
            manual_token = st.text_input(
                "Service token (DASHBOARD_SERVICE_TOKEN)",
                value=os.environ.get("DASHBOARD_SERVICE_TOKEN", ""),
                type="password",
                help="Commander-role JWT from POST /auth/login; used instead of "
                     "the login form. Never displayed or logged.",
            )
    risk_filter = st.multiselect("Risk filter", RISK_ORDER, default=RISK_ORDER)
    auto_refresh = st.toggle(
        "Live auto-refresh", value=True,
        help="When on in LIVE mode, critical alerts and the fleet snapshot "
             "refresh automatically and a new aircraft report updates the page "
             "without a manual Refresh.",
    )
    st.divider()
    if st.button("Refresh"):
        st.session_state.refresh_nonce += 1
        st.cache_data.clear()

demo_mode = st.session_state.demo_mode

# ---------------------------------------------------------------------------
# LIVE mode gate: require a bearer token. Login form mints a JWT via
# POST /auth/login; a pasted service token is an advanced fallback. DEMO MODE
# bypasses all of this (no API, no credentials).
# ---------------------------------------------------------------------------
token = None
if not demo_mode:
    token = st.session_state.get("auth_token") or (manual_token or None)
    if not token:
        st.title(f"{APP_NAME} — {APP_TAGLINE}")
        st.subheader("Sign in to the ground station")
        st.caption(
            "Local demo authentication. Credentials are checked by the ground "
            "station's POST /auth/login over a local port (JWT, HS256) — no real "
            "operational data or accounts. Demo users: commander / commander123, "
            "maintainer / maintainer123, admin / admin123."
        )
        with st.form("login_form"):
            login_id = st.text_input("ID")
            login_pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", width="stretch")
        if submitted:
            tok, role, err = login(base_url, login_id, login_pw)
            if err:
                st.error(err)
            else:
                st.session_state.auth_token = tok
                st.session_state.auth_role = role
                st.session_state.auth_actor = login_id
                st.rerun()
        st.stop()

fleet_res = fetch_fleet(base_url, demo_mode, token)
fleet = fleet_res["fleet"]

bases = sorted({a["base"] for a in fleet if a.get("base")}) if fleet else []
with st.sidebar:
    base_filter = st.multiselect("Base filter", bases, default=bases if bases else [])
    st.divider()
    st.caption(
        ("Data source: DEMO (demo_data.json)" if demo_mode
         else f"Data source: {base_url}")
        + "\n\nAll timestamps are UTC."
    )

if fleet_res["error"] and not demo_mode:
    st.error(fleet_res["error"])

# Aircraft with no report yet (risk=None) always stay visible: they matter to
# an operator, and the risk bands cannot classify them.
filtered = [
    a for a in fleet
    if (a.get("risk") in risk_filter or a.get("risk") is None)
    and (a.get("base") in base_filter)
]

schedule_res = fetch_schedule(base_url, demo_mode, token)
plan = None
if schedule_res["payload"] is not None:
    plan = schedule_res["payload"].get("plan") if schedule_res["payload"].get("available") else None
if schedule_res["error"] and schedule_res["error"] != fleet_res["error"]:
    st.error(schedule_res["error"])

# ---------------------------------------------------------------------------
# Critical-alerts monitor. LIVE: an auto-refreshing fragment polls GET /alerts
# on a short cadence so an urgent CRITICAL alert appears without a manual
# refresh, and a brand-new aircraft report triggers a full page rerun so the
# KPIs and the Aircraft Detail profile update live. The station keeps this list
# honest: an aircraft's alert is ACTIVE while its latest reading is CRITICAL and
# moves to CLEARED when a newer reading recovers, so the banner is exactly the
# grounded-aircraft set rather than every aircraft that was ever critical.
# DEMO MODE: no live feed, so the bundled /alerts snapshot from demo_data.json
# is rendered instead (read-only
# station, session-local acknowledgement) — the alert path stays demonstrable
# offline.
# ---------------------------------------------------------------------------
if not demo_mode and token:
    @st.fragment(run_every=REFRESH_SECONDS if auto_refresh else None)
    def _live_monitor():
        alerts, aerr = fetch_alerts(base_url, token)
        if aerr:
            st.error(aerr)
        active = [a for a in (alerts or []) if a.get("status") == "ACTIVE"]
        if active:
            st.error(
                f"{len(active)} ACTIVE CRITICAL ALERT(S) — aircraft below the "
                "grounding threshold. Do not dispatch until cleared."
            )
            alert_rows(active, lambda aid: ack_alert(base_url, token, aid))
        else:
            st.success("No active critical alerts.")

        snap = fetch_fleet(base_url, False, token)["fleet"]
        last = max((x.get("latest_report_at") or "" for x in snap), default="") if snap else ""
        prev = st.session_state._last_seen_report
        st.session_state._last_seen_report = last
        when = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cadence = (f"auto-refresh every {REFRESH_SECONDS}s" if auto_refresh
                   else "auto-refresh off — use Refresh")
        st.caption(
            f"Fleet snapshot read at {when}Z · latest aircraft report "
            f"{last or 'n/a'} · {cadence}"
        )
        # A new report has landed at the ground since the last render.
        if prev is not None and last and last != prev:
            st.rerun()

    _live_monitor()

elif demo_mode:
    bundled, captured_at = demo_alerts()
    if "demo_acked" not in st.session_state:
        st.session_state.demo_acked = set()
    active = [a for a in bundled if a.get("status") == "ACTIVE"
              and a.get("id") not in st.session_state.demo_acked]
    if active:
        st.error(
            f"{len(active)} ACTIVE CRITICAL ALERT(S) — aircraft below the "
            "grounding threshold. Do not dispatch until cleared."
        )

        def _demo_ack(alert_id: int):
            st.session_state.demo_acked.add(alert_id)
            return None

        alert_rows(active, _demo_ack)
    elif bundled:
        st.success("No active critical alerts (bundled snapshot cleared).")
    if bundled:
        st.caption(
            f"DEMO MODE: alerts are the bundled snapshot captured {captured_at} "
            "from the seeded station — read-only, and acknowledging clears them "
            "for this browser session only. Sign in with LIVE mode for the real "
            "feed from an edge device."
        )

# ---------------------------------------------------------------------------
# Section navigation
# ---------------------------------------------------------------------------
# Only the active section renders, so a click on an aircraft elsewhere on the
# page can both pick that aircraft and move to its profile in one step — the
# reason this replaces st.tabs, which cannot be selected from code.
if "nav_section" not in st.session_state:
    st.session_state.nav_section = NAV_FLEET
nav = st.segmented_control(
    "Dashboard section",
    SECTIONS,
    key="nav_section",
    selection_mode="single",
    required=True,
    label_visibility="collapsed",
    width="stretch",
)

# --- 1) Fleet Readiness Decision Support -------------------------------------
if nav == NAV_FLEET:
    st.header("Fleet Readiness Decision Support")

    # KPI computed from live /fleet + /schedule/latest state on every render.
    ready = compute_readiness(fleet, plan)
    avail_txt = (
        f"{100 * ready['availability']:.1f}%" if ready["availability"] is not None else "no data"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Fleet Availability", avail_txt,
        help=("Share of roster aircraft that are NOT in CRITICAL risk and are NOT "
              "scheduled for maintenance today (UTC). Computed live from GET /fleet "
              "and GET /schedule/latest."),
    )
    k2.metric("Available now", f"{ready['available']} / {ready['total']}")
    k3.metric("At risk (CRITICAL + HIGH)", ready["at_risk"])
    k4.metric("In maintenance today", len(ready["scheduled_today"]))

    breakdown = (
        f"Roster {ready['total']} | not CRITICAL {ready['total'] - ready['critical']} | "
        f"scheduled today {len(ready['scheduled_today'])} | "
        f"unreported (counted as not-CRITICAL) {ready['unreported']}"
    )
    if ready["plan_today_index"] is None:
        breakdown += " | no plan for today (plan absent, stale, or outside horizon)"
    st.caption(breakdown + ". KPI uses the full roster; sidebar filters affect the views below.")

    # Q1 — availability of the (filtered) list, one explicit status per aircraft.
    st.subheader("Which aircraft are available now?")
    if filtered:
        def readiness_status(a):
            if a.get("risk") == "CRITICAL":
                return "GROUNDED (CRITICAL)"
            if a["aircraft_id"] in ready["scheduled_today"]:
                return "IN MAINTENANCE TODAY"
            return "AVAILABLE"

        table = pd.DataFrame(
            [
                {
                    "aircraft_id": a["aircraft_id"],
                    "availability": readiness_status(a),
                    "base": a["base"],
                    "mission_priority": a["mission_priority"],
                    "spare_available": a["spare_available"],
                    "rul_cycles": a["rul_cycles"],
                    "engine_health": a["engine_health"],
                    "risk": a["risk"],
                    "latest_report_at": a["latest_report_at"],
                }
                for a in filtered
            ]
        )
        selectable(table, "fleet_table",
                   "Fleet readiness table — click a row to open that aircraft profile")
        st.caption("Click a row to open that aircraft's profile.")
    else:
        st.info("No aircraft match the current filters (or no data is available).")

    # Q2 — risk picture: band counts + RUL/health scatter.
    st.subheader("Which aircraft are at risk?")
    reported = [a for a in filtered if a.get("rul_cycles") is not None]
    risk_counts = {r: sum(1 for a in filtered if a.get("risk") == r) for r in RISK_ORDER}
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("CRITICAL", risk_counts["CRITICAL"])
    r2.metric("HIGH", risk_counts["HIGH"])
    r3.metric("MEDIUM", risk_counts["MEDIUM"])
    r4.metric("LOW", risk_counts["LOW"])
    if reported:
        scatter_df = pd.DataFrame(reported)[["rul_cycles", "engine_health", "risk", "aircraft_id"]]
        fig = px.scatter(
            scatter_df, x="rul_cycles", y="engine_health", color="risk",
            hover_data=["aircraft_id"], color_discrete_map=RISK_COLORS,
            labels={"rul_cycles": "RUL (cycles)", "engine_health": "Engine health (%)"},
            category_orders={"risk": RISK_ORDER},
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No reported sensor windows yet — nothing to plot.")

    # Q3 — next 7 days of the plan, with calendar dates and deadlines.
    st.subheader("Which aircraft must enter maintenance, and by when?")
    if ready["upcoming_7d"]:
        base_day = datetime.strptime(plan["generated_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        up_df = pd.DataFrame(
            [
                {
                    "aircraft_id": t["aircraft_id"],
                    "scheduled_day": t["scheduled_day"],
                    "date": str(base_day + timedelta(days=t["scheduled_day"] - 1)),
                    "deadline_day": t.get("maintenance_deadline_day"),
                    "rul_cycles": t.get("rul_cycles"),
                    "risk": t.get("risk"),
                }
                for t in ready["upcoming_7d"]
            ]
        )
        selectable(up_df, "upcoming_table",
                   "Next 7 days of maintenance — click a row to open that aircraft profile")
    elif plan:
        st.info("No maintenance scheduled within the next 7 days of the plan horizon.")
    else:
        st.info("No schedule available yet — see the Maintenance Plan section.")

    # Q4 — daily load across the horizon.
    st.subheader("What is the daily maintenance load?")
    if plan and plan.get("daily_load"):
        horizon = plan.get("horizon_days", 30)
        load_df = pd.DataFrame(
            {"day": list(range(1, horizon + 1))},
        )
        load_df["tasks"] = load_df["day"].map(
            lambda d: plan["daily_load"].get(str(d), 0)
        )
        fig = px.bar(load_df, x="day", y="tasks",
                     labels={"day": "Day (1-30)", "tasks": "Tasks scheduled"})
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No schedule available yet — see the Maintenance Plan section.")

    # Q5 — deferrals with reasons (full detail also in the Maintenance Plan section).
    st.subheader("What actions are deferred, and why?")
    if ready["deferred"]:
        def_df = pd.DataFrame(
            [
                {"aircraft_id": t["aircraft_id"], "risk": t.get("risk"),
                 "reason": t.get("reason")}
                for t in ready["deferred"]
            ]
        )
        selectable(def_df, "deferred_table",
                   "Deferred maintenance actions — click a row to open that aircraft profile")
    elif plan:
        st.info("Nothing deferred in the current plan.")
    else:
        st.info("No schedule available yet — see the Maintenance Plan section.")

# --- 2) Aircraft Models ---------------------------------------------------
# One profile per aircraft MODEL (the demo fleet is 6 models x 15 airframes),
# aggregated by the ground station from the synthetic profiles + live state +
# the last persisted plan. Airframe rows stay one click from that aircraft's
# own profile.
if nav == NAV_MODELS:
    st.header("Aircraft Models")
    models_res = fetch_models(base_url, demo_mode, token)
    if models_res["error"]:
        st.error(models_res["error"])
    models = models_res["models"]

    if not models:
        st.info("No aircraft-model profiles available.")
    else:
        st.info(
            "Synthetic demonstration fleet profiles. No real IAF operational or "
            "maintenance data is used; the photos are illustrative type images, "
            "never a specific airframe."
        )
        total_airframes = sum(m["aircraft_count"] for m in models)
        st.caption(
            f"{len(models)} aircraft models · {total_airframes} airframes · "
            "model health is aggregated from GET /fleet state and "
            + (f"the plan generated {models_res['plan_generated_at']} UTC."
               if models_res["plan_generated_at"] else "no plan yet (run POST /schedule).")
        )
        # Sidebar filters apply to the airframe tables, exactly as they do to the
        # fleet views, so a filtered view stays internally consistent.
        visible_ids = {a["aircraft_id"] for a in filtered}

        for m in models:
            st.divider()
            st.subheader(f"{m['type']} · {m['model_code']}")
            ids = m.get("aircraft_ids") or []
            span = f"{ids[0]}–{ids[-1]}" if ids else "n/a"
            img = _type_image({"image": m.get("image")})
            if img:
                ic1, ic2 = st.columns([1, 1.35])
                with ic1:
                    st.image(img, width="stretch")
                    st.caption(f"{m['type']} — reference photo (illustrative type "
                               "image, not any specific airframe).")
                body = ic2
            else:
                body = st
            with body:
                st.write(
                    f"**{m.get('role', 'n/a')}**  \n"
                    f"Origin: {m.get('origin', 'n/a')}   |   "
                    f"Engines: {m.get('engines', 'n/a')}   |   "
                    f"Airframes: {m['aircraft_count']} ({span})"
                )
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Airframes", m["aircraft_count"],
                          help=f"{m['reporting_count']} of {m['aircraft_count']} have "
                               "reported to the ground station.")
                m2.metric("Ready to fly", m["ready_count"],
                          help="Airframes of this model not in CRITICAL risk "
                               "(point estimate, same rule as the Fleet "
                               "Availability KPI numerator's first test).")
                m3.metric("Grounded (CRITICAL)", m["critical"],
                          help="RUL below the 30-cycle grounding threshold.")
                m4.metric("Avg RUL (cycles)",
                          m["avg_rul_cycles"] if m["avg_rul_cycles"] is not None else "no data",
                          help="Mean over the airframes of this model that have reported.")
                m5.metric("In 30-day plan",
                          f"{m['scheduled_in_plan']} / {m['deferred_in_plan']}",
                          help="Scheduled vs deferred airframes of this model in the "
                               "latest CP-SAT plan.")
                mix = m.get("risk_mix") or {}
                st.caption(
                    "Risk mix: "
                    + " · ".join(f"{band} {mix.get(band, 0)}" for band in RISK_ORDER)
                    + f"  |  no telemetry {m['unreported']}"
                    + f"  |  lowest RUL {m['lowest_rul_cycles'] if m['lowest_rul_cycles'] is not None else 'n/a'}"
                    + f"  |  no spare on {m['spares_missing']}"
                    + f"  |  bases: {', '.join(m['bases']) if m['bases'] else 'n/a'}"
                )
                if m.get("issue_systems"):
                    st.caption(
                        "Issue history (synthetic): "
                        + ", ".join(f"{d['system']} ({d['count']})"
                                    for d in m["issue_systems"])
                        + f" — {m['open_issues']} OPEN across the block."
                    )

            rows = [r for r in m.get("aircraft", [])
                    if r["aircraft_id"] in visible_ids]
            if rows:
                ac_df = pd.DataFrame(
                    rows,
                    columns=["aircraft_id", "base", "mission_priority",
                             "spare_available", "rul_cycles", "engine_health",
                             "risk", "plan_status", "plan_day", "latest_report_at"],
                )
                selectable(ac_df, f"model_table_{m['model_code']}",
                           f"{m['type']} airframes — click a row to open that "
                           "aircraft profile")
                st.caption("Click a row to open that aircraft's profile.")
            elif visible_ids:
                st.caption("No airframe of this model matches the current filters.")

# --- 3) Aircraft Detail -----------------------------------------------------
if nav == NAV_DETAIL:
    st.header("Aircraft Detail")
    all_ids = [a["aircraft_id"] for a in fleet] if fleet else []
    if not all_ids:
        st.info("No aircraft loaded — start the ground station or use DEMO MODE.")
    else:
        # detail_id is the durable "which aircraft am I looking at": clicks write
        # it, the picker writes it back on change, and it is re-applied here
        # because this section's widgets unmount when another section is shown.
        # A click from any table/alert row therefore lands on that aircraft.
        if st.session_state.get("detail_id") not in all_ids:
            st.session_state.detail_id = ("AC-017" if "AC-017" in all_ids
                                          else all_ids[0])
        st.session_state.detail_aircraft = st.session_state.detail_id
        selected = st.selectbox("Aircraft", all_ids, key="detail_aircraft",
                                on_change=sync_detail_id)
        det_res = fetch_aircraft(base_url, demo_mode, selected, token)
        if det_res["error"]:
            st.error(det_res["error"])
        elif det_res.get("missing_in_demo"):
            st.info(f"No demo data captured for {selected}.")
        else:
            detail = det_res["detail"]
            state = detail["aircraft"]
            history = detail.get("history", [])
            profile = detail.get("profile") or {}

            st.info(
                "Synthetic demonstration fleet profile. No real IAF "
                "operational or maintenance data is used."
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RUL (cycles)", state["rul_cycles"] if state["rul_cycles"] is not None else "no data")
            c2.metric("Engine health (%)", state["engine_health"] if state["engine_health"] is not None else "no data")
            c3.metric("Risk", state["risk"] if state["risk"] else "no data")
            c4.metric("Reports", detail.get("report_count", 0))

            st.text(
                f"Base: {state['base']}   |   Mission priority: {state['mission_priority']}   |   "
                f"Spare available: {state['spare_available']}"
            )

            # --- Synthetic type profile + maintenance metadata (GET /fleet/{id}
            #     "profile" key; absent -> compact fallback line, no crash). ---
            if profile:
                st.subheader("Aircraft profile (synthetic)")
                img = _type_image(profile)
                if img:
                    ic1, ic2 = st.columns([1, 1.35])
                    with ic1:
                        st.image(img, width="stretch")
                        st.caption(
                            f"{profile.get('type', 'Type')} — reference photo "
                            "(illustrative type image, not this specific airframe)."
                        )
                    body = ic2
                else:
                    body = st
                with body:
                    st.write(
                        f"**{profile.get('type', 'n/a')}** ({profile.get('model_code', 'n/a')}) — "
                        f"{profile.get('role', 'n/a')}  \n"
                        f"Origin: {profile.get('origin', 'n/a')}   |   "
                        f"Engines: {profile.get('engines', 'n/a')}   |   "
                        f"Status: {profile.get('operational_status', 'n/a')}"
                    )
                    lm = profile.get("last_maintenance") or {}
                    if lm:
                        st.write(
                            f"Last maintenance: **{lm.get('date', 'n/a')}** — {lm.get('type', 'n/a')} "
                            f"at {lm.get('location', 'n/a')}  \n"
                            f"Cycles since maintenance: {lm.get('cycles_since_maintenance', 'n/a')}   |   "
                            f"Next due: **{lm.get('next_due', 'n/a')}**"
                        )
            else:
                st.caption("No type profile attached to this aircraft "
                           "(ground station started without aircraft_profiles.json).")

            issues = profile.get("issue_history") or []
            st.subheader("Issue history (synthetic)")
            if issues:
                st.dataframe(pd.json_normalize(issues), hide_index=True,
                             width='stretch')
            else:
                st.caption("No historical issues logged for this airframe.")

            if state["rul_cycles"] is None:
                st.info(
                    "This aircraft has no health reports at the ground station yet. "
                    "An edge node report (POST /ingest_report) will populate it."
                )
            else:
                st.subheader("Top issues")
                if state["top_issues"]:
                    st.dataframe(pd.json_normalize(state["top_issues"]),
                                 hide_index=True, width='content')
                else:
                    st.caption("No issues flagged.")

                st.subheader("Recommended action")
                st.write(state["recommended_action"] or "None recorded.")

                conf = load_conformal()
                if conf:
                    hw = float(conf["halfwidths_cycles"]["90"])
                    lo = max(0.0, float(state["rul_cycles"]) - hw)
                    hi = min(CONFORMAL_MAX_RUL, float(state["rul_cycles"]) + hw)
                    st.subheader("Uncertainty (conformal 90% interval)")
                    st.write(
                        f"RUL interval: **[{lo:.1f}, {hi:.1f}] cycles** "
                        f"(point estimate ± {hw:.2f} cycles). On the FD001 "
                        f"cross-fit data this interval contained the true RUL for "
                        f"{float(conf['empirical_coverage_oof']['90']):.0%} of "
                        f"held-out engine windows."
                    )
                    st.write(_grounding_phrase(lo, hi))
                    bd = (conf.get("band_diagnostics_90") or {}).get(state.get("risk") or "")
                    if bd and bd.get("empirical_coverage_90_clipped") is not None:
                        st.caption(
                            f"Measured 90% interval coverage within the "
                            f"{state['risk']} band specifically: "
                            f"{bd['empirical_coverage_90_clipped']:.0%} "
                            f"(global half-widths are conservative where this is "
                            f"high, thin where it is low)."
                        )
                    st.caption(
                        "Cross-conformal calibration on the frozen Ridge recipe "
                        "(engine-wise 5-fold, per-fold scaler refit); heuristic "
                        "coverage, FD001 distribution only. Point-estimate risk "
                        "bands are unchanged. Source: conformal/risk_calibration.json."
                    )

            st.subheader("Report history")
            if len(history) >= 2:
                # Oldest -> newest so the degradation trend reads left to right.
                trend = pd.DataFrame(
                    [{"timestamp": h["timestamp"], "rul_cycles": h["rul_cycles"],
                      "risk": h["risk"]} for h in reversed(history)]
                )
                trend["timestamp"] = pd.to_datetime(trend["timestamp"])
                fig = px.line(trend, x="timestamp", y="rul_cycles", markers=True,
                              title="RUL trend (predicted cycles remaining)")
                st.plotly_chart(fig, width="stretch")
            if history:
                hist_df = pd.DataFrame(
                    [
                        {
                            "timestamp": h["timestamp"],
                            "rul_cycles": h["rul_cycles"],
                            "engine_health": h["engine_health"],
                            "risk": h["risk"],
                            "recommended_action": h["recommended_action"],
                            "received_at": h["received_at"],
                        }
                        for h in history
                    ]
                )
                st.dataframe(hist_df, hide_index=True, width='stretch')
            else:
                st.info("No reports stored for this aircraft.")

            st.subheader("Maintenance plan status")
            task = next((t for t in ((plan or {}).get("tasks") or [])
                         if t.get("aircraft_id") == selected), None)
            if task is None:
                st.info("No maintenance plan available yet — see the Maintenance Plan section.")
            elif task["status"] == "SCHEDULED":
                st.write(
                    f"**SCHEDULED** for plan day {task['scheduled_day']} "
                    f"(deadline day {task.get('maintenance_deadline_day')}). "
                    f"{task.get('reason', '')}"
                )
            else:
                st.write(
                    f"**DEFERRED** (deadline day {task.get('maintenance_deadline_day')}). "
                    f"{task.get('reason', '')}"
                )

# --- 4) Maintenance Plan ----------------------------------------------------
if nav == NAV_PLAN:
    st.header("Maintenance Plan")
    if not plan:
        st.info(
            "No schedule has been generated yet. POST /schedule on the ground "
            "station to solve the CP-SAT plan, then reload this page."
        )
    else:
        st.write(
            f"Generated at **{plan['generated_at']}** | solver status: "
            f"**{plan['solver_status']}** | horizon: {plan['horizon_days']} days | "
            f"max {plan['max_daily_maintenance']} maintenance tasks/day."
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Scheduled", plan["scheduled_count"])
        p2.metric("Deferred", plan["deferred_count"])
        p3.metric("Peak daily load", max(plan["daily_load"].values()) if plan["daily_load"] else 0)

        sched = [t for t in plan["tasks"] if t["status"] == "SCHEDULED"]
        if sched:
            base_day = datetime.strptime(plan["generated_at"], "%Y-%m-%dT%H:%M:%SZ").date()
            gantt = pd.DataFrame(
                {
                    "aircraft_id": [t["aircraft_id"] for t in sched],
                    "day": [t["scheduled_day"] for t in sched],
                    "risk": [t["risk"] for t in sched],
                    "deadline_day": [t["maintenance_deadline_day"] for t in sched],
                    "start": [datetime.combine(base_day + timedelta(days=t["scheduled_day"] - 1),
                                               datetime.min.time()) for t in sched],
                    "end": [datetime.combine(base_day + timedelta(days=t["scheduled_day"]),
                                             datetime.min.time()) for t in sched],
                }
            ).sort_values(["day", "aircraft_id"])

            fig = px.timeline(
                gantt, x_start="start", x_end="end", y="aircraft_id",
                color="risk", color_discrete_map=RISK_COLORS,
                hover_data=["day", "deadline_day"],
                category_orders={"aircraft_id": gantt["aircraft_id"].tolist()},
                labels={"start": "Day"},
            )
            fig.update_yaxes(autorange="reversed")  # day 1 at the top
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No SCHEDULED tasks in the current plan.")

        st.subheader("All tasks")
        tasks_df = pd.DataFrame(
            [
                {
                    "aircraft_id": t["aircraft_id"],
                    "status": t["status"],
                    "scheduled_day": t.get("scheduled_day"),
                    "deadline_day": t.get("maintenance_deadline_day"),
                    "rul_cycles": t.get("rul_cycles"),
                    "risk": t.get("risk"),
                    "mission_priority": t.get("mission_priority"),
                    "spare_available": t.get("spare_available"),
                    "reason": t.get("reason"),
                }
                for t in plan["tasks"]
            ]
        )
        selectable(tasks_df, "plan_tasks_table",
                   "All CP-SAT plan tasks — click a row to open that aircraft profile")

# --- 5) About ---------------------------------------------------------------
if nav == NAV_ABOUT:
    st.header("About")
    st.markdown(
        """
        **How this system works.**

        1. An **edge node** on each aircraft (a Jetson-class AHMS field processor —
           see `docs/JETSON_FIELD_NODE.md` for the deployment path) watches a
           30-cycle window of engine sensor data and runs the frozen Ridge model
           to predict Remaining Useful Life (RUL). It posts a small JSON health
           report to the ground.
        2. The **ground station** stores every report in SQLite, keeps the
           latest state per aircraft, and answers the read-only endpoints this
           dashboard uses: `GET /fleet`, `GET /fleet/{aircraft_id}`,
           `GET /models`, `GET /schedule/latest`.
        3. A **CP-SAT optimizer** (Google OR-Tools) turns current fleet health
           into a 30-day maintenance plan: at most one task per aircraft, at
           most eight hangar slots per day (`MAX_DAILY_MAINTENANCE`, overridable
           per request), each repair on or before its health-derived deadline,
           spares lead-time respected when still feasible. Every aircraft is
           either SCHEDULED or DEFERRED with a written reason.

        **How the fleet is built.** The demonstration roster is 90 fictional
        airframes grouped as 6 aircraft models x 15 airframes each (Rafale EH,
        Su-30MKI, Tejas Mk1A, MiG-29UPG, Mirage 2000I/TI, Jaguar IS/IM —
        `ground/make_profiles.py`). Every model carries 15 profiles of the same
        health shape: 3 CRITICAL, 3 HIGH, 3 MEDIUM, 4 LOW and 2 airframes with
        no telemetry yet. All identities, maintenance dates and issue logs are
        synthetic; the per-type photos are illustrative images of the type, not
        of any specific airframe, and no real IAF operational or maintenance
        data is used anywhere.

        **Where the numbers come from.** `rul_cycles` and `engine_health` are
        model outputs from the frozen Ridge RUL model (validated on the NASA
        C-MAPSS FD001 dataset variant). `top_issues` and `recommended_action`
        are operational heuristics, not model outputs — they highlight the
        sensors with the strongest negative contribution to the prediction and
        map the risk band to a standing playbook.
        """
    )
    st.caption("All timestamps are UTC (ISO-8601, e.g. 2026-10-11T13:07:00Z).")
