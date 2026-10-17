# AARIS — Fleet Dashboard

Plain Streamlit UI over the ground station API. No custom CSS, no themes —
standard components, default Plotly styling, muted colors only for risk
status. Read-only: it never writes to the ground station, edge node, or the
model artifacts.

## Files

```
dashboard/
  app.py           the dashboard (single file)
  demo_data.json   offline fallback captured from the live ground station
                   (regenerate: ../.venv/Scripts/python.exe demo/capture_demo_data.py)
  app_tests.py     Streamlit AppTest suite for the auth/DEMO behaviour (see Tests)
  assets/
    aaris_heading.png  wide banner logo at the top of the sidebar
    aaris_logo.png     square emblem (browser-tab icon)
    plane_types/       per-type reference photos (su30mki, rafale, tejas,
                       mig29upg, mirage2000, jaguar .jpg) shown in Aircraft
                       Detail and Aircraft Models
  README.md        this file
```

Streamlit's built-in "Deploy" toolbar button (and its hosting dialog) is
disabled repo-wide via `.streamlit/config.toml` → `client.toolbarMode =
"minimal"`; it applies because the dashboard is always started from the repo
root (`start.bat`, `demo/run_demo.ps1`, and the manual command above).

## Run

**One command (Windows): double-click `start.bat` in the repo root** — it
launches the edge node (8000), ground station (8001) and this dashboard (8501)
in minimized windows and opens http://localhost:8501. `stop.bat` shuts them
down.

Adding **`--live-feed`** (`start.bat --live-feed`) also starts
`../edge/live_reporter.py` for AC-017 in its own minimized window: it streams
SYNTHETIC health reports to the local ground station on a compressed rehearsal
cadence (20 s routine tick, 5 s while degraded/critical), so the live profile
updates and a CRITICAL alert appears on the dashboard banner within a couple of
minutes. Flags combine with `--no-browser` (order-free). `stop.bat` kills the
reporter window *and* its python child — the reporter owns no port, so a
port-based kill alone would leave it pushing reports.

**`--fleet-feed`** (`start.bat --fleet-feed`) runs the same reporter with
`--fleet`: one synthetic feed per roster aircraft (AC-001…AC-090), each using
its own device token and staggered in time and along the degradation timeline,
so **every** Aircraft Detail updates live and critical alerts arrive one at a
time instead of as a single burst. The banner then lists exactly the aircraft
that are CRITICAL right now — the ground marks an alert `CLEARED` when that
aircraft's next newer reading recovers. Note that this replaces the canonical
AC-017 demo reading (112.76 / LOW) and moves the plan/KPI numbers until you
reseed with `demo\run_demo.ps1 -Reset`.

Manually, start the ground station first (port 8001):

```bash
cd ../ground
../.venv/Scripts/python.exe -m uvicorn ground_station:app --port 8001
```

Then, from the repo root:

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Dependencies: `streamlit`, `plotly`,
`requests` (see requirements below).

## What it shows

- **Sidebar (left navigation):** AARIS banner logo (`assets/aaris_heading.png`)
  at the top, directly above a prominent primary **DEMO MODE** button (label
  shows the current mode: "DEMO MODE: OFF — use demo data" /
  "DEMO MODE: ON — go LIVE"). Below it: ground station URL; in LIVE mode a
  **sign-in** state (signed-in user + Log out, or a prompt to use the on-page
  login form) with an **Advanced** expander that still accepts a pasted
  `DASHBOARD_SERVICE_TOKEN` JWT; risk filter; a **Live auto-refresh** toggle;
  base filter; Refresh button.
- **Live critical-alerts banner (above the section nav, so visible in every
  section):** an
  auto-refreshing region that reads `GET /alerts` on a short cadence in LIVE
  mode. Active
  CRITICAL alerts appear at the top (newest first) with an **Acknowledge**
  button and a **Profile** button each (the `POST /alerts/{id}/ack` call stays
  behind Acknowledge), and a caption shows when the snapshot
  was read and the latest aircraft report time. When a new aircraft report
  lands, the whole page refreshes so the KPIs and Aircraft Detail profile track
  it in near-real-time. Toggling **Live auto-refresh** off stops the polling
  (use Refresh to update manually). In **DEMO MODE** there is no live feed, so
  the banner renders from the bundled alert snapshot in `demo_data.json`
  instead and Acknowledge is session-local.
- **Section nav (`st.segmented_control` under the banner):** *Fleet Readiness ·
  Aircraft Models · Aircraft Detail · Maintenance Plan · About*. It is a session-state value, not
  `st.tabs`, precisely so the app can switch sections from code: **one click on
  an aircraft opens its profile.** Clicking any row in the readiness, next-7-days,
  deferred, plan-task or per-model airframe tables — or **Profile** on an alert —
  selects that aircraft in the Aircraft Detail picker and shows it. Each such
  table is captioned "Click a row to open that aircraft's profile", and the
  latest click wins over a manual picker selection. Verified headlessly by
  checks **T10** (alert Profile click) and **T11** (Aircraft Models render) in
  `app_tests.py`; table-row clicks go through the same `open_profile` path but
  need a real browser, since AppTest cannot select an `st.dataframe` row.
- **Fleet Readiness** (headline section, framed as decision support — it answers
  the five operator questions):
  - **Fleet Availability KPI** = aircraft NOT in CRITICAL risk AND NOT
    scheduled for maintenance today (UTC), divided by the total roster.
    Computed from live `GET /fleet` + `GET /schedule/latest` on every render —
    never hardcoded. "Today" is mapped through the plan's `generated_at` date;
    if no plan covers today, the caption says so. Unreported aircraft count as
    not-CRITICAL, and their number is surfaced in the caption.
  - *Which aircraft are available now?* — table with an explicit
    AVAILABLE / GROUNDED (CRITICAL) / IN MAINTENANCE TODAY status per aircraft.
  - *Which aircraft are at risk?* — risk-band counts + RUL-vs-health scatter.
  - *Which aircraft must enter maintenance, and by when?* — next-7-days table
    with calendar dates and deadlines.
  - *What is the daily maintenance load?* — daily-load bar chart from the plan.
  - *What actions are deferred, and why?* — deferred table with solver reasons.
- **Aircraft Models:** one profile card per aircraft model — the roster's 6 models
  × 15 airframes (Rafale EH, Su-30MKI, Tejas Mk1A, MiG-29UPG, Mirage 2000I/TI,
  Jaguar IS/IM) — each card showing the illustrative type photo, role/origin/engines,
  airframe count, ready/grounded split, average RUL, in-plan scheduled/deferred
  split, risk mix and the synthetic issue history, above a 15-row airframe table.
  Clicking any aircraft row here (or anywhere else) opens that aircraft's profile.
  Aggregates come from `GET /models`; card content is the synthetic demo profile data.
- **Aircraft Detail:** picker per aircraft (reachable in one click from any
  table row or alert); latest RUL, health, risk, top
  issues, recommended action, **conformal 90% uncertainty interval** (from
  `conformal/risk_calibration.json`, ±27.19 cycles — with a plain-language
  statement whether the interval crosses the 30-cycle grounding threshold and
  the measured coverage for that aircraft's risk band), and report history
  table. Aircraft with no reports are stated plainly, not hidden. If the
  calibration sidecar is absent, the uncertainty section is simply omitted.
  When the ground station returns a `"profile"` (see `GET /fleet/{aircraft_id}`),
  the profile also shows a **per-type reference photo** (resolved from
  `profile["image"]` under `assets/plane_types/`, captioned as an illustrative
  type image — not the specific airframe) beside the **synthetic aircraft profile**
  (type/model_code/role,
  origin, engines, operational status, last-maintenance record + next due), an
  **issue-history table**, a **RUL trend line chart** (from the report history,
  shown once there are ≥2 points), and this aircraft's **maintenance-plan status**
  (SCHEDULED day + reason, or DEFERRED reason). A plain Streamlit info banner reads
  **"Synthetic demonstration fleet profile. No real IAF operational or maintenance
  data is used."** Aircraft with no profile just omit these sections — nothing else
  changes.
- **Maintenance Plan:** 30-day Gantt-style timeline of SCHEDULED tasks
  (day-ordered), the full tasks table including DEFERRED aircraft with their
  reasons, plus `scheduled_count`, `deferred_count`, `solver_status`, and
  daily load.
- **About:** plain-English description of the pipeline (including the
  Jetson-class field-node deployment path, `docs/JETSON_FIELD_NODE.md`), and
  what is model output (RUL and health — frozen Ridge model) versus operational
  heuristics (top_issues, recommended_action).

## Data sources

Reads these endpoints, with a 5-second timeout each:

- `GET /fleet`
- `GET /models` (Aircraft Models section aggregates)
- `GET /fleet/{aircraft_id}`
- `GET /schedule/latest`
- `GET /alerts` (+ `POST /alerts/{id}/ack` when you Acknowledge) — LIVE mode only

If the ground station is unreachable, the page shows the exact command to
start it. **DEMO MODE** (sidebar button under the logo) serves the same read responses
from `demo_data.json` — captured 2026-10-17 from the live station with the
seeded demo fleet (90 aircraft, 390 reports — 5 for each of the 78 reporting
airframes incl. synthetic trend history, 18 ACTIVE critical alerts, CP-SAT
plan: 54 scheduled / 36 deferred), so the dashboard works
fully offline.

> **Data provenance.** The RUL and engine-health numbers come from the **NASA
> C-MAPSS FD001** Ridge model running on the edge node. The aircraft
> identities, types, bases, maintenance records, issue logs and report trends
> shown in Aircraft Detail are **synthetic demonstration profiles**
> (`../ground/aircraft_profiles.json`) — no real Indian Air Force operational,
> maintenance, tail-number, sortie, defect or classified data is used, scraped,
> inferred, or implied. The per-type photos in `assets/plane_types/` are generic
> aircraft-type reference images — they depict the type, not the fictional
> airframe, and carry no association to any real tail number or record. The
> dashboard labels this with a banner on the Aircraft Detail section.

> Security note (2026-10-11): those read endpoints (plus `GET /alerts`) are
> access-controlled (see `../ground/README.md` → Security), so **LIVE mode signs
> in with an ID + password** — the dashboard calls the ground station's
> `POST /auth/login` and keeps the returned commander JWT only in the browser
> session. Without signing in it shows a login form instead of a wall of 401s,
> and **DEMO MODE remains the zero-credential path**: it serves the same
> responses from `demo_data.json` and needs no API or login at all.

All timestamps are UTC ISO-8601, exactly as stored by the ground station.

## LIVE mode — sign in with ID + password

The ground station protects the read + alert endpoints with RBAC. The dashboard
runs as a **read-only `commander` viewer**. In LIVE mode (DEMO MODE off) the page
opens on a **sign-in form**:

1. Enter a demo user and click **Log in** — the dashboard POSTs
   `/auth/login` (form-encoded) and stores the returned JWT in the session:
   - `commander` / `commander123` (viewer — the normal choice), or
   - `maintainer` / `maintainer123`, or `admin` / `admin123`.
2. Every subsequent request (`GET /fleet`, `GET /fleet/{id}`, `GET /models`,
   `GET /schedule/latest`, `GET /alerts`, and the Acknowledge `POST`) sends
   `Authorization: Bearer <session JWT>`. Timeouts stay at 5 s.
3. **Log out** in the sidebar clears the session token.

- **Token hygiene:** the JWT is never displayed or logged; the login field is
  masked and error messages quote only HTTP statuses.
- **Real-time:** with **Live auto-refresh** on (default), the critical-alerts
  banner polls `GET /alerts` every ~20 s, and a newly-arrived aircraft report
  triggers a page refresh so KPIs + Aircraft Detail track it live. A CRITICAL
  report (e.g. from `edge/live_reporter.py`) surfaces in the banner with an
  **Acknowledge** button. Turn the toggle off to update via **Refresh** instead.
- **Errors are explained per status:** 401 → credentials rejected (re-sign in;
  the JWT expires after 60 minutes); 403 → the signed-in role lacks permission.

### Advanced: pasted service token (fallback)

The sidebar **Advanced → "paste a service token instead"** expander still accepts
a commander JWT directly (prefilled from `DASHBOARD_SERVICE_TOKEN`), used when
you have not signed in via the form. Mint one with:

```bash
curl -X POST http://127.0.0.1:8001/auth/login -d "username=commander&password=commander123"
```

…then paste the `access_token`, or export `DASHBOARD_SERVICE_TOKEN` before
launching Streamlit. Demo credentials come from `ground/security.py`; change
`AEROGUARD_SECRET_KEY` and the seeded passwords before any real deployment.

### DEMO MODE always remains available

Unchecking nothing is required: ticking **DEMO MODE** bypasses every API call
and serves the bundled `demo_data.json` snapshot — same fleet (90 aircraft),
same AC-017 values (112.76 / 90.21 / LOW), same 54/36 plan, **plus the bundled
critical-alerts snapshot**, so the dashboard works fully offline or with no
credentials at all. In DEMO MODE the alerts banner renders from that snapshot
and **Acknowledge** clears an alert for the current browser session only (the
bundled file is never written); the caption says so. LIVE mode is the real
thing — it polls `GET /alerts` and posts `POST /alerts/{id}/ack`.

## Tests

`app_tests.py` (Streamlit `AppTest`, no browser needed; 10 checks, all green)
covers: the LIVE
sign-in form with zero network calls, the `Authorization: Bearer` header on every
LIVE GET (`/fleet`, `/schedule/latest`, `/models`, `/fleet/{id}` and the
`/alerts` poll — token never rendered in the UI),
readable 401 and 403 messages, env-var prefill, DEMO MODE with the network
forbidden (incl. the bundled alert banner and the date-derived Fleet
Availability, re-derived in-test by `expected_readiness()`), a click on the
sidebar DEMO MODE button (OFF → ON, demo renders, the nav is the five-section
segmented control, zero network), a DEMO MODE **Acknowledge**
click (18 alerts → 17, zero network), a DEMO MODE **Profile** click (jumps from
the alert to that aircraft's profile, picker and nav both follow, zero network),
a DEMO MODE **Aircraft Models** render (all 6 bundled model profiles, 15
airframes each — 3 CRITICAL / 12 ready / 9 scheduled per block — zero network),
and an optional LIVE run against a real
ground station on 8001 when `DASHBOARD_SERVICE_TOKEN` is set:

```bash
.venv/Scripts/python.exe dashboard/app_tests.py
```

## Expected values (with the ground station on 8001)

- AC-017: `rul_cycles = 112.76`, `engine_health = 90.21`, `risk = LOW`
- Maintenance Plan: 54 scheduled, 36 deferred, solver status OPTIMAL,
  no day above 8 tasks
- Fleet Availability on the seeded demo fleet (with a freshly reset demo, plan
  generated that day):
  **80.0% — 72 / 90** (18 CRITICAL grounded; the 8 airframes booked into
  maintenance on plan day 1 are all already CRITICAL; 12 unreported counted as
  not-CRITICAL). The KPI is date-dependent — plan days are offsets from the
  plan's `generated_at` — e.g. six days later it reads **71.1% — 64 / 90**
  because 8 non-CRITICAL airframes are then in the hangar.

## Requirements

```bash
../.venv/Scripts/python.exe -m pip install streamlit plotly requests
```

(Python 3.11+; already installed in the repo `.venv`.)

## Limitations (deliberate)

- Read-only: scheduling is done by `POST /schedule` on the ground station,
  not from this UI.
- Auto-refresh (LIVE mode) polls only `GET /alerts`; the rest of the page
  updates on the next interaction or via the **Refresh** button.
- Risk/base filters apply to the readiness table, risk counts, and scatter;
  the Fleet Availability KPI is always computed over the full roster, and
  Aircraft Detail and Maintenance Plan always show the full roster/plan so
  nothing scheduled is hidden by a filter.
