# docs/screenshots

Every PNG here is a real capture of the running AARIS stack — no mock-ups.
They are referenced from the repository README and are not needed at runtime
(the system works with this folder empty).

State captured: the fresh demo baseline from `run_demo.ps1 -Reset` on
**2026-10-17** — 90 aircraft (6 models x 15 airframes), 390 health reports,
plan **OPTIMAL, 54 scheduled / 36 deferred**, peak 8 tasks/day, 18 ACTIVE
critical alerts, Fleet Availability **80.0 % (72 / 90)**.

| File | What it shows |
| --- | --- |
| `01_login.png` | Dashboard sign-in screen: AARIS banner, ground station URL, the credential form and its "local demo authentication … no real operational data" note. |
| `02_fleet_readiness.png` | Fleet Readiness section: KPI row (**80.0 % Fleet Availability**, **72 / 90 available now**, **36 at risk**, **8 in maintenance today**), the roster breakdown caption, the per-aircraft AVAILABLE / GROUNDED / IN-MAINTENANCE-TODAY table, and the 18-18-18-24 risk counts. |
| `03_critical_alerts.png` | The persistent alert block: **18 ACTIVE CRITICAL ALERT(S)** banner with one row per grounded airframe (AC-079 … AC-001), each with **Acknowledge** and **Profile** actions. |
| `04_aircraft_models.png` | Aircraft Models section: the synthetic-data disclaimer, "6 aircraft models · 90 airframes", and the RAFALE-EH profile card (type photo, role/origin/engines, 15 airframes · 12 ready · 3 grounded · avg RUL 64.88 · 9/6 in plan, risk mix, issue history) above its airframe table. |
| `05_model_airframes.png` | The SU30MKI card and its 15-airframe roster table (`plan_status` / `plan_day` per aircraft; AC-017 shows the 112.76-cycle edge fixture as DEFERRED). |
| `06_aircraft_detail.png` | Aircraft Detail on **AC-017**: picker, RUL 112.76 / health 90.21 % / LOW / 5 reports, the Su-30MKI profile card, and the synthetic issue-history handling. |
| `07_maintenance_plan.png` | Maintenance Plan: generated-at + **OPTIMAL**, 30-day horizon, max 8/day; **54 scheduled / 36 deferred / peak 8**; the CP-SAT Gantt (Oct 17-24, coloured by risk) and the All-tasks table with deadlines and reasons. |
| `08_about.png` | About: the edge → ground → CP-SAT explanation, how the 6 x 15 roster is built, and where each number comes from (model output vs operational heuristic). |
| `09_edge_docs.png` | Edge node OpenAPI contract at `http://127.0.0.1:8000/docs`: `GET /health`, `POST /predict`, `GET /report`. |
| `10_ground_docs.png` | Ground station OpenAPI contract at `http://127.0.0.1:8001/docs`: login, ingest, `/fleet`, `/fleet/{id}`, `/models`, `/alerts`, `/alerts/{id}/ack`, `/schedule`, `/schedule/latest`, `/health` — lock icons show the RBAC-protected routes. |

## Regenerating them

The script drives headless Chrome over the DevTools Protocol, signs in through
the dashboard's own login form, clicks each nav section and screenshots it. It
is read-only: it never posts a report, acks an alert or touches the database.

```powershell
# 1. start a clean demo (seeds 90 aircraft, generates the plan)
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset -NoBrowser

# 2. capture all ten PNGs into docs\screenshots\
.venv\Scripts\python.exe demo\capture_screenshots.py
```

Options: `--user/--password` (default `commander`/`commander123`),
`--width/--height` (default 1440x1100), `--chrome <path>` if neither
Chrome nor Edge is in a standard location, `--dashboard/--edge/--ground` for
non-default URLs.

Two gotchas the script already handles, worth knowing if you edit it:

- Start the dashboard **without** `DASHBOARD_SERVICE_TOKEN` or the login
  screen (`01_login.png`) never appears — the app auto-signs in.
- Streamlit paints skeleton placeholders before the real widgets, so waits
  target selectors (`wait_for_selector`) rather than body text, and each
  section is captured immediately after its nav click — the nav is a
  single-page state switch, so a later click would replace what is on screen.

If you re-seed the fleet, re-capture `dashboard/demo_data.json`
(`.venv\Scripts\python.exe demo\capture_demo_data.py`) too, so DEMO MODE,
`dashboard/app_tests.py` and these pictures all describe the same state.
