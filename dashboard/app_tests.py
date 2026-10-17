"""AARIS dashboard — authentication behavior test suite (Streamlit AppTest).

Verifies the security-aware LIVE mode without needing a running ground
station (requests.get is monkeypatched), and DEMO MODE with no API and no
token at all:

  T1  LIVE + no credentials -> login form, ZERO network attempts
  T2  LIVE + token (Advanced field) -> every ground GET carries Authorization:
                          Bearer, data renders, token never appears in UI text
  T3  LIVE + 401       -> "Ground station rejected the dashboard credentials."
  T4  LIVE + 403       -> "Dashboard service account does not have permission..."
  T5  env prefill      -> DASHBOARD_SERVICE_TOKEN env var populates the field
  T6  DEMO MODE        -> works with no API (network stub raises) and no token
  T8  DEMO MODE button -> sidebar button click toggles OFF -> ON, renders demo,
                          no network
  T9  DEMO Acknowledge -> offline ack clears an alert, no network
  T10 one-click profile-> clicking Profile on an alert row switches the nav to
                          Aircraft Detail with THAT aircraft selected, offline
                          (row clicks in the tables use the same open_profile
                          path but need a real browser — AppTest cannot select a
                          st.dataframe row)
  T11 DEMO Models      -> the Aircraft Models section renders all 6 model
                          profiles (15 airframes each, 3 CRITICAL each) from the
                          bundled GET /models snapshot, no network

Sections render one at a time (nav = st.segmented_control backed by
session_state["nav_section"]), so profile assertions switch section via
goto_section() instead of relying on hidden-tab rendering.

Run (from dashboard/):  ../.venv/Scripts/python.exe app_tests.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")
with open(os.path.join(HERE, "demo_data.json"), "r", encoding="utf-8") as f:
    DEMO = json.load(f)

REAL_GET = requests.get
_ORIG_ENV_TOKEN = os.environ.get("DASHBOARD_SERVICE_TOKEN")  # for live T7


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def payload_for(url: str):
    if url.rstrip("/").endswith("/fleet"):
        return DEMO["fleet"]
    if "/schedule/latest" in url:
        return DEMO["schedule"]
    if url.rstrip("/").endswith("/models"):   # live aircraft-model profiles
        return DEMO["models"]
    if "/alerts" in url:                      # live critical-alerts poll
        return DEMO.get("alerts") or {"count": 0, "alerts": []}
    if "/fleet/" in url:                      # /fleet/{aircraft_id}
        aid = url.rsplit("/", 1)[-1]
        return DEMO["aircraft"].get(aid, {"aircraft": {"aircraft_id": aid}})
    raise AssertionError(f"unexpected URL {url}")


def install_fake(status: int = 200, explode: bool = False):
    """Patch requests.get; returns the list that records (url, headers)."""
    calls: list[tuple[str, dict]] = []

    def fake_get(url, timeout=None, headers=None, **kw):
        calls.append((url, headers or {}))
        if explode:
            raise requests.exceptions.ConnectionError("network must not be used")
        return FakeResp(status, payload_for(url))

    requests.get = fake_get
    return calls


def uninstall_fake():
    requests.get = REAL_GET


def run_app(token: str | None = None, demo: bool = False,
            env_token: str | None = ...):
    """Fresh AppTest run. Widget defaults pick up DASHBOARD_SERVICE_TOKEN from
    the environment, so env_token only needs the env set BEFORE at.run().
    env_token=...  -> leave the ambient env untouched.
    env_token=None -> force an EMPTY token (what the no-credentials tests need).
    """
    if env_token is ...:
        pass
    elif env_token is None:
        os.environ.pop("DASHBOARD_SERVICE_TOKEN", None)
    else:
        os.environ["DASHBOARD_SERVICE_TOKEN"] = env_token
    at = AppTest.from_file(APP, default_timeout=30)
    if demo:
        # DEMO MODE is a sidebar button now (session_state-backed); seed the
        # state directly instead of simulating the click (see T8 for a click).
        at.session_state["demo_mode"] = True
    at.run()
    if token is not None:
        at.sidebar.text_input[1].set_value(token).run()
    return at


def demo_button(at: AppTest):
    """The DEMO MODE toggle button (rendered in the sidebar, under the logo)."""
    hits = [b for b in at.sidebar.button if "DEMO MODE" in b.label]
    assert len(hits) == 1, f"expected exactly 1 DEMO MODE button, got {len(hits)}"
    return hits[0]


def goto_section(at: AppTest, section: str) -> AppTest:
    """Switch dashboard sections the way the nav bar does (session-state key)."""
    assert section in ("Fleet Readiness", "Aircraft Models", "Aircraft Detail",
                       "Maintenance Plan", "About"), section
    at.session_state["nav_section"] = section
    return at.run()


def metric_values(at: AppTest, label: str) -> list[str]:
    """Values of every st.metric with this label, as rendered strings."""
    return [str(m.value) for m in at.metric if m.label == label]


def expected_readiness() -> dict:
    """The Fleet Availability KPI the app *should* show for the bundled fixture.

    The KPI is deliberately time-dependent: plan days are offsets from
    plan['generated_at'], so "in maintenance today" moves as the demo ages
    (8 airframes on plan days 1-7, none once the 30-day horizon passes). A
    hardcoded percentage would therefore be wrong on most days. This helper
    re-derives the number from the fixture with its own copy of the documented
    formula; the tests then check the app agrees *and* that the
    date-independent invariants (roster, CRITICAL count, the
    available = total - CRITICAL - in-hangar identity) still hold.
    """
    fleet = DEMO["fleet"]["fleet"]
    plan = DEMO["schedule"]["plan"]
    critical = {a["aircraft_id"] for a in fleet if a.get("risk") == "CRITICAL"}
    base = datetime.strptime(plan["generated_at"], "%Y-%m-%dT%H:%M:%SZ").date()
    index = (datetime.now(timezone.utc).date() - base).days + 1
    in_hangar = ({t["aircraft_id"] for t in plan["tasks"]
                  if t["status"] == "SCHEDULED"
                  and t.get("scheduled_day") == index}
                 if 1 <= index <= plan.get("horizon_days", 30) else set())
    available = sum(1 for a in fleet
                    if a["aircraft_id"] not in critical
                    and a["aircraft_id"] not in in_hangar)
    return {
        "total": len(fleet),
        "critical": len(critical),
        "in_hangar": len(in_hangar),
        "available": available,
        "pct": f"{100 * available / len(fleet):.1f}%",
        "index": index,
    }


def ui_text(at: AppTest) -> str:
    """All user-visible text we can reach, for absence/presence assertions."""
    chunks = []
    for name in ("header", "subheader", "text", "caption", "error", "info",
                 "warning", "markdown", "metric", "selectbox", "exception"):
        coll = getattr(at, name, None)
        if coll is not None:
            chunks.extend(str(e.value) for e in coll)
    return "\n".join(chunks)


def no_exception(at: AppTest, name: str) -> None:
    assert len(at.exception) == 0, f"{name}: {at.exception[0].value}"


def main() -> int:
    checks = []

    # T1 — LIVE, no credentials: login form shown, zero network attempts
    calls = install_fake(explode=True)
    try:
        at = run_app(env_token=None)
        no_exception(at, "T1")
        txt = ui_text(at)
        assert "Sign in to the ground station" in txt, txt[:400]
        demo_button(at)  # sidebar DEMO MODE toggle still present (zero-cred path)
        assert calls == [], f"T1 tried the network anyway: {calls}"
        checks.append("T1 LIVE without credentials -> login form, no requests")
    finally:
        uninstall_fake()

    # T2 — LIVE with token: Authorization header on every GET, token not shown
    calls = install_fake()
    try:
        at = run_app(token="test-jwt-abc123")
        no_exception(at, "T2")
        paths = [u.split("8001", 1)[-1] for u, _ in calls]
        assert "/fleet" in paths and "/schedule/latest" in paths, paths
        assert all(h.get("Authorization") == "Bearer test-jwt-abc123"
                   for _, h in calls), calls
        txt = ui_text(at)
        assert "test-jwt-abc123" not in txt, "token material leaked into UI text"
        # the critical-alerts banner is fed by the /alerts poll
        assert "ACTIVE CRITICAL ALERT" in txt, txt[:400]
        assert "Fleet Readiness Decision Support" in txt, txt[:400]
        # Only the active section renders, so the profile needs a nav step —
        # and that GET must be authenticated too.
        goto_section(at, "Aircraft Detail")
        no_exception(at, "T2 profile section")
        joined = str([m.value for m in at.metric]) + ui_text(at)
        assert "112.76" in joined and "AC-017" in joined, joined[:400]
        assert "Synthetic demonstration fleet profile" in joined
        assert "test-jwt-abc123" not in joined, "token leaked after navigating"
        # the model-profile section is the other ground read (GET /models)
        goto_section(at, "Aircraft Models")
        no_exception(at, "T2 models section")
        joined = str([m.value for m in at.metric]) + ui_text(at)
        assert "Dassault Rafale EH" in joined and "Jaguar IS/IM" in joined, joined[:400]
        assert "test-jwt-abc123" not in joined, "token leaked on the models page"
        paths = [u.split("8001", 1)[-1] for u, _ in calls]
        assert "/fleet/AC-017" in paths and "/models" in paths, paths
        assert all(h.get("Authorization") == "Bearer test-jwt-abc123"
                   for _, h in calls), calls
        checks.append(f"T2 LIVE with token -> Bearer header on {len(calls)} GETs, "
                      "data + alert banner + profile + models render, token not displayed")
    finally:
        uninstall_fake()

    # T3 — 401 handled distinctly and readably
    install_fake(status=401)
    try:
        at = run_app(token="whatever")
        no_exception(at, "T3")
        assert "Ground station rejected the dashboard credentials" in ui_text(at)
        assert "does not have permission" not in ui_text(at)
        checks.append("T3 HTTP 401 -> 'rejected the dashboard credentials' message")
    finally:
        uninstall_fake()

    # T4 — 403 handled distinctly and readably
    install_fake(status=403)
    try:
        at = run_app(token="maintainer-token")
        no_exception(at, "T4")
        assert "Dashboard service account does not have permission" in ui_text(at)
        assert "rejected the dashboard credentials" not in ui_text(at)
        checks.append("T4 HTTP 403 -> 'does not have permission' message")
    finally:
        uninstall_fake()

    # T5 — env var prefills the sidebar field and is actually used
    calls = install_fake()
    try:
        at = run_app(env_token="env-token-xyz")
        no_exception(at, "T5")
        assert calls, "no requests made with env token"
        assert all(h.get("Authorization") == "Bearer env-token-xyz"
                   for _, h in calls), calls
        assert "env-token-xyz" not in ui_text(at)
        checks.append(f"T5 DASHBOARD_SERVICE_TOKEN env -> used on {len(calls)} GETs")
    finally:
        uninstall_fake()

    # T6 — DEMO MODE unchanged: no API, no token, network forbidden
    calls = install_fake(explode=True)
    try:
        at = run_app(demo=True, env_token=None)
        no_exception(at, "T6")
        assert calls == [], f"DEMO MODE touched the network: {calls}"
        exp = expected_readiness()
        # date-independent invariants of the frozen 90-aircraft fixture: 18
        # CRITICAL are grounded, and plan-day tasks can only take the number
        # down from there (up to MAX_DAILY_MAINTENANCE of them, minus any that
        # are already CRITICAL and so counted twice otherwise).
        assert (exp["total"], exp["critical"]) == (90, 18), exp
        assert exp["in_hangar"] <= 8, exp
        assert 64 <= exp["available"] <= 72, exp
        # the KPI row must show exactly what that formula yields today
        assert metric_values(at, "Fleet Availability") == [exp["pct"]], (
            metric_values(at, "Fleet Availability"), exp)
        assert metric_values(at, "Available now") == [
            f"{exp['available']} / {exp['total']}"], metric_values(at, "Available now")
        assert metric_values(at, "In maintenance today") == [
            str(exp["in_hangar"])], metric_values(at, "In maintenance today")
        assert metric_values(at, "At risk (CRITICAL + HIGH)") == [
            "36"], metric_values(at, "At risk (CRITICAL + HIGH)")
        # the bundled /alerts snapshot makes the critical-alert path demoable offline
        assert "ACTIVE CRITICAL ALERT" in ui_text(at)
        assert "bundled snapshot" in ui_text(at)
        goto_section(at, "Aircraft Detail")
        no_exception(at, "T6 profile section")
        joined = str([m.value for m in at.metric]) + ui_text(at)
        assert "112.76" in joined, joined[:400]
        assert "Synthetic demonstration fleet profile" in joined
        assert calls == [], f"DEMO MODE touched the network while navigating: {calls}"
        checks.append(f"T6 DEMO MODE without API/token -> {exp['pct']} KPI "
                      f"(plan day {exp['index']}, {exp['in_hangar']} in hangar) "
                      "+ 112.76 profile + bundled critical-alert banner, zero network")
    finally:
        uninstall_fake()

    # T8 — the sidebar DEMO MODE button visibly toggles the mode
    calls = install_fake(explode=True)
    try:
        os.environ.pop("DASHBOARD_SERVICE_TOKEN", None)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        no_exception(at, "T8")
        assert not at.session_state.get("demo_mode"), "demo should start OFF"
        assert "OFF" in demo_button(at).label, demo_button(at).label
        demo_button(at).click().run()
        no_exception(at, "T8 (after click)")
        assert at.session_state.get("demo_mode") is True, "button did not enable DEMO MODE"
        assert "ON" in demo_button(at).label
        assert calls == [], f"button click touched the network: {calls}"
        assert metric_values(at, "Fleet Availability") == [
            expected_readiness()["pct"]], metric_values(at, "Fleet Availability")
        # the nav bar is a state-driven segmented control with five sections
        assert list(at.segmented_control[0].options) == [
            "Fleet Readiness", "Aircraft Models", "Aircraft Detail",
            "Maintenance Plan", "About"], at.segmented_control[0].options
        checks.append("T8 DEMO MODE button click -> ON state, demo renders, no network")
    finally:
        uninstall_fake()

    # T9 — DEMO MODE Acknowledge works offline (session-local, never networked)
    calls = install_fake(explode=True)
    try:
        os.environ.pop("DASHBOARD_SERVICE_TOKEN", None)
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["demo_mode"] = True
        at.run()
        no_exception(at, "T9")
        n_before = len([b for b in at.button if b.label == "Acknowledge"])
        assert n_before == len(DEMO["alerts"]["alerts"]), (
            n_before, len(DEMO["alerts"]["alerts"]))
        banner = [e.value for e in at.error if "ACTIVE CRITICAL ALERT" in e.value]
        assert banner and banner[0].startswith(f"{n_before} ACTIVE"), banner
        [b for b in at.button if b.label == "Acknowledge"][0].click().run()
        no_exception(at, "T9 (after ack)")
        n_after = len([b for b in at.button if b.label == "Acknowledge"])
        assert n_after == n_before - 1, (n_before, n_after)
        assert calls == [], f"DEMO MODE ack touched the network: {calls}"
        checks.append(f"T9 DEMO MODE Acknowledge -> {n_before} alerts down to "
                      f"{n_after}, no network")
    finally:
        uninstall_fake()

    # T10 — one click from a critical-alert row to that aircraft's profile:
    # the same open_profile path a table-row click takes (AppTest cannot select
    # an st.dataframe row, so the browser check covers that widget separately).
    calls = install_fake(explode=True)
    try:
        os.environ.pop("DASHBOARD_SERVICE_TOKEN", None)
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["demo_mode"] = True
        at.run()
        no_exception(at, "T10")
        assert at.session_state.get("nav_section") == "Fleet Readiness", \
            at.session_state.get("nav_section")
        target = DEMO["alerts"]["alerts"][0]["aircraft_id"]
        profiles = [b for b in at.button if b.label == "Profile"]
        assert len(profiles) == len(DEMO["alerts"]["alerts"]), len(profiles)
        profiles[0].click().run()
        no_exception(at, "T10 (after click)")
        assert at.session_state.get("nav_section") == "Aircraft Detail", \
            at.session_state.get("nav_section")
        assert at.session_state.get("detail_id") == target, \
            at.session_state.get("detail_id")
        assert at.selectbox[0].value == target, at.selectbox[0].value
        joined = str([m.value for m in at.metric]) + ui_text(at)
        assert "Synthetic demonstration fleet profile" in joined
        assert target in joined, joined[:400]
        assert calls == [], f"click-through touched the network: {calls}"
        checks.append(f"T10 Profile click -> Aircraft Detail on {target}: nav and "
                      "picker both follow, no network")
    finally:
        uninstall_fake()

    # T11 — the Aircraft Models section renders the bundled GET /models snapshot
    # offline: one profile per aircraft model, 15 airframes each.
    calls = install_fake(explode=True)
    try:
        os.environ.pop("DASHBOARD_SERVICE_TOKEN", None)
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["demo_mode"] = True
        at.run()
        no_exception(at, "T11")
        goto_section(at, "Aircraft Models")
        no_exception(at, "T11 (models section)")
        snapshot = DEMO["models"]["models"]
        assert len(snapshot) == 6 and DEMO["models"]["aircraft_total"] == 90
        heads = [s.value for s in at.subheader]
        for m in snapshot:
            assert f"{m['type']} · {m['model_code']}" in heads, heads
        assert "6 aircraft models · 90 airframes" in ui_text(at)
        assert metric_values(at, "Airframes") == ["15"] * 6, metric_values(at, "Airframes")
        assert metric_values(at, "Grounded (CRITICAL)") == ["3"] * 6
        assert metric_values(at, "Ready to fly") == ["12"] * 6
        assert metric_values(at, "In 30-day plan") == ["9 / 6"] * 6
        txt = ui_text(at)
        assert "no telemetry 2" in txt and "illustrative type image" in txt, txt[:400]
        assert "No real IAF operational or maintenance data" in txt
        assert calls == [], f"models section touched the network: {calls}"
        checks.append("T11 DEMO MODE Aircraft Models -> 6 model profiles x 15 "
                      "airframes (3 CRITICAL / 12 ready / 9 scheduled each), no network")
    finally:
        uninstall_fake()

    # Optional T7 — LIVE against the REAL ground station if one is up
    try:
        REAL_GET("http://127.0.0.1:8001/health", timeout=2)
    except requests.exceptions.RequestException:
        print("T7 SKIPPED (no ground station on :8001 — run it to verify LIVE end-to-end)")
    else:
        tok = _ORIG_ENV_TOKEN
        if not tok:
            print("T7 SKIPPED (ground station up, but DASHBOARD_SERVICE_TOKEN not set)")
        else:
            at = run_app(env_token=tok)
            no_exception(at, "T7")
            assert len(at.error) == 0, [e.value for e in at.error]
            joined = str([m.value for m in at.metric]) + ui_text(at)
            assert "112.76" in joined  # AC-017 fixture from the secured station
            checks.append("T7 LIVE against the real secured ground station -> renders")

    for c in checks:
        print("[PASS]", c)
    print(f"\nPASS: dashboard auth suite ({len(checks)} checks) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
