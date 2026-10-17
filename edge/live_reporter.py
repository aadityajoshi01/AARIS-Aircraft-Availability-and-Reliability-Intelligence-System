"""AARIS — live edge reporter (per-aircraft demo feed).

Treats an edge node (a Jetson at a real deployment; any machine for the demo)
as ONE aircraft that talks to the ground station over a local port:

  * posts a health report on a routine cadence (default 15 min = 900 s) so the
    aircraft's profile at the ground updates continuously, and
  * when a reading crosses into CRITICAL, sends an urgent report immediately and
    keeps re-reporting on a short cadence until the reading recovers.

`--fleet` runs the same feed for every aircraft on the roster at once, each with
its own token, timeline phase and clock offset, so the base station shows 25
profiles updating live instead of only AC-017.

The ground station auto-raises a critical alert on any CRITICAL report
(POST /ingest_report -> alerts table), which the dashboard shows as a live
banner. This reporter does NOT compute RUL — it streams a SYNTHETIC degradation
timeline so the whole low -> critical -> alert path is visible on cue. Nothing
here touches the frozen Ridge artifacts or the /predict path in edge_node.py.

Auth: sends `Authorization: Bearer <EDGE_<AIRCRAFT>_TOKEN>`, a static device
token bound to exactly that one aircraft (default: the guessable local-demo
value). The token is never logged.

Run (from `edge/`), local host:
    ../.venv/Scripts/python.exe live_reporter.py

Fast rehearsal (a 15-min tick every 20 s so a full low->critical->alert cycle
plays in ~2-3 minutes; Ctrl-C to stop):
    ../.venv/Scripts/python.exe live_reporter.py --fast

Every aircraft on the roster, live and staggered (same rehearsal speeds):
    ../.venv/Scripts/python.exe live_reporter.py --fast --ramp --fleet

One-shot critical to test the alert path right now:
    ../.venv/Scripts/python.exe live_reporter.py --once --level CRITICAL

Point it at a base station on another machine (LAN) with:
    --ground-url http://<host-ip>:8001   (run the ground with GROUND_HOST=0.0.0.0)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

AIRCRAFT_ID = "AC-017"
DEVICE_TOKEN = os.environ.get("EDGE_AC017_TOKEN", "edge-ac017-dev-token")
DEFAULT_GROUND = os.environ.get("GROUND_URL", "http://127.0.0.1:8001")

MAX_RUL = 125.0
DEFAULT_INTERVAL = 900.0      # routine report cadence (s) = 15 min
DEFAULT_RECHECK = 60.0        # urgent cadence while CRITICAL (s)
DEFAULT_POLL = 5.0            # sleep granularity (s)
DEFAULT_FLEET_SIZE = 90       # roster size fallback (see roster_ids())

# SYNTHETIC degradation timeline: predicted RUL (cycles) per successive reading.
# Walks a healthy engine down into CRITICAL (<30) then loops, so the profile
# updates live and a critical alert fires on cue. Values are illustrative only.
RUL_TIMELINE = [112.76, 98.4, 88.1, 79.5, 71.0, 63.2, 55.8, 48.9,
                42.5, 36.1, 31.0, 27.4, 22.8, 18.5, 15.2, 24.0, 41.0]

_ACTION_BY_RISK = {
    "LOW": "Routine monitoring; no action. Fly per scheduled maintenance program.",
    "MEDIUM": "Add to planned-maintenance queue; inspect at next convenient turnaround.",
    "HIGH": "Book maintenance intervention within the current cycle window; stage parts.",
    "CRITICAL": "Ground the engine. Do not dispatch until inspected and cleared by maintenance.",
}

# Level presets for --once: force a report at a specific risk band immediately.
_LEVEL_RUL = {
    "LOW": 110.0,
    "MEDIUM": 70.0,
    "HIGH": 45.0,
    "CRITICAL": 21.0,
}


def risk_band(rul_cycles: float) -> str:
    """Same bands as edge_node._risk_band (>=100/60/30 else CRITICAL)."""
    if rul_cycles >= 100:
        return "LOW"
    if rul_cycles >= 60:
        return "MEDIUM"
    if rul_cycles >= 30:
        return "HIGH"
    return "CRITICAL"


def build_report(aircraft_id: str, rul_cycles: float) -> dict:
    """Assemble a ground-compatible EdgeReport payload for one reading."""
    rul = round(float(rul_cycles), 2)
    health = round(min(100.0, max(0.0, rul / MAX_RUL * 100.0)), 2)
    risk = risk_band(rul)
    return {
        "aircraft_id": aircraft_id,
        "engine_id": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rul_cycles": rul,
        "engine_health": health,
        "risk": risk,
        "top_issues": [{"sensor": "simulated_degradation", "impact_cycles": 0.0,
                        "note": "synthetic live-feed demo reading"}],
        "recommended_action": _ACTION_BY_RISK[risk],
    }


def push(ground_url: str, report: dict, token: str, timeout: float = 8.0):
    """Best-effort POST of one report to /ingest_report.

    Returns (ok, parsed_response_or_error_string). The token only ever goes in
    the Authorization header; it is never printed."""
    url = ground_url.rstrip("/") + "/ingest_report"
    body = json.dumps(report).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return True, {"_raw": raw}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        return False, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _emit(report: dict, ok: bool, result, urgent: bool) -> None:
    tag = "URGENT/CRITICAL" if urgent else "routine"
    if ok:
        alert = isinstance(result, dict) and result.get("alert_created")
        extra = "  -> ground raised a critical alert" if alert else ""
        print(f"[reporter] sent {tag} report "
              f"aircraft={report['aircraft_id']} rul={report['rul_cycles']} "
              f"risk={report['risk']} at {report['timestamp']}{extra}")
    else:
        print(f"[reporter] FAILED to send {tag} report: {result}", file=sys.stderr)


def sleep_until(deadline: float, stop: dict) -> None:
    """Sleep until `deadline` in <= poll-size chunks so Ctrl-C stays responsive."""
    while time.monotonic() < deadline and not stop.get("stop"):
        remaining = deadline - time.monotonic()
        time.sleep(max(0.05, min(DEFAULT_POLL, remaining)))


def env_slot(aircraft_id: str) -> str:
    """Env var holding this aircraft's edge device token."""
    return f"EDGE_{aircraft_id.replace('-', '').upper()}_TOKEN"


def dev_token_for(aircraft_id: str) -> str:
    """Local-demo device token, mirroring ground/security.dev_token_for() exactly
    (AC-017 -> edge-ac017-dev-token). Guessable by design; override with
    EDGE_<AIRCRAFT>_TOKEN on both sides for anything real."""
    return "edge-" + aircraft_id.replace("-", "").lower() + "-dev-token"


def fleet_ids(count: int) -> list[str]:
    """The first `count` aircraft on the ground roster — the fleet feed should
    never stream a tail number the base station doesn't know about."""
    return roster_ids()[:count]


def roster_ids() -> list[str]:
    """Aircraft ids from ground/fleet_config.json (the file that seeds the base
    station). Falls back to AC-001..AC-090 when it isn't present, e.g. when this
    node runs standalone on a Jetson."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "ground", "fleet_config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            ids = [a["aircraft_id"] for a in json.load(f).get("fleet", [])
                   if a.get("aircraft_id")]
    except (OSError, ValueError, KeyError, TypeError):
        ids = []
    return ids or [f"AC-{i:03d}" for i in range(1, DEFAULT_FLEET_SIZE + 1)]


FLEET_SIZE = len(roster_ids())


def run_fleet(ground_url: str, ids: list[str], interval: float, recheck: float,
              ramp: bool) -> int:
    """Every aircraft gets its own live profile feed: same synthetic timeline,
    started at a different point and on a different clock phase, so the ground
    station sees a continuous stream of per-aircraft reports and CRITICAL
    readings (and their alerts) arrive staggered rather than all at once."""
    n = len(ids)
    print(f"[reporter] streaming SYNTHETIC live feed for {n} aircraft -> "
          f"{ground_url} (routine cadence {interval:.0f}s per aircraft, urgent "
          f"recheck {recheck:.0f}s, phases staggered). Ctrl-C to stop.")
    stop = {"stop": False}
    now = time.monotonic()
    craft = [{
        "aircraft_id": aid,
        "token": os.environ.get(env_slot(aid)) or dev_token_for(aid),
        "phase": (i * 3) % len(RUL_TIMELINE),
        "step": 0,
        # spread first reports across one interval so there is no burst
        "due": now + (interval * i / n),
    } for i, aid in enumerate(ids)]

    while not stop.get("stop"):
        due = [c for c in craft if c["due"] <= time.monotonic()]
        if not due:
            sleep_until(min(c["due"] for c in craft), stop)
            continue
        for c in due:
            rul = RUL_TIMELINE[(c["phase"] + c["step"]) % len(RUL_TIMELINE)]
            c["step"] += 1
            report = build_report(c["aircraft_id"], rul)
            urgent = report["risk"] == "CRITICAL"
            ok, result = push(ground_url, report, c["token"])
            _emit(report, ok, result, urgent)
            if urgent:
                wait = recheck
            elif ramp and report["risk"] != "LOW":
                wait = recheck
            else:
                wait = interval
            c["due"] = time.monotonic() + wait
        # an aircraft that just went CRITICAL is due immediately (urgent push)
        continue_now = any(c["due"] <= time.monotonic() for c in craft)
        if not continue_now:
            sleep_until(min(c["due"] for c in craft), stop)
    print("\n[reporter] stopped")
    return 0


def run_once(ground_url: str, aircraft_id: str, token: str, level: str,
            rul: float | None) -> int:
    report = build_report(aircraft_id, rul if rul is not None else _LEVEL_RUL[level])
    ok, result = push(ground_url, report, token)
    _emit(report, ok, result, urgent=(report["risk"] == "CRITICAL"))
    return 0 if ok else 1


def run_loop(ground_url: str, aircraft_id: str, token: str,
             interval: float, recheck: float, ramp: bool) -> int:
    print(f"[reporter] streaming SYNTHETIC live feed for {aircraft_id} -> "
          f"{ground_url} (routine cadence {interval:.0f}s, urgent recheck "
          f"{recheck:.0f}s). Ctrl-C to stop.")
    stop = {"stop": False}
    idx = 0
    while not stop.get("stop"):
        rul = RUL_TIMELINE[idx % len(RUL_TIMELINE)]
        idx += 1
        report = build_report(aircraft_id, rul)
        urgent = report["risk"] == "CRITICAL"
        ok, result = push(ground_url, report, token)
        _emit(report, ok, result, urgent)

        # Urgent cadence only while CRITICAL; routine otherwise. If ramp mode is
        # on, hold the long cadence only during the healthy LOW stretch.
        if urgent:
            wait = recheck
        elif ramp and report["risk"] == "LOW":
            wait = interval
        else:
            wait = interval if not ramp else recheck

        # A CRITICAL reading is sent immediately (no extra wait); otherwise we
        # honor the computed cadence before the next reading.
        if urgent:
            continue
        sleep_until(time.monotonic() + wait, stop)
    print("\n[reporter] stopped")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AARIS live edge reporter (demo feed).")
    p.add_argument("--ground-url", default=DEFAULT_GROUND,
                   help="Base station URL (default %(default)s; use http://<host-ip>:8001 for LAN).")
    p.add_argument("--aircraft-id", default=AIRCRAFT_ID,
                   help="Aircraft this edge device reports for (default %(default)s).")
    p.add_argument("--token", default=DEVICE_TOKEN,
                   help="Edge device bearer token (default from EDGE_AC017_TOKEN).")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help="Routine report cadence in seconds (default 900 = 15 min).")
    p.add_argument("--critical-recheck", type=float, default=DEFAULT_RECHECK,
                   help="Urgent cadence in seconds while a reading is CRITICAL (default 60).")
    p.add_argument("--fast", action="store_true",
                   help="Rehearsal shortcut: interval=20s, critical-recheck=5s.")
    p.add_argument("--ramp", action="store_true",
                   help="Advance on the urgent cadence for every non-LOW reading "
                        "(faster ramp to critical in a live demo).")
    p.add_argument("--once", action="store_true",
                   help="Send exactly one report then exit.")
    p.add_argument("--level", default="CRITICAL", choices=list(_LEVEL_RUL),
                   help="Risk level for --once (default CRITICAL).")
    p.add_argument("--rul", type=float, default=None,
                   help="Explicit RUL for --once (overrides --level).")
    p.add_argument("--fleet", action="store_true",
                   help="Stream a separate live profile feed for every aircraft "
                        f"in the roster (AC-001..AC-{FLEET_SIZE:03d}) instead of "
                        "just --aircraft-id. Each aircraft uses its own device "
                        "token (EDGE_<AIRCRAFT>_TOKEN, dev default "
                        "edge-ac###-dev-token). NOTE: overwrites the canonical "
                        "AC-017 112.76 reading and shifts the plan/KPI numbers.")
    p.add_argument("--fleet-count", type=int, default=FLEET_SIZE,
                   help=f"Aircraft to feed with --fleet, 1..{FLEET_SIZE} "
                        f"(default {FLEET_SIZE} = whole roster).")
    args = p.parse_args(argv)

    if args.rul is not None and not (0.0 <= args.rul <= MAX_RUL):
        p.error(f"--rul must be within [0, {MAX_RUL}]")

    if args.once:
        return run_once(args.ground_url, args.aircraft_id, args.token,
                        args.level, args.rul)

    interval = args.interval
    recheck = args.critical_recheck
    if args.fast:
        interval = min(interval, 20.0)
        recheck = min(recheck, 5.0)

    if args.fleet:
        if not (1 <= args.fleet_count <= FLEET_SIZE):
            p.error(f"--fleet-count must be within [1, {FLEET_SIZE}]")
        try:
            return run_fleet(args.ground_url, fleet_ids(args.fleet_count),
                             interval, recheck, args.ramp)
        except KeyboardInterrupt:
            print("\n[reporter] stopped (KeyboardInterrupt)")
            return 0

    try:
        return run_loop(args.ground_url, args.aircraft_id, args.token,
                        interval, recheck, args.ramp)
    except KeyboardInterrupt:
        print("\n[reporter] stopped (KeyboardInterrupt)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
