"""AARIS — build the synthetic fleet: fleet_config.json + aircraft_profiles.json.

Generates a 90-aircraft demonstration fleet as 6 aircraft MODELS x 15 airframes
each, every one of them clearly SYNTHETIC. This script is the single source of
truth for the roster: it writes both ground/fleet_config.json (the metadata the
ground station seeds into SQLite) and ground/aircraft_profiles.json (the
profiles the station attaches to GET /fleet/{id} and aggregates in GET /models).

DATA PROVENANCE — read before editing:
  * The RUL model behind every report is trained on NASA's public C-MAPSS
    dataset (FD001). Nothing here comes from any real air force.
  * Aircraft identities, maintenance dates, issue logs, summaries and
    actions are FICTIONAL, generated deterministically from a fixed seed.
    They do NOT represent, reference, or imply access to real Indian Air
    Force (or any other) maintenance records, tail numbers, sortie data,
    defect reports, or classified information.
  * The per-type photos under dashboard/assets/plane_types/ are illustrative
    TYPE images only (user-supplied). They are never presented as the specific
    fictional airframe.
  * base / mission_priority / spare_available follow the pattern established on
    the original AC-001..AC-025 roster exactly (base cycles over 10 synthetic
    base codes, mission_priority cycles 1..5, no spare on every third airframe)
    and AC-026..AC-090 simply continue it.

MODEL BLOCKS — airframes are grouped so a model's 15 aircraft are contiguous:
  AC-001-015 Dassault Rafale EH   AC-016-030 Sukhoi Su-30MKI
  AC-031-045 HAL Tejas Mk1A       AC-046-060 MiG-29UPG
  AC-061-075 Mirage 2000I/TI      AC-076-090 Jaguar IS/IM
The Su-30MKI block deliberately starts at AC-016 so that **AC-017 remains a
Sukhoi Su-30MKI at exactly 112.76 cycles / 90.21% / LOW** — the verified edge
fixture every test suite and document quotes.

Within every block the 15 airframes occupy the same health "slots", so each
model shows a realistic spread instead of one long degradation tail:
  slot 0..3   3 CRITICAL + 1 healthy showcase (slot 1)
  slot 4..6   3 HIGH
  slot 7..9   3 MEDIUM
  slot 10..12 3 more LOW
  slot 13..14 2 airframes with NO report yet (telemetry gap demo)
i.e. 3 CRITICAL / 3 HIGH / 3 MEDIUM / 4 LOW / 2 unreported per model, which is
18 / 18 / 18 / 24 / 12 across the 90-aircraft fleet.

The "reports" block per aircraft carries both the current RUL (consumed by
seed_demo.py as the SEED list, so the seed and the profiles can never drift
apart) and the synthetic degradation TREND that seed_demo.py ingests as older
reports to populate the dashboard trend chart. seed_demo ingests history first
and the current report last, so the latest state — and therefore the CP-SAT
scheduler input — is always the current value above.

Run (from ground/):  ../.venv/Scripts/python.exe make_profiles.py
"""

import json
import os
import random
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_CONFIG = os.path.join(HERE, "fleet_config.json")
OUT = os.path.join(HERE, "aircraft_profiles.json")

TODAY = date(2026, 10, 11)          # fixed reference date -> deterministic file
MAX_RUL = 125.0
PER_MODEL_COUNT = 15

BASES = ["DEL", "BOM", "HYD", "BLR", "CCU", "GAIA", "BBI", "ASN", "TRV", "COK"]

# Same bands as seed_demo.risk_band / edge node.
def risk_band(rul: float) -> str:
    if rul >= 100:
        return "LOW"
    if rul >= 60:
        return "MEDIUM"
    if rul >= 30:
        return "HIGH"
    return "CRITICAL"


TYPE_TABLE = {
    "Sukhoi Su-30MKI": {
        "model_code": "SU30MKI",
        "role": "Air superiority / deep strike",
        "origin": "Russia — HAL licensed assembly",
        "engines": "2 × AL-31FP class turbofans",
        "image": "su30mki.jpg",
        "codes": ["ENG-OIL-CHIP", "ENG-VIB-HIGH", "HYD-PRESS-DRIFT", "GEAR-INDICATION"],
    },
    "Dassault Rafale EH": {
        "model_code": "RAFALE-EH",
        "role": "Omnirole fighter",
        "origin": "France",
        "engines": "2 × M88 class turbofans",
        "image": "rafale.jpg",
        "codes": ["AVIONICS-SOFTWARE", "COOLING-EFF", "RWR-FAULT", "ENG-OIL-PRESS-LOW"],
    },
    "HAL Tejas Mk1A": {
        "model_code": "TEJAS-MK1A",
        "role": "Light multirole fighter",
        "origin": "India — HAL",
        "engines": "1 × F404 class turbofan",
        "image": "tejas.jpg",
        "codes": ["ENG-OIL-PRESS-LOW", "FBW-FAULT", "AVIONICS-SOFTWARE", "GEAR-INDICATION"],
    },
    "MiG-29UPG": {
        "model_code": "MG29UPG",
        "role": "Frontline fighter / upgrade",
        "origin": "Russia — upgraded",
        "engines": "2 × RD-33 class turbofans",
        "image": "mig29upg.jpg",
        "codes": ["ENGINE-EGT-MARGIN", "ENG-VIB-HIGH", "RADAR-CAL", "ENG-OIL-CHIP"],
    },
    "Mirage 2000I/TI": {
        "model_code": "M2000IT",
        "role": "Precision strike / trainer conversion",
        "origin": "France",
        "engines": "1 × M53 class turbofan",
        "image": "mirage2000.jpg",
        "codes": ["RADAR-CAL", "HYD-PRESS-DRIFT", "AVIONICS-INTERMITTENT", "COOLING-EFF"],
    },
    "Jaguar IS/IM": {
        "model_code": "JAGUAR-IM",
        "role": "Strike / dual-seat conversion",
        "origin": "Anglo-French — HAL built",
        "engines": "2 × Adour class turbofans",
        "image": "jaguar.jpg",
        "codes": ["HYD-LEAK", "STRUCT-CORROSION", "STRUCT-AGEING", "AVIONICS-INTERMITTENT", "GEAR-INDICATION"],
    },
}

# One health ladder per model: 15 slot values (None = no telemetry yet).
# Slot 1 is each model's healthy showcase airframe; for the Su-30MKI block that
# is AC-017 and MUST stay exactly 112.76 (the verified edge /predict fixture).
# Slot 2 values are all >= 6.5 cycles so a no-spare airframe landing in that
# slot still has a feasible day-5 slot under the 5-day spare lead time.
MODEL_LADDERS = {
    "Dassault Rafale EH": [2.4, 108.6, 7.0, 13.5, 33.0, 45.0, 57.5,
                           64.0, 76.5, 93.0, 102.5, 117.0, 123.5, None, None],
    "Sukhoi Su-30MKI": [1.6, 112.76, 6.5, 12.0, 31.5, 42.0, 55.0,
                        67.5, 79.0, 90.5, 105.0, 119.5, 122.0, None, None],
    "HAL Tejas Mk1A": [3.0, 104.5, 8.5, 15.0, 36.0, 48.5, 58.5,
                       62.0, 73.0, 96.0, 101.0, 115.0, 124.0, None, None],
    "MiG-29UPG": [5.5, 116.0, 10.0, 20.5, 30.5, 51.0, 59.0,
                  71.0, 84.5, 88.0, 108.0, 121.0, 124.0, None, None],
    "Mirage 2000I/TI": [2.0, 121.0, 9.0, 24.0, 38.0, 41.5, 52.0,
                        68.0, 81.0, 98.5, 103.5, 110.0, 119.0, None, None],
    "Jaguar IS/IM": [1.5, 109.5, 6.8, 11.0, 34.5, 47.0, 56.5,
                     63.5, 77.0, 94.5, 106.0, 118.5, 122.0, None, None],
}

# Block order decides the AC- ranges documented above.
MODEL_BLOCKS = ["Dassault Rafale EH", "Sukhoi Su-30MKI", "HAL Tejas Mk1A",
                "MiG-29UPG", "Mirage 2000I/TI", "Jaguar IS/IM"]


def roster_ids(count: int = len(MODEL_BLOCKS) * PER_MODEL_COUNT) -> list[str]:
    return [f"AC-{i + 1:03d}" for i in range(count)]


def build_assignment() -> list[str]:
    """airframe index -> model name, one contiguous block of 15 per model."""
    return [name for name in MODEL_BLOCKS for _ in range(PER_MODEL_COUNT)]


def build_current_rul() -> dict[str, float | None]:
    """airframe id -> current seeded RUL (None = no report yet)."""
    out: dict[str, float | None] = {}
    for i, aid in enumerate(roster_ids()):
        out[aid] = MODEL_LADDERS[ASSIGNMENT[i]][i % PER_MODEL_COUNT]
    return out


ASSIGNMENT = build_assignment()
CURRENT_RUL = build_current_rul()

# fleet_config.json keeps its original description/schema verbatim (functional
# identifier from before the AARIS rebrand); only the roster itself is generated.
FLEET_CONFIG_DESCRIPTION = "AeroGuard fleet roster — seeds ground/fleet.db aircraft metadata at startup."
AIRCRAFT_SCHEMA = ["aircraft_id", "base", "mission_priority", "spare_available"]

MAINT_TYPES = ["Periodic Inspection", "Engine Condition Check", "Avionics Retrofit",
               "Structural Survey", "Landing Gear Overhaul", "Composite Repair"]

SYSTEM_OF = {
    "ENG-OIL-CHIP": "ENGINE", "ENG-VIB-HIGH": "ENGINE", "ENG-OIL-PRESS-LOW": "ENGINE",
    "ENGINE-EGT-MARGIN": "ENGINE", "FBW-FAULT": "FLIGHT CONTROLS",
    "RWR-FAULT": "SELF-PROTECTION", "AVIONICS-SOFTWARE": "AVIONICS",
    "AVIONICS-INTERMITTENT": "AVIONICS", "COOLING-EFF": "ENVIRONMENTAL",
    "RADAR-CAL": "RADAR", "HYD-PRESS-DRIFT": "HYDRAULICS", "HYD-LEAK": "HYDRAULICS",
    "GEAR-INDICATION": "LANDING GEAR", "STRUCT-CORROSION": "STRUCTURE",
    "STRUCT-AGEING": "STRUCTURE",
}

SUMMARY_OF = {  # clearly fictional one-liners
    "ENG-OIL-CHIP": "Synthetic scenario: chip detector finding flagged during routine oil sample.",
    "ENG-VIB-HIGH": "Synthetic scenario: vibration channel above advisory band on one bore sensor.",
    "ENG-OIL-PRESS-LOW": "Synthetic scenario: oil pressure trend drifting below nominal band at idle.",
    "ENGINE-EGT-MARGIN": "Synthetic scenario: EGT margin reduced vs fleet baseline on climb.",
    "FBW-FAULT": "Synthetic scenario: flight-control built-in test reported a transient fault code.",
    "RWR-FAULT": "Synthetic scenario: threat-receiver self-test anomaly after ground power cycle.",
    "AVIONICS-SOFTWARE": "Synthetic scenario: mission computer reboot logged during pre-flight BIT.",
    "AVIONICS-INTERMITTENT": "Synthetic scenario: intermittent display bus dropout reported on ground run.",
    "COOLING-EFF": "Synthetic scenario: bay temperature higher than expected on hot-day taxi.",
    "RADAR-CAL": "Synthetic scenario: calibration check outside tolerance on ground test set.",
    "HYD-PRESS-DRIFT": "Synthetic scenario: hydraulic pressure slow bleed between daily inspections.",
    "HYD-LEAK": "Synthetic scenario: seep found at coupling during walk-around (staged finding).",
    "GEAR-INDICATION": "Synthetic scenario: gear position indication flicker on retraction test.",
    "STRUCT-CORROSION": "Synthetic scenario: surface corrosion noted at inspection zone (staged).",
    "STRUCT-AGEING": "Synthetic scenario: fatigue-life accounting milestone reached for the airframe block.",
}

ACTION_BY_STATUS = {
    "OPEN": "Maintenance action pending — queued for next hangar slot.",
    "MONITOR": "Under trend monitoring; no intervention yet.",
    "CLOSED": "Rectified and signed off at the synthetic base depot.",
}


def build_roster() -> list[dict]:
    """fleet_config.json entries — the original cycling pattern, continued to 90."""
    fleet = []
    for i, aid in enumerate(roster_ids()):
        fleet.append({
            "aircraft_id": aid,
            "base": BASES[i % len(BASES)],
            "mission_priority": (i % 5) + 1,
            "spare_available": (i + 1) % 3 != 0,   # no spare on every third airframe
        })
    return fleet


def build_issue_history(rng: random.Random, aid: str, risk: str | None) -> list[dict]:
    """0-4 synthetic issues; degraded aircraft get more/OPEN, healthy get few/CLOSED."""
    if risk == "CRITICAL":
        n = rng.choice([3, 4])
    elif risk == "HIGH":
        n = rng.choice([2, 3])
    elif risk == "MEDIUM":
        n = rng.choice([1, 2])
    elif risk == "LOW":
        n = rng.choice([0, 1])
    else:  # no telemetry yet
        n = rng.choice([1, 2])

    t = TYPE_TABLE[ASSIGNMENT[int(aid.split("-")[1]) - 1]]
    statuses = []
    for _ in range(n):
        if risk in ("CRITICAL", "HIGH"):
            statuses.append(rng.choice(["OPEN", "OPEN", "MONITOR"]))
        elif risk == "MEDIUM":
            statuses.append(rng.choice(["OPEN", "MONITOR", "CLOSED"]))
        else:
            statuses.append(rng.choice(["CLOSED", "MONITOR"]))

    issues = []
    days_spread = rng.sample(range(3, 90), n) if n else []
    for k in range(n):
        code = t["codes"][k % len(t["codes"])]
        sev = (rng.choice(["MAJOR", "SERIOUS"]) if risk in ("CRITICAL", "HIGH")
               else rng.choice(["MINOR", "MAJOR"]))
        issues.append({
            "date": (TODAY - timedelta(days=sorted(days_spread, reverse=True)[k])).isoformat(),
            "severity": sev,
            "system": SYSTEM_OF[code],
            "code": code,
            "summary": SUMMARY_OF[code],
            "action": ACTION_BY_STATUS[statuses[k]],
            "status": statuses[k],
        })
    return issues


def build_reports_history(aid: str) -> dict:
    """Synthetic degradation trend leading up to the current seeded RUL."""
    current = CURRENT_RUL[aid]
    if current is None:
        return {"current": None,
                "note": "No edge reports ingested yet — first report will populate history.",
                "history": []}
    hist = []
    for days_ago, step in ((21, 22.0), (14, 16.0), (7, 10.0), (3, 5.0)):
        past = round(min(MAX_RUL, current + step), 2)
        hist.append({
            "days_ago": days_ago,
            "rul_cycles": past,
            "engine_health": round(max(0.0, min(100.0, past / MAX_RUL * 100.0)), 2),
            "risk": risk_band(past),
            "recommended_action": "synthetic trend point — monitor",
        })
    return {"current": current,
            "note": "Synthetic degradation trend; ingested by seed_demo.py as older "
                    "edge-style reports. Current report is seeded at run time.",
            "history": hist}


def build_profiles(fleet: list[dict]) -> list[dict]:
    rng = random.Random(2026)  # fixed seed -> byte-stable regeneration
    roster = {a["aircraft_id"]: a for a in fleet}
    aircraft = []
    for i, entry in enumerate(roster_ids()):
        aid = entry
        t_name = ASSIGNMENT[i]
        t = TYPE_TABLE[t_name]
        r = roster[aid]
        risk = risk_band(CURRENT_RUL[aid]) if CURRENT_RUL[aid] is not None else None
        lm_days_ago = 4 + (i % 26) * 3                  # 4..79 days ago (within 90)
        lm_date = TODAY - timedelta(days=lm_days_ago)
        aircraft.append({
            "aircraft_id": aid,
            "type": t_name,
            "model_code": t["model_code"],
            "role": t["role"],
            "origin": t["origin"],
            "engines": t["engines"],
            "image": t["image"],
            "base": r["base"],
            "mission_priority": r["mission_priority"],
            "spare_available": r["spare_available"],
            "operational_status": "ACTIVE",
            "last_maintenance": {
                "date": lm_date.isoformat(),
                "type": MAINT_TYPES[i % len(MAINT_TYPES)],
                "location": f"{r['base']} Synthetic Maintenance Hangar {(i % 3) + 1}",
                "cycles_since_maintenance": 35 + (i % 20) * 12 + (60 if risk == "CRITICAL" else 0),
                "next_due": (lm_date + timedelta(days=120)).isoformat(),
            },
            "issue_history": build_issue_history(rng, aid, risk),
            "reports": build_reports_history(aid),
        })
    return aircraft


def write_fleet_config(fleet: list[dict]) -> None:
    out = {
        "description": FLEET_CONFIG_DESCRIPTION,
        "aircraft_schema": AIRCRAFT_SCHEMA,
        "count": len(fleet),
        "fleet": fleet,
    }
    with open(FLEET_CONFIG, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[roster] wrote {FLEET_CONFIG}: {len(fleet)} aircraft")


def write_profiles(aircraft: list[dict]) -> None:
    out = {
        "description": "AARIS synthetic fleet profiles for the SIH demo. ALL aircraft "
                       "identities, maintenance dates, issue logs and report histories in "
                       "this file are SYNTHETIC demonstration data. No real Indian Air "
                       "Force maintenance records, tail numbers, sortie data, defect "
                       "reports, classified or operational information is used, scraped, "
                       "inferred, or implied.",
        "data_provenance": {
            "rul_model": "NASA C-MAPSS (public run-to-failure dataset, FD001) — see docs/RESULTS.md",
            "fleet_profiles": "synthetic, generated by ground/make_profiles.py (fixed seed 2026)",
            "real_iaf_data": "none",
            "images": "Per-type illustrative reference photos (dashboard/assets/plane_types/). "
                      "They depict the aircraft TYPE only — not the specific fictional airframe — "
                      "and carry no association to any real tail number, base or record.",
        },
        "generated_on": TODAY.isoformat(),
        "per_model_count": PER_MODEL_COUNT,
        "model_count": len(MODEL_BLOCKS),
        "count": len(aircraft),
        "aircraft": aircraft,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[profiles] wrote {OUT}: {len(aircraft)} synthetic aircraft")


def self_check(fleet: list[dict], aircraft: list[dict]) -> None:
    assert len(fleet) == len(aircraft) == len(MODEL_BLOCKS) * PER_MODEL_COUNT == 90
    assert [a["aircraft_id"] for a in fleet] == [a["aircraft_id"] for a in aircraft]

    # model blocks: 15 contiguous airframes per model
    by_type: dict[str, int] = {}
    for a in aircraft:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    assert set(by_type) == set(MODEL_BLOCKS) and all(
        n == PER_MODEL_COUNT for n in by_type.values()), by_type
    for b, name in enumerate(MODEL_BLOCKS):
        block = aircraft[b * PER_MODEL_COUNT:(b + 1) * PER_MODEL_COUNT]
        assert {a["type"] for a in block} == {name}, (name, b)

    # the verified edge fixture must stay untouched
    ac17 = aircraft[16]
    assert ac17["aircraft_id"] == "AC-017" and ac17["type"] == "Sukhoi Su-30MKI"
    assert ac17["model_code"] == "SU30MKI" and ac17["image"] == "su30mki.jpg"
    assert CURRENT_RUL["AC-017"] == 112.76
    assert ac17["reports"]["current"] == 112.76
    assert ac17["base"] == "BBI" and ac17["mission_priority"] == 2
    assert ac17["spare_available"] is True

    # every model shows the same documented health shape
    for b, name in enumerate(MODEL_BLOCKS):
        block = aircraft[b * PER_MODEL_COUNT:(b + 1) * PER_MODEL_COUNT]
        bands = [risk_band(a["reports"]["current"]) if a["reports"]["current"] is not None
                 else None for a in block]
        assert bands.count("CRITICAL") == 3 and bands.count("HIGH") == 3, (name, bands)
        assert bands.count("MEDIUM") == 3 and bands.count("LOW") == 4, (name, bands)
        assert bands.count(None) == 2, (name, bands)
        # a no-spare airframe in the tight slot must still reach day 5 (spare lead)
        for a, band in zip(block, bands):
            if band == "CRITICAL" and not a["spare_available"]:
                assert max(1, min(30, int(a["reports"]["current"] * 0.8))) >= 5, a["aircraft_id"]

    # exactly 30 no-spare aircraft, 3 per model block
    assert sum(1 for a in aircraft if not a["spare_available"]) == 30

    statuses = {iss["status"] for a in aircraft for iss in a["issue_history"]}
    assert {"OPEN", "MONITOR", "CLOSED"} <= statuses, statuses
    reporting = [a for a in aircraft if a["reports"]["current"] is not None]
    assert len(reporting) == 78 and all(len(a["reports"]["history"]) == 4 for a in reporting)

    img_dir = os.path.join(HERE, "..", "dashboard", "assets", "plane_types")
    for t_name, t in TYPE_TABLE.items():
        p = os.path.join(img_dir, t["image"])
        assert os.path.exists(p), f"missing type image for {t_name}: {p}"

    print(f"[profiles] all {len(TYPE_TABLE)} type reference images present")
    print(f"[profiles] model mix: {json.dumps(by_type)}")
    fleet_bands = {}
    for a in aircraft:
        cur = a["reports"]["current"]
        key = risk_band(cur) if cur is not None else "NO DATA"
        fleet_bands[key] = fleet_bands.get(key, 0) + 1
    print(f"[profiles] fleet bands: {json.dumps(fleet_bands)}")
    print(f"[profiles] issue statuses present: {sorted(statuses)}")
    print(f"[profiles] AC-017 = Su-30MKI, RUL {CURRENT_RUL['AC-017']} (edge fixture intact)")


def main() -> None:
    fleet = build_roster()
    write_fleet_config(fleet)
    aircraft = build_profiles(fleet)
    write_profiles(aircraft)
    self_check(fleet, aircraft)


if __name__ == "__main__":
    main()
