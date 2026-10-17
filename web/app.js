/* AARIS — interactive demo site.
 *
 * Renders the frozen snapshot in ./data.json (captured read-only from the local
 * AARIS ground station by dashboard/capture_demo_data.py). There is no backend:
 * every number on the page is derived in this file from that one snapshot, using
 * the same rules the Streamlit dashboard applies — risk bands, the
 * cross-conformal 90% interval, and the plan-date-derived Fleet Availability KPI
 * (dashboard/app.py:compute_readiness).
 *
 * The conformal half-widths and model-evidence figures below are copied from
 * conformal/risk_calibration.json and RESULTS.md §8 (audited NASA scoring); they
 * are constants of the frozen Ridge artifact, not live measurements.
 */
(() => {
  "use strict";

  // ------------------------------------------------------------- constants
  const MAX_RUL = 125;
  const SPARE_LEAD_DAYS = 5;
  const CAL = {
    halfwidth: 27.19,
    levels: { 80: 21.81, 90: 27.19, 95: 31.32 },
    coverage: { 80: 0.8001, 90: 0.9001, 95: 0.9501 },
    oofRmse: 16.41,
    nWindows: 17731,
    nEngines: 100,
    bands: {
      LOW: 0.9676, MEDIUM: 0.7634, HIGH: 0.9751, CRITICAL: 1.0,
    },
    groundingThreshold: 30,
  };

  const EVIDENCE = [
    { model: "Ridge regression on flattened 30-cycle windows", rmse: "16.22", mae: "13.21", nasa: "413", role: "Deployed at the edge" },
    { model: "1-layer LSTM (h=64, 30 epochs)", rmse: "17.65", mae: "12.54", nasa: "425", role: "Sequence baseline" },
    { model: "Transformer (3 seeds, mean ± std)", rmse: "18.46 ± 0.76", mae: "17.02 ± 0.84", nasa: "544 – 800", role: "Researched, not deployed" },
    { model: "Constant-38 predictor", rmse: "55.99", mae: "47.76", nasa: "257,838", role: "Degenerate floor" },
  ];

  const RISK_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, NONE: 4 };
  const RISK_COLOR = { LOW: "#3fb950", MEDIUM: "#d4a017", HIGH: "#f0883e", CRITICAL: "#f85149", NONE: "#5b6673" };

  // ---------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s === null || s === undefined ? "" : s)
    .replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const f1 = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(1));
  const f2 = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(2));
  const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");
  const DAY_MS = 86400000;

  const parseUtc = (s) => (s ? new Date(String(s).replace(" ", "T") + (/[zZ]$/.test(s) ? "" : "Z")) : null);
  const fmtDay = (d) => (d ? d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", timeZone: "UTC" }) : "—");
  const fmtStamp = (s) => {
    const d = parseUtc(s);
    if (!d) return "—";
    return d.toLocaleString("en-GB", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";
  };

  const band = (rul) => {
    if (rul === null || rul === undefined) return "NONE";
    if (rul >= 100) return "LOW";
    if (rul >= 60) return "MEDIUM";
    if (rul >= 30) return "HIGH";
    return "CRITICAL";
  };
  const riskOf = (a) => a.risk || band(a.rul_cycles) || "NONE";
  const chip = (r) => `<span class="chip ${r}">${r}</span>`;
  const interval = (rul) => {
    if (rul === null || rul === undefined) return null;
    return [Math.max(0, rul - CAL.halfwidth), Math.min(MAX_RUL, rul + CAL.halfwidth)];
  };

  // ----------------------------------------------------------------- state
  const state = {
    section: "readiness",
    planDay: 1,
    search: "",
    riskFilter: "ALL",
    baseFilter: "ALL",
    modelFilter: "ALL",
    sort: { key: "risk", dir: 1 },
    selected: "AC-017",
    planView: "SCHEDULED",
  };

  let D = null;   // snapshot
  let FLEET = []; // fleet rows, enriched
  let BY_ID = {}; // aircraft_id -> {aircraft, history, profile}
  let PLAN_BY_ID = {}; // aircraft_id -> plan task row

  // ------------------------------------------------------- derived metrics
  function planDaySet(day) {
    return new Set(D.schedule.plan.tasks
      .filter((t) => t.status === "SCHEDULED" && t.scheduled_day === day)
      .map((t) => t.aircraft_id));
  }

  /** Mirrors dashboard/app.py:compute_readiness for a chosen plan day. */
  function readiness(day) {
    const total = FLEET.length;
    const critical = new Set(FLEET.filter((a) => a.risk === "CRITICAL").map((a) => a.aircraft_id));
    const scheduled = planDaySet(day);
    const available = FLEET.filter((a) => !critical.has(a.aircraft_id) && !scheduled.has(a.aircraft_id));
    return {
      total,
      critical: critical.size,
      scheduled,
      available: available.length,
      rate: total ? available.length / total : null,
      atRisk: FLEET.filter((a) => a.risk === "CRITICAL" || a.risk === "HIGH").length,
      unreported: FLEET.filter((a) => a.rul_cycles === null || a.rul_cycles === undefined).length,
    };
  }

  const planDate = (day) => new Date(D._planBase.getTime() + (day - 1) * DAY_MS);
  const maxPlanDay = () => Math.max(...D.schedule.plan.tasks
    .filter((t) => t.status === "SCHEDULED").map((t) => t.scheduled_day || 0));

  // ------------------------------------------------------------- rendering
  function render() {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("is-active", b.dataset.section === state.section));
    ["readiness", "models", "detail", "plan", "about"].forEach((s) => {
      $("#s-" + s).hidden = s !== state.section;
    });
    ({ readiness: renderReadiness, models: renderModels, detail: renderDetail, plan: renderPlan, about: renderAbout })[state.section]();
  }

  // ---------------------------------------------------- 1. fleet readiness
  function renderReadiness() {
    const r = readiness(state.planDay);
    const mix = { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0, NONE: 0 };
    FLEET.forEach((a) => { mix[riskOf(a)]++; });
    const totalReports = Object.values(BY_ID).reduce((n, a) => n + (a.report_count || 0), 0);
    const alerts = D.alerts.alerts;
    const criticals = FLEET.filter((a) => a.risk === "CRITICAL")
      .sort((x, y) => x.rul_cycles - y.rul_cycles);

    const seg = (k) => mix[k] ? `<i style="flex:${mix[k]};background:${RISK_COLOR[k]}" title="${k}: ${mix[k]}"></i>` : "";

    $("#s-readiness").innerHTML = `
      <h1>Fleet Readiness Decision Support</h1>
      <p class="lede">Five questions this system answers for a base: how many aircraft can fly today,
      which ones are degrading, why, what the maintenance schedule looks like under real constraints,
      and what each decision is grounded in.</p>

      <div class="grid kpigrid" style="margin-top:1rem">
        <div class="card">
          <h3>Fleet Availability <span class="note">plan day ${state.planDay} · ${fmtDay(planDate(state.planDay))}</span></h3>
          <div class="kpi">
            <div class="big">${pct(r.rate)}<small>${r.available} / ${r.total} aircraft</small></div>
            <div class="spacer"></div>
            <div class="metric"><span class="v">${r.scheduled.size}</span><span class="k">in hangar</span></div>
            <div class="metric"><span class="v">${r.critical}</span><span class="k">critical</span></div>
            <div class="metric"><span class="v">${r.atRisk}</span><span class="k">critical + high</span></div>
          </div>
          <p class="sub" style="margin-top:0.7rem">
            Aircraft that are <b>not CRITICAL</b> and <b>not booked into a hangar</b> on the selected plan day.
            ${r.unreported} of ${r.total} have no telemetry in this snapshot and are counted as not-CRITICAL —
            that assumption is surfaced, not hidden.
          </p>
          <label class="field" style="margin-top:0.9rem">
            Age the plan <span id="dayLabel">day ${state.planDay} · ${fmtDay(planDate(state.planDay))}</span>
            <input type="range" id="daySlider" min="1" max="${Math.max(14, maxPlanDay() + 7)}" value="${state.planDay}">
          </label>
          <p class="note">${r.scheduled.size === 0
            ? `No maintenance is booked on day ${state.planDay}, so availability returns to the not-CRITICAL count.`
            : `Day ${state.planDay} books ${r.scheduled.size} aircraft: ${[...r.scheduled].sort().join(", ")}.`}
            The KPI is derived from the plan date, so it moves as the plan ages: a freshly generated plan reads
            <b>80.0% (72/90)</b>, the same plan six days later reads 71.1%.</p>
        </div>

        <div class="card">
          <h3>Risk distribution <span class="note">latest report per aircraft</span></h3>
          <div class="gauge">${["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"].map(seg).join("")}</div>
          <div class="axis">${["CRITICAL", "HIGH", "MEDIUM", "LOW", "NO REPORT"].map((k, i) => {
            const key = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"][i];
            return `<span style="color:${RISK_COLOR[key]}">${mix[key]} ${k}</span>`;
          }).join("")}</div>
          <div class="grid g4" style="margin-top:1rem;gap:0.5rem">
            <div class="metric"><span class="v">${r.total}</span><span class="k">aircraft</span></div>
            <div class="metric"><span class="v">${totalReports}</span><span class="k">reports</span></div>
            <div class="metric"><span class="v">${r.total - r.unreported}</span><span class="k">reporting</span></div>
            <div class="metric"><span class="v">${alerts.length}</span><span class="k">active alerts</span></div>
          </div>
          <p class="note" style="margin-top:0.7rem">Bands are point estimates from the frozen Ridge model:
          LOW ≥ 100, MEDIUM 60–99, HIGH 30–59, CRITICAL &lt; 30 predicted cycles remaining.</p>
        </div>
      </div>

      <h2>Ground now — ${criticals.length} aircraft below 30 cycles <span class="hint">sorted by remaining useful life</span></h2>
      ${fleetTable(criticals, { compact: true, cols: ["aircraft", "type", "base", "rul", "interval", "health", "spare", "plan"] })}

      <h2>Active CRITICAL alerts <span class="hint">raised automatically on ingest, cleared on a strictly newer healthy report</span></h2>
      <div class="grid g2">
        ${alertsCard(alerts)}
        ${attentionCard(criticals, r)}
      </div>

      <h2>All ${r.total} aircraft <span class="hint">click any row for the full record</span></h2>
      <div class="row" style="margin-bottom:0.6rem">
        <input type="search" id="fleetSearch" placeholder="Search AC-### or type…" value="${esc(state.search)}" style="min-width:200px">
        <select id="riskFilter">${["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"].map((v) => `<option ${state.riskFilter === v ? "selected" : ""}>${v}</option>`).join("")}</select>
        <select id="baseFilter">${["ALL", ...bases()].map((v) => `<option ${state.baseFilter === v ? "selected" : ""}>${v === "ALL" ? "all bases" : esc(v)}</option>`).join("")}</select>
        <select id="modelFilter">${["ALL", ...D.models.models.map((m) => m.model_code)].map((v) => `<option ${state.modelFilter === v ? "selected" : ""}>${v === "ALL" ? "all types" : esc(v)}</option>`).join("")}</select>
        <span class="note" id="fleetCount"></span>
      </div>
      <div id="fleetTable"></div>
    `;

    wireReadiness();
    renderFleetTable();
  }

  function alertsCard(alerts) {
    return `<div class="card">
      <h3>Alert feed <span class="note">${alerts.length} ACTIVE</span></h3>
      <div class="tablewrap" style="max-height:330px;overflow-y:auto">
        <table><thead><tr><th>Aircraft</th><th class="num">RUL</th><th class="num">Health</th><th>Raised</th></tr></thead>
        <tbody>${alerts.map((a) => `<tr class="clickable" data-goto="${esc(a.aircraft_id)}">
          <td><b>${esc(a.aircraft_id)}</b></td>
          <td class="num">${f1(a.rul_cycles)}</td>
          <td class="num">${f1(a.engine_health)}%</td>
          <td class="nowrap note">${fmtStamp(a.created_at)}</td>
        </tr>`).join("")}</tbody></table>
      </div>
      <p class="note">The station keeps exactly one ACTIVE alert per aircraft and refreshes it instead of
      stacking duplicates; recovery on a newer report moves it to CLEARED.</p>
    </div>`;
  }

  function attentionCard(criticals, r) {
    const inHangarToday = criticals.filter((a) => r.scheduled.has(a.aircraft_id)).length;
    const noSpare = criticals.filter((a) => !a.spare_available).length;
    const soon = criticals.filter((a) => a._plan && a._plan.scheduled_day && a._plan.scheduled_day <= 3).length;
    return `<div class="card">
      <h3>What the schedule is doing about it</h3>
      <div class="grid g2" style="gap:0.6rem">
        <div class="metric"><span class="v">${inHangarToday}</span><span class="k">of them in a hangar on day ${state.planDay}</span></div>
        <div class="metric"><span class="v">${soon}</span><span class="k">booked within 3 days</span></div>
        <div class="metric"><span class="v">${noSpare}</span><span class="k">have no engine spare</span></div>
        <div class="metric"><span class="v">${D.schedule.plan.scheduled_count} / ${D.schedule.plan.deferred_count}</span><span class="k">scheduled vs deferred</span></div>
      </div>
      <p class="note" style="margin-top:0.8rem">Deadlines are <span class="mono">floor(RUL × 0.8)</span> cycles, capped to the
      ${D.schedule.plan.horizon_days}-day horizon, with at most ${D.schedule.plan.max_daily_maintenance}
      maintenance slots per day and a ${SPARE_LEAD_DAYS}-day lead on aircraft with no spare. CP-SAT solves it to optimality;
      deferrals always carry a written reason.</p>
      <p class="note">Read the whole plan in <button class="btn-link" data-tab="plan">Maintenance Plan</button>.</p>
    </div>`;
  }

  function bases() {
    return [...new Set(FLEET.map((a) => a.base).filter(Boolean))].sort();
  }

  const COLS = {
    aircraft: { label: "Aircraft", cell: (a) => `<b>${esc(a.aircraft_id)}</b>`, sort: (a) => a.aircraft_id },
    type: { label: "Type", cell: (a) => `<span class="note">${esc(a._type || "—")}</span>`, sort: (a) => a._type || "" },
    base: { label: "Base", cell: (a) => esc(a.base || "—"), sort: (a) => a.base || "" },
    rul: { label: "RUL (cycles)", num: true, cell: (a) => f1(a.rul_cycles), sort: (a) => a.rul_cycles },
    health: { label: "Health", num: true, cell: (a) => (a.engine_health === null || a.engine_health === undefined ? "—" : `${f1(a.engine_health)}%`), sort: (a) => a.engine_health },
    interval: { label: "90% interval", num: true, cell: (a) => {
      const iv = interval(a.rul_cycles);
      return iv ? `${f1(iv[0])} – ${f1(iv[1])}` : "—";
    }, sort: (a) => a.rul_cycles },
    risk: { label: "Risk", cell: (a) => chip(riskOf(a)), sort: (a) => RISK_ORDER[riskOf(a)] },
    spare: { label: "Spare", cell: (a) => (a.spare_available ? `<span class="note">yes</span>` : `<span class="warn note">none</span>`), sort: (a) => (a.spare_available ? 1 : 0) },
    plan: { label: "Plan", cell: (a) => (a._plan
      ? (a._plan.status === "SCHEDULED"
        ? `<span class="note">day <b>${a._plan.scheduled_day}</b> · due ${a._plan.maintenance_deadline_day}</span>`
        : `<span class="note">deferred</span>`) : `<span class="note">—</span>`), sort: (a) => a._plan?.scheduled_day },
    report: { label: "Last report", cell: (a) => `<span class="note">${fmtStamp(a.latest_report_at)}</span>`, sort: (a) => a.latest_report_at || "" },
  };

  function fleetTable(rows, opts = {}) {
    const cols = opts.cols || Object.keys(COLS);
    return `<div class="tablewrap"><table>
      <thead><tr>${cols.map((c) => `<th class="${COLS[c].num ? "num" : ""}" data-sort="${c}"${opts.compact ? "" : ' style="cursor:pointer"'}>${COLS[c].label}${!opts.compact && state.sort.key === c ? (state.sort.dir > 0 ? " ↑" : " ↓") : ""}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((a) => `<tr class="clickable" data-goto="${esc(a.aircraft_id)}">
        ${cols.map((c) => `<td class="${COLS[c].num ? "num" : ""}">${COLS[c].cell(a)}</td>`).join("")}
      </tr>`).join("")}</tbody>
    </table></div>${rows.length === 0 ? `<p class="note">No aircraft matches these filters.</p>` : ""}`;
  }

  function filteredFleet() {
    const q = state.search.trim().toLowerCase();
    let rows = FLEET.filter((a) => {
      if (state.riskFilter !== "ALL" && riskOf(a) !== state.riskFilter) return false;
      if (state.baseFilter !== "ALL" && a.base !== state.baseFilter) return false;
      if (state.modelFilter !== "ALL" && a._model !== state.modelFilter) return false;
      if (q && !(a.aircraft_id.toLowerCase().includes(q) || (a._type || "").toLowerCase().includes(q))) return false;
      return true;
    });
    const col = COLS[state.sort.key];
    const isNil = (v) => v === null || v === undefined;
    rows.sort((x, y) => {
      const a = col.sort(x), b = col.sort(y);
      // Aircraft with no telemetry sort last in either direction — they are a
      // data gap, not an extreme value.
      if (isNil(a) || isNil(b)) {
        if (isNil(a) && isNil(b)) return x.aircraft_id < y.aircraft_id ? -1 : 1;
        return isNil(a) ? 1 : -1;
      }
      if (a < b) return -state.sort.dir;
      if (a > b) return state.sort.dir;
      return x.aircraft_id < y.aircraft_id ? -1 : 1;
    });
    return rows;
  }

  function renderFleetTable() {
    const rows = filteredFleet();
    const el = $("#fleetTable");
    if (el) el.innerHTML = fleetTable(rows);
    const count = $("#fleetCount");
    if (count) count.textContent = `${rows.length} of ${FLEET.length} shown · sorted by ${COLS[state.sort.key].label}`;
    el?.querySelectorAll("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
      const key = th.dataset.sort;
      state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
      renderFleetTable();
    }));
    el?.querySelectorAll("tr[data-goto]").forEach((tr) => tr.addEventListener("click", () => selectAircraft(tr.dataset.goto, "detail")));
  }

  function wireReadiness() {
    $("#daySlider")?.addEventListener("input", (e) => { state.planDay = +e.target.value; renderReadiness(); });
    $("#fleetSearch")?.addEventListener("input", (e) => { state.search = e.target.value; renderFleetTable(); });
    $("#riskFilter")?.addEventListener("change", (e) => { state.riskFilter = e.target.value; renderFleetTable(); });
    $("#baseFilter")?.addEventListener("change", (e) => { state.baseFilter = e.target.value; renderFleetTable(); });
    $("#modelFilter")?.addEventListener("change", (e) => { state.modelFilter = e.target.value; renderFleetTable(); });
    $("#s-readiness").querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => go(b.dataset.tab)));
    $("#s-readiness").querySelectorAll("tr[data-goto]").forEach((tr) => tr.addEventListener("click", () => selectAircraft(tr.dataset.goto, "detail")));
  }

  // ------------------------------------------------------- 2. aircraft models
  function renderModels() {
    const ms = D.models.models;
    $("#s-models").innerHTML = `
      <h1>Aircraft Models</h1>
      <p class="lede">The ${D.models.aircraft_total}-aircraft demonstration fleet is ${D.models.count} synthetic
      types of ${ms[0].aircraft_count} airframes each. Aggregates come from <span class="mono">GET /models</span>
      on the ground station, which rolls up telemetry, plan state and open issues per type.</p>

      <div class="grid g3" style="margin-top:1.1rem">
        ${ms.map((m) => modelCard(m)).join("")}
      </div>

      <h2>Type comparison</h2>
      <div class="tablewrap"><table>
        <thead><tr><th>Type</th><th class="num">Airframes</th><th class="num">Reporting</th><th class="num">No telemetry</th>
        <th class="num">CRITICAL</th><th class="num">Avg RUL</th><th class="num">Min RUL</th><th class="num">Ready</th>
        <th class="num">No spare</th><th class="num">Open issues</th><th class="num">Scheduled</th><th class="num">Deferred</th></tr></thead>
        <tbody>${ms.map((m) => `<tr class="clickable" data-model="${esc(m.model_code)}">
          <td><b>${esc(m.type)}</b><br><span class="note">${esc(m.model_code)} · ${esc(m.role)}</span></td>
          <td class="num">${m.aircraft_count}</td><td class="num">${m.reporting_count}</td><td class="num">${m.unreported}</td>
          <td class="num" style="color:${RISK_COLOR.CRITICAL}">${m.critical}</td>
          <td class="num">${f1(m.avg_rul_cycles)}</td><td class="num">${f1(m.lowest_rul_cycles)}</td>
          <td class="num">${m.ready_count}</td><td class="num">${m.spares_missing}</td><td class="num">${m.open_issues}</td>
          <td class="num">${m.scheduled_in_plan}</td><td class="num">${m.deferred_in_plan}</td>
        </tr>`).join("")}</tbody>
      </table></div>
      <p class="note" style="margin-top:0.5rem">${esc(D.models.note)}</p>
    `;
    $("#s-models").querySelectorAll("[data-model]").forEach((tr) => tr.addEventListener("click", () => {
      state.modelFilter = tr.dataset.model; go("readiness");
    }));
    $("#s-models").querySelectorAll("[data-air]").forEach((b) => b.addEventListener("click", () => selectAircraft(b.dataset.air, "detail")));
  }

  function modelCard(m) {
    const mix = m.risk_mix || {};
    const seg = (k) => (mix[k] ? `<i style="flex:${mix[k]};background:${RISK_COLOR[k]}" title="${k} ${mix[k]}"></i>` : "");
    return `<article class="card modelcard">
      <img class="photo" src="assets/types/${esc(m.image)}" alt="Illustrative reference photo of the ${esc(m.type)} type" loading="lazy">
      <div class="body">
        <div>
          <div class="name">${esc(m.type)}</div>
          <div class="meta">${esc(m.model_code)} · ${esc(m.role)} · ${esc(m.origin)} · ${esc(m.engines)}</div>
        </div>
        <div>
          <div class="mixbar">${["CRITICAL", "HIGH", "MEDIUM", "LOW"].map(seg).join("")}</div>
          <div class="mixlegend">${["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((k) => `<span style="color:${RISK_COLOR[k]}">${mix[k] || 0} ${k}</span>`).join("")}<span class="note">${m.unreported} no report</span></div>
        </div>
        <div class="statline"><span>Airframes</span><b>${m.aircraft_count} (${m.reporting_count} reporting, ${m.unreported} silent)</b></div>
        <div class="statline"><span>Avg RUL / health</span><b>${f1(m.avg_rul_cycles)} cycles · ${f1(m.avg_engine_health)}%</b></div>
        <div class="statline"><span>Lowest airframe</span><b>${f1(m.lowest_rul_cycles)} cycles</b></div>
        <div class="statline"><span>Ready for tasking</span><b>${m.ready_count} · ${m.critical} critical</b></div>
        <div class="statline"><span>Plan</span><b>${m.scheduled_in_plan} scheduled / ${m.deferred_in_plan} deferred</b></div>
        <div class="statline"><span>Without engine spare</span><b>${m.spares_missing}</b></div>
        <div class="statline"><span>Open synthetic issues</span><b>${m.open_issues}</b></div>
        <div class="row" style="margin-top:0.3rem">
          <button class="btn" data-air="${esc(m.aircraft_ids[0])}">Open an airframe</button>
          <button class="btn" data-model-jump="${esc(m.model_code)}">Filter fleet</button>
        </div>
        <p class="note">Reference photo is an illustrative type image, not this specific fictional airframe.</p>
      </div>
    </article>`;
  }

  // ------------------------------------------------------- 3. aircraft detail
  function renderDetail() {
    const q = state.detailSearch || "";
    const list = FLEET.filter((a) => !q || a.aircraft_id.toLowerCase().includes(q.toLowerCase()) || (a._type || "").toLowerCase().includes(q.toLowerCase()));
    const cur = BY_ID[state.selected];
    $("#s-detail").innerHTML = `
      <h1>Aircraft Detail <span class="hint">one click on any row, anywhere, lands here</span></h1>
      <div class="split" style="margin-top:0.9rem">
        <div>
          <input type="search" id="detailSearch" placeholder="Filter ${FLEET.length} aircraft…" value="${esc(q)}" style="width:100%;margin-bottom:0.5rem">
          <div class="picker">
            ${list.map((a) => `<button data-air="${esc(a.aircraft_id)}" class="${a.aircraft_id === state.selected ? "is-on" : ""}">
              <span class="dot" style="background:${RISK_COLOR[riskOf(a)]}"></span>
              <span class="mono">${esc(a.aircraft_id)}</span>
              <span class="tail">${esc(a._model || "")}</span>
            </button>`).join("")}
          </div>
        </div>
        <div>${cur ? aircraftPanel(state.selected) : `<p class="note">Select an aircraft.</p>`}</div>
      </div>
    `;
    $("#detailSearch")?.addEventListener("input", (e) => {
      state.detailSearch = e.target.value; renderDetail();
      const el = $("#detailSearch"); el.focus(); el.setSelectionRange(el.value.length, el.value.length);
    });
    $("#s-detail").querySelectorAll(".picker button[data-air]").forEach((b) => b.addEventListener("click", () => selectAircraft(b.dataset.air, "detail")));
  }

  function aircraftPanel(id) {
    const rec = BY_ID[id];
    const a = rec.aircraft;
    const p = rec.profile || {};
    const r = riskOf(a);
    const iv = interval(a.rul_cycles);
    const planTask = PLAN_BY_ID[id];
    const hist = (rec.history || []).slice().reverse();
    const trend = hist.filter((h) => h.rul_cycles !== null && h.rul_cycles !== undefined);

    return `
      <div class="card">
        <div class="detail-head">
          <div>
            <h3 style="margin:0"><span class="mono" style="font-size:1.25rem">${esc(id)}</span> ${chip(r)}</h3>
            <p class="lede" style="margin-top:0.15rem">${esc(p.type || "Unprofiled airframe")} · ${esc(p.role || "")} · base ${esc(a.base || "—")} · mission priority ${esc(a.mission_priority ?? "—")} · ${a.spare_available ? "engine spare available" : `<span class="warn">no engine spare</span>`}</p>
            <dl class="facts">
              <dt>Predicted RUL</dt><dd><b>${f2(a.rul_cycles)} cycles</b> <span class="note">(engine health ${f2(a.engine_health)}%)</span></dd>
              <dt>90% interval</dt><dd>${iv ? `<b>${f1(iv[0])} – ${f1(iv[1])} cycles</b> <span class="note">±${CAL.halfwidth} cross-conformal half-width</span>` : "—"}</dd>
              <dt>Reports</dt><dd>${rec.report_count} ingested <span class="note">(latest ${fmtStamp(a.latest_report_at)})</span></dd>
              <dt>Status</dt><dd>${esc(p.operational_status || "—")}</dd>
              <dt>Engines</dt><dd>${esc(p.engines || "—")}</dd>
              <dt>Last maintenance</dt><dd>${esc(p.last_maintenance?.date || "—")} · ${esc(p.last_maintenance?.type || "")} · ${esc(p.last_maintenance?.location || "")}</dd>
              <dt>Next due</dt><dd>${esc(p.last_maintenance?.next_due || "—")} <span class="note">(${p.last_maintenance?.cycles_since_maintenance ?? "—"} cycles since)</span></dd>
            </dl>
          </div>
          <div class="photoBox">${p.image ? `<img src="assets/types/${esc(p.image)}" alt="Illustrative ${esc(p.type || "")} type reference photo" loading="lazy">
            <p class="note">Illustrative type image — not this specific airframe.</p>` : ""}</div>
        </div>

        ${a.rul_cycles !== null && a.rul_cycles !== undefined ? `
        <div style="margin-top:1rem">
          <h3 style="font-size:0.8rem;color:var(--muted)">Remaining useful life on the 0–125 cycle scale</h3>
          ${rulBar(a.rul_cycles, iv)}
        </div>` : `<p class="note" style="margin-top:0.8rem">No telemetry in this snapshot — the scheduler treats this airframe as unreported rather than guessing a health value.</p>`}
      </div>

      <div class="grid g2" style="margin-top:0.9rem">
        <div class="card">
          <h3>Maintenance plan <span class="note">${planTask ? planTask.status : "not in plan"}</span></h3>
          ${planTask ? `
            <p class="lede">${planTask.status === "SCHEDULED"
              ? `Hangar slot on <b>day ${planTask.scheduled_day}</b> (${fmtDay(planDate(planTask.scheduled_day))}); deadline day ${planTask.maintenance_deadline_day}.`
              : `Deferred — no slot inside the ${D.schedule.plan.horizon_days}-day horizon.`}</p>
            <div class="reason">${esc(planTask.reason)}</div>` : `<p class="note">This aircraft has no plan row.</p>`}
        </div>
        <div class="card">
          <h3>What is driving the prediction</h3>
          ${(a.top_issues || []).length ? `<div class="tablewrap"><table>
            <thead><tr><th>Sensor</th><th class="num">Impact (cycles)</th><th>Note</th></tr></thead>
            <tbody>${a.top_issues.map((t) => `<tr><td class="mono">${esc(t.sensor)}</td><td class="num" style="color:${t.impact_cycles < 0 ? RISK_COLOR.CRITICAL : RISK_COLOR.LOW}">${f1(t.impact_cycles)}</td><td class="note">${esc(t.note)}</td></tr>`).join("")}</tbody>
          </table></div>` : `<p class="note">No issue breakdown for this report.</p>`}
          <p class="note" style="margin-top:0.5rem">Recommended action from the edge node: <b>${esc(a.recommended_action || "—")}</b></p>
          <p class="note">The edge node ranks only sensors whose window-summed Ridge contribution is negative, up
          to five; a window with no downward driver lists nothing.</p>
        </div>
      </div>

      ${trend.length >= 2 ? `<div class="card" style="margin-top:0.9rem">
        <h3>RUL trend <span class="note">${trend.length} ingested points, oldest → newest</span></h3>
        ${sparkline(trend)}
        <div class="axis"><span>${fmtStamp(trend[0].timestamp)}</span><span>${fmtStamp(trend[trend.length - 1].timestamp)}</span></div>
        <p class="note">${esc(p.reports_note || "Synthetic degradation trend used to exercise the ingest path.")}</p>
      </div>` : ""}

      ${(p.issue_history || []).length ? `<div class="card" style="margin-top:0.9rem">
        <h3>Issue history <span class="note">${p.issue_history.length} synthetic entries</span></h3>
        <div class="tablewrap"><table>
          <thead><tr><th>Date</th><th>Severity</th><th>System</th><th>Code</th><th>Summary</th><th>Action</th><th>Status</th></tr></thead>
          <tbody>${p.issue_history.map((i) => `<tr>
            <td class="nowrap">${esc(i.date)}</td><td>${esc(i.severity)}</td><td>${esc(i.system)}</td>
            <td class="mono note">${esc(i.code)}</td><td>${esc(i.summary)}</td><td class="note">${esc(i.action)}</td>
            <td>${esc(i.status)}</td></tr>`).join("")}</tbody>
        </table></div>
      </div>` : ""}

      <p class="note" style="margin-top:0.7rem">${p.synthetic ? "Synthetic demonstration profile — fictional identity, dates and issue log. No real operational or maintenance data is used." : ""}</p>
    `;
  }

  function rulBar(rul, iv) {
    const x = (v) => (Math.max(0, Math.min(MAX_RUL, v)) / MAX_RUL) * 100;
    const marks = [["CRITICAL", 0, 30], ["HIGH", 30, 60], ["MEDIUM", 60, 100], ["LOW", 100, 125]];
    return `<div class="rulbar">
      ${marks.map(([k, lo, hi]) => `<span style="position:absolute;left:${x(lo)}%;width:${x(hi) - x(lo)}%;top:0;bottom:0;background:${RISK_COLOR[k]}14;border-right:1px solid #212b36"></span>`).join("")}
      ${iv ? `<span class="band" style="left:${x(iv[0])}%;width:${x(iv[1]) - x(iv[0])}%"></span>` : ""}
      <span class="pt" style="left:${x(rul)}%"></span>
      <span class="lbl">${f1(rul)} cycles · interval ${iv ? f1(iv[0]) + "–" + f1(iv[1]) : "—"}</span>
    </div>
    <div class="axis"><span>0</span><span>30</span><span>60</span><span>100</span><span>125</span></div>`;
  }

  function sparkline(points) {
    // Fixed 0-125 cycle axis on purpose: auto-scaling to the point range would
    // turn a 12-cycle drift over five reports into a visual cliff.
    const w = 720, h = 96, pad = 8;
    const vals = points.map((p) => p.rul_cycles);
    const X = (i) => pad + (i * (w - 2 * pad)) / Math.max(1, points.length - 1);
    const Y = (v) => h - pad - (Math.max(0, Math.min(MAX_RUL, v)) / MAX_RUL) * (h - 2 * pad);
    const line = points.map((p, i) => `${X(i).toFixed(1)},${Y(p.rul_cycles).toFixed(1)}`).join(" ");
    const area = `${pad},${h - pad} ${line} ${X(points.length - 1).toFixed(1)},${h - pad}`;
    const grid = [0, 30, 60, 100, 125].map((v) =>
      `<line x1="${pad}" x2="${w - pad}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="#1b242e"/>
       <text x="${w - pad}" y="${(Y(v) - 3).toFixed(1)}" fill="#5b6673" font-size="9" text-anchor="end">${v}</text>`).join("");
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="RUL trend from ${f1(vals[0])} to ${f1(vals[vals.length - 1])} cycles on a fixed 0 to 125 axis">
      ${grid}
      <polygon points="${area}" fill="rgba(88,166,255,0.10)"/>
      <polyline points="${line}" fill="none" stroke="#58a6ff" stroke-width="2"/>
      ${points.map((p, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(p.rul_cycles).toFixed(1)}" r="3" fill="${RISK_COLOR[band(p.rul_cycles)]}"/>`).join("")}
    </svg>`;
  }

  // -------------------------------------------------------- 4. plan
  function renderPlan() {
    const plan = D.schedule.plan;
    const sched = plan.tasks.filter((t) => t.status === "SCHEDULED").sort((a, b) => a.scheduled_day - b.scheduled_day || a.aircraft_id.localeCompare(b.aircraft_id));
    const defer = plan.tasks.filter((t) => t.status === "DEFERRED");
    const byDay = new Map();
    sched.forEach((t) => { if (!byDay.has(t.scheduled_day)) byDay.set(t.scheduled_day, []); byDay.get(t.scheduled_day).push(t); });
    const days = [...byDay.keys()].sort((a, b) => a - b);
    // Prefer the station's own daily_load rollup; fall back to counting tasks.
    const load = (plan.daily_load && Object.keys(plan.daily_load).length)
      ? Object.entries(plan.daily_load).map(([d, n]) => [+d, n]).sort((a, b) => a[0] - b[0])
      : days.map((d) => [d, byDay.get(d).length]);
    const maxLoad = Math.max(...load.map(([, n]) => n));
    const late = sched.filter((t) => t.scheduled_day > t.maintenance_deadline_day);

    $("#s-plan").innerHTML = `
      <h1>Maintenance Plan <span class="hint">CP-SAT, ${plan.horizon_days}-day horizon, ${plan.max_daily_maintenance} slots/day</span></h1>
      <p class="lede">Generated ${fmtStamp(plan.generated_at)} over all ${plan.tasks.length} aircraft.
      Solver status: <b>${esc(plan.solver_status || "—")}</b> — ${plan.scheduled_count} aircraft receive a hangar slot,
      ${plan.deferred_count} are deferred with a written reason. Deterministic run (single worker, fixed seed),
      so the same inputs reproduce the same plan byte for byte.</p>

      <div class="grid g4" style="margin-top:1rem">
        <div class="card tight metric"><span class="v">${plan.scheduled_count}</span><span class="k">scheduled</span></div>
        <div class="card tight metric"><span class="v">${plan.deferred_count}</span><span class="k">deferred</span></div>
        <div class="card tight metric"><span class="v">${maxLoad} / ${plan.max_daily_maintenance}</span><span class="k">peak daily load</span></div>
        <div class="card tight metric"><span class="v">${late.length}</span><span class="k">booked after deadline</span></div>
      </div>

      <h2>Hangar loading <span class="hint">days 1–${days[days.length - 1]} carry every booking</span></h2>
      <div class="card">
        <div class="loadbar">${load.map(([d, n]) =>
          `<div class="${n === plan.max_daily_maintenance ? "full" : ""}" style="height:${(n / plan.max_daily_maintenance) * 100}%" title="Day ${d}: ${n} bookings — click to see availability that day" data-day="${d}"><em>${d}</em></div>`).join("")}</div>
        <p class="note" style="margin-top:1.4rem">Capacity is the binding constraint: ${maxLoad} of ${plan.max_daily_maintenance}
        slots used on the busiest day, and no booking lands after day ${days[days.length - 1]}.</p>
      </div>

      <h2>Day by day <span class="hint">selected plan day ${state.planDay} is highlighted</span></h2>
      <div class="daygrid">
        ${days.map((d) => `<div class="day ${d === state.planDay ? "is-today" : ""}">
          <div class="head"><b>Day ${d}</b><span>${fmtDay(planDate(d))} · ${byDay.get(d).length}/${plan.max_daily_maintenance}</span></div>
          ${byDay.get(d).map((t) => `<div class="slot clickable" data-goto="${esc(t.aircraft_id)}" title="${esc(t.reason)}">
            <span class="dot" style="width:7px;height:7px;border-radius:50%;background:${RISK_COLOR[t.risk || "NONE"]}"></span>
            <span class="id mono">${esc(t.aircraft_id)}</span>
            <span class="note">due ${t.maintenance_deadline_day}</span>
            <span class="rul">${f1(t.rul_cycles)}</span>
          </div>`).join("")}
        </div>`).join("")}
      </div>

      <h2>${state.planView === "SCHEDULED" ? `Scheduled bookings (${sched.length})` : `Deferrals (${defer.length})`}
        <span class="hint">
          <button class="btn ${state.planView === "SCHEDULED" ? "is-on" : ""}" data-view="SCHEDULED">Scheduled</button>
          <button class="btn ${state.planView === "DEFERRED" ? "is-on" : ""}" data-view="DEFERRED">Deferred</button>
        </span>
      </h2>
      ${planViewTable(state.planView, state.planView === "SCHEDULED" ? sched : defer)}
    `;

    $("#s-plan").querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => { state.planView = b.dataset.view; renderPlan(); }));
    $("#s-plan").querySelectorAll("[data-goto]").forEach((el) => el.addEventListener("click", () => selectAircraft(el.dataset.goto, "detail")));
    $("#s-plan").querySelectorAll("[data-day]").forEach((el) => el.addEventListener("click", () => { state.planDay = +el.dataset.day; go("readiness"); }));
  }

  function planViewTable(kind, rows) {
    if (kind === "SCHEDULED") {
      return `<div class="tablewrap"><table>
        <thead><tr><th>Day</th><th>Aircraft</th><th>Risk</th><th class="num">RUL</th><th class="num">Deadline</th><th>Priority</th><th>Spare</th><th>Reason</th></tr></thead>
        <tbody>${rows.map((t) => `<tr class="clickable" data-goto="${esc(t.aircraft_id)}">
          <td><b>${t.scheduled_day}</b></td><td class="mono">${esc(t.aircraft_id)}</td><td>${chip(t.risk || "NONE")}</td>
          <td class="num">${f1(t.rul_cycles)}</td><td class="num">${t.maintenance_deadline_day}</td>
          <td class="num">${t.mission_priority}</td><td>${t.spare_available ? "yes" : `<span class="warn">none</span>`}</td>
          <td class="note">${esc(t.reason)}</td></tr>`).join("")}</tbody>
      </table></div>`;
    }
    const groups = {};
    rows.forEach((t) => {
      const k = t.rul_cycles === null || t.rul_cycles === undefined
        ? "No telemetry received — cannot be assessed"
        : (t.risk === "LOW" ? "Healthy beyond the horizon" : "Lower priority inside the horizon");
      (groups[k] ||= []).push(t);
    });
    return Object.entries(groups).map(([k, list]) => `
      <h3 style="margin-top:1.2rem">${esc(k)} <span class="note">(${list.length} aircraft)</span></h3>
      <div class="tablewrap"><table>
        <thead><tr><th>Aircraft</th><th>Risk</th><th class="num">RUL</th><th>Reason</th></tr></thead>
        <tbody>${list.map((t) => `<tr class="clickable" data-goto="${esc(t.aircraft_id)}">
          <td class="mono">${esc(t.aircraft_id)}</td><td>${chip(t.risk || "NONE")}</td>
          <td class="num">${f1(t.rul_cycles)}</td><td class="note">${esc(t.reason)}</td></tr>`).join("")}</tbody>
      </table></div>`).join("");
  }

  // -------------------------------------------------------- 5. about
  function renderAbout() {
    const totalReports = Object.values(BY_ID).reduce((n, a) => n + (a.report_count || 0), 0);
    $("#s-about").innerHTML = `
      <h1>About &amp; Evidence</h1>
      <p class="lede">AARIS predicts how many flight cycles each engine has left, turns that into a per-aircraft
      health report on the aircraft itself, ships the report to a base station, and schedules hangar slots under
      real constraints. This page is a static, shareable view of one captured demonstration run.</p>

      <div class="grid g3" style="margin-top:1rem">
        <div class="card tight metric"><span class="v">${D.fleet.count}</span><span class="k">aircraft in the snapshot</span></div>
        <div class="card tight metric"><span class="v">${totalReports}</span><span class="k">health reports ingested</span></div>
        <div class="card tight metric"><span class="v">${D.alerts.count}</span><span class="k">active critical alerts</span></div>
      </div>

      <h2>What runs where</h2>
      <div class="tablewrap"><table>
        <thead><tr><th>Part</th><th>Runs on</th><th>Responsibility</th><th>State</th></tr></thead>
        <tbody>
          <tr><td><b>Edge node</b><br><span class="note">FastAPI :8000</span></td><td>On the aircraft (Jetson-class ARM SBC)</td>
            <td>Frozen Ridge model + conformal sidecar turn a 30-cycle sensor window into a health report</td><td class="note">Read-only artifacts, ~6 KB of pickles</td></tr>
          <tr><td><b>Ground station</b><br><span class="note">FastAPI :8001</span></td><td>At the base</td>
            <td>Authenticated ingest, latest-state store, alert lifecycle, CP-SAT scheduler, read API</td><td class="note">SQLite, six tables</td></tr>
          <tr><td><b>Dashboard</b><br><span class="note">Streamlit :8501</span></td><td>Operator workstation</td>
            <td>Five sections behind a JWT login; degrades to a captured snapshot when the station is unreachable</td><td class="note">This page mirrors its DEMO MODE</td></tr>
        </tbody>
      </table></div>

      <h2>Architecture</h2>
      <div class="diagram"><pre>${esc(ARCH)}</pre></div>

      <h2>Why the simple model ships <span class="hint">measured, not assumed</span></h2>
      <p class="lede">All numbers below come from the same split, scaler, test windows and metric code on NASA
      C-MAPSS FD001 (100 test engines). NASA scores use the audited PHM08-standard penalty scales — the repo's
      original metric had the 13/10 scales swapped, which inflated every score; see RESULTS.md §8.</p>
      <div class="tablewrap"><table>
        <thead><tr><th>Model</th><th class="num">Test RMSE (cycles)</th><th class="num">MAE</th><th class="num">NASA score</th><th>Role here</th></tr></thead>
        <tbody>${EVIDENCE.map((e) => `<tr><td>${esc(e.model)}</td><td class="num"><b>${esc(e.rmse)}</b></td><td class="num">${esc(e.mae)}</td><td class="num">${esc(e.nasa)}</td><td class="note">${esc(e.role)}</td></tr>`).join("")}</tbody>
      </table></div>
      <p class="note" style="margin-top:0.5rem">The Transformer is the researched model, not the deployed one:
      it clearly beats the degenerate constant predictor (55.30 → 18.46 across three seeds) and matches the LSTM,
      but no head-to-head evidence supports "Transformer &gt; regression" on this dataset variant, so no such claim
      is made here. Ridge wins on RMSE and on artifact size, which is what an edge node needs.</p>

      <h2>Uncertainty, stated honestly</h2>
      <div class="grid g2">
        <div class="card">
          <h3>Cross-conformal interval</h3>
          <p class="lede">Engine-wise GroupKFold cross-fit (k=5, per-fold scaler and model refit) over
          ${CAL.nWindows.toLocaleString()} windows from ${CAL.nEngines} training engines; absolute out-of-fold
          residuals in the final clipped-cycles scale. Out-of-fold RMSE ${CAL.oofRmse} cycles, against a frozen
          test RMSE of 16.22 — the calibration scale tracks held-out error.</p>
          <div class="tablewrap"><table>
            <thead><tr><th>Nominal level</th><th class="num">Half-width (cycles)</th><th class="num">Empirical OOF coverage</th></tr></thead>
            <tbody>${[80, 90, 95].map((k) => `<tr><td>${k}%</td><td class="num">${CAL.levels[k]}</td><td class="num">${(CAL.coverage[k] * 100).toFixed(1)}%</td></tr>`).join("")}</tbody>
          </table></div>
        </div>
        <div class="card">
          <h3>Where that interval is thin <span class="chip MEDIUM">caveat</span></h3>
          <p class="lede">Half-widths are global, so per-band coverage is uneven:</p>
          <div class="tablewrap"><table>
            <thead><tr><th>Band</th><th class="num">90% coverage</th></tr></thead>
            <tbody>${Object.entries(CAL.bands).map(([k, v]) => `<tr><td>${chip(k)}</td><td class="num" style="color:${v < 0.9 ? RISK_COLOR.MEDIUM : "inherit"}">${(v * 100).toFixed(1)}%</td></tr>`).join("")}</tbody>
          </table></div>
          <p class="note warn" style="margin-top:0.5rem">The MEDIUM band under-covers (76.3% against a nominal 90%).
          Cross-conformal coverage is a heuristic that needs approximate exchangeability, not an exact
          finite-sample guarantee. Risk bands stay point estimates — the interval never re-labels an aircraft.</p>
        </div>
      </div>

      <h2>Limits of this page and of the system</h2>
      <ul class="clean">
        <li><b>This is a snapshot, not a live deployment.</b> No server sits behind this URL: the data was read once
          from a local ground station (<span class="mono">${esc(D.source)}</span>) at ${fmtStamp(D.captured_at_utc)} and frozen into
          <span class="mono">data.json</span>. Nothing updates, no report can be ingested, no plan can be re-solved.</li>
        <li><b>Every aircraft identity is synthetic.</b> Types, bases, tail-style IDs, maintenance dates and issue logs
          are generated deterministically for demonstration. No real air force maintenance records, tail numbers, sortie
          data, defect reports or classified information are used, scraped, inferred or implied. Type photos are
          illustrative images only, never presented as a specific airframe.</li>
        <li><b>The physics is NASA engine data, not airframes.</b> RUL comes from a model trained on C-MAPSS FD001
          turbofan degradation run-ups; mapping those cycles onto fighter airframes is a demonstration framing.</li>
        <li><b>Field hardware is not yet exercised.</b> The edge node is field-node-ready and verified on a development
          machine; it has not been run on real Jetson hardware.</li>
        <li><b>Single engine per airframe, single station.</b> Reports carry one engine stream, and the scheduler
          assumes one base's hangar capacity.</li>
      </ul>

      <h2>Security model, in one table</h2>
      <div class="tablewrap"><table>
        <thead><tr><th>Caller</th><th>Credential</th><th>May do</th></tr></thead>
        <tbody>
          <tr><td>Aircraft edge node</td><td class="note">per-aircraft bearer token, bound to one ID</td><td class="note">ingest its own report only; another aircraft's ID is rejected 403</td></tr>
          <tr><td>Maintenance officer</td><td class="note">JWT (role: maintenance_officer)</td><td class="note">read fleet and plans, trigger the scheduler, acknowledge alerts</td></tr>
          <tr><td>Commanding officer</td><td class="note">JWT (role: commander)</td><td class="note">read everything, acknowledge alerts, may not schedule</td></tr>
          <tr><td>Admin</td><td class="note">JWT (role: admin)</td><td class="note">all of the above plus roster changes</td></tr>
        </tbody>
      </table></div>
      <p class="note">Deny-by-default on every endpoint except <span class="mono">/health</span> and
      <span class="mono">/auth/login</span>. The audit table records logins, denials, ingestions and reads —
      never tokens, passwords or raw payloads.</p>

      <h2>The Streamlit original</h2>
      <p class="lede">The operator-facing application is a Streamlit dashboard with the same five sections; these are
      captures of it running against the live station that produced this snapshot.</p>
      <div class="grid g2">
        ${SHOTS.map(([file, cap]) => `<figure style="margin:0"><img class="shot" src="assets/screenshots/${file}" alt="${esc(cap)}" loading="lazy"><figcaption class="shotcap">${esc(cap)}</figcaption></figure>`).join("")}
      </div>

      <h2>Run the real thing</h2>
      <div class="diagram"><pre>${esc(RUN)}</pre></div>
      <p class="note">Requires Python 3.14 (the artifacts were pickled with scikit-learn 1.9.1 / numpy 2.5.3 —
      hard-pinned for edge compatibility), then <span class="mono">start.bat</span> brings up all three services.</p>
    `;
  }

  const SHOTS = [
    ["02_fleet_readiness.png", "Fleet Readiness — the availability KPI, risk mix and grounded list"],
    ["03_critical_alerts.png", "CRITICAL alert banner with per-alert acknowledge, above the section nav"],
    ["04_aircraft_models.png", "Aircraft Models — six synthetic types with per-type rollups"],
    ["06_aircraft_detail.png", "Aircraft Detail — profile, conformal interval, trend and plan row"],
    ["07_maintenance_plan.png", "Maintenance Plan — day-by-day hangar loading and deferral reasons"],
    ["08_about.png", "About — architecture, evidence and limitations"],
  ];

  const ARCH = `ON THE AIRCRAFT  (edge node, FastAPI :8000)
  30-cycle window, 21 sensors + 3 operating settings
        |
        +-- MinMaxScaler -> flatten 30x24 -> Ridge(alpha=10) -> clip [0,125]
        +-- cross-conformal sidecar: +/- 27.19 cycles at 90%
        v
  POST /predict  ->  { rul_cycles, engine_health, risk, uncertainty,
                       top_issues, recommended_action }
        |
        |  bearer token bound to this aircraft
        v
AT THE BASE  (ground station, FastAPI :8001, SQLite)
  POST /ingest_report   -> latest state + report history
        |
        +-- risk == CRITICAL  -> alerts table (one ACTIVE per aircraft,
        |                        refreshed, auto-CLEARED on recovery)
        +-- POST /schedule    -> CP-SAT: 30-day horizon, 8 slots/day,
        |                        deadline floor(RUL x 0.8), 5-day spare lead
        v
  GET /fleet  /models  /fleet/{id}  /alerts  /schedule/latest   (JWT RBAC)
        |
        v
OPERATOR  (Streamlit :8501)
  Fleet Readiness | Aircraft Models | Aircraft Detail | Maintenance Plan | About
  -> availability KPI, grounded list, type rollups, plan, deferral reasons
  -> falls back to a captured snapshot when the base is unreachable`;

  const RUN = `# clone, then create the environment
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt

# one command: edge :8000 + ground :8001 + dashboard :8501
start.bat

# seeded, self-contained demo (resets the station, schedules, verifies)
powershell -ExecutionPolicy Bypass -File demo\\run_demo.ps1 -Reset -NoBrowser
.venv\\Scripts\\python.exe demo\\verify_pipeline.py

# live synthetic feed from the edge node, then the dashboard at http://localhost:8501
start.bat --fleet-feed`;

  // ------------------------------------------------------------- navigation
  function go(section) {
    state.section = section;
    location.hash = section;
    render();
    window.scrollTo({ top: 0 });
  }
  function selectAircraft(id, section) {
    state.selected = id;
    go(section || "detail");
  }

  // ------------------------------------------------------------------ init
  async function boot() {
    const res = await fetch("data.json", { cache: "force-cache" });
    if (!res.ok) throw new Error("data.json " + res.status);
    D = await res.json();
    D._planBase = parseUtc(D.schedule.plan.generated_at);

    BY_ID = D.aircraft;
    D.schedule.plan.tasks.forEach((t) => { PLAN_BY_ID[t.aircraft_id] = t; });
    FLEET = D.fleet.fleet.map((a) => {
      const p = BY_ID[a.aircraft_id]?.profile || {};
      return Object.assign({}, a, { _type: p.type, _model: p.model_code, _plan: PLAN_BY_ID[a.aircraft_id] });
    });

    $("#snapshot-badge").innerHTML =
      `DEMO SNAPSHOT · <b>${D.fleet.count} aircraft</b> · captured <b>${fmtStamp(D.captured_at_utc)}</b> · no live backend`;
    $("#synthetic-banner").hidden = false;

    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => go(b.dataset.section)));
    document.addEventListener("click", (e) => {
      const jump = e.target.closest("[data-model-jump]");
      if (jump) { state.modelFilter = jump.dataset.modelJump; go("readiness"); }
    });

    const initial = location.hash.replace("#", "");
    if (["readiness", "models", "detail", "plan", "about"].includes(initial)) state.section = initial;
    window.addEventListener("hashchange", () => {
      const s = location.hash.replace("#", "");
      if (["readiness", "models", "detail", "plan", "about"].includes(s)) { state.section = s; render(); }
    });
    render();
  }

  boot().catch((err) => {
    document.getElementById("view").innerHTML =
      `<div class="card"><h1>Snapshot failed to load</h1><p class="lede">${esc(err.message)}</p></div>`;
  });
})();
