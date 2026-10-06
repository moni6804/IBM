const $ = (s, el = document) => el.querySelector(s);
const SHIFT_KEYS = ["morning", "evening", "night"];
let state = null;
let open = null; // {empId, day} for the shift popover

const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const dayShort = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
const pad = (n) => String(n % 24).padStart(2, "0");
const timeRange = (s) => `${pad(s.start)}:00–${pad(s.start + s.hours)}:00`;
const pct = (p) => `${Math.round(p * 100)}%`;
const lvl = (p) => (p >= 0.25 ? "high" : p >= 0.12 ? "medium" : "low");
const WD = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

async function api(path, method = "GET", body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, data };
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.id); toast.id = setTimeout(() => (t.hidden = true), 3200);
}

function isOff(emp, day, shift) {
  if (emp.time_off.includes(day)) return true;
  const wd = WD[(new Date(day + "T00:00:00").getDay() + 6) % 7];
  return emp.unavailable.includes(`${wd}:all`) || (shift && emp.unavailable.includes(`${wd}:${shift}`));
}

function load(data) { state = data; render(); if (open) showPopover(); }

function render() {
  const days = Object.keys(state.coverage);
  $("#weekLabel").textContent = `Ward 4 roster · ${dayShort(days[0])} – ${dayShort(days[6])}`;
  const pill = $("#aiPill");
  const live = state.ai.provider === "watsonx";
  pill.className = "pill" + (live ? " live" : "");
  pill.textContent = live ? `Granite · ${state.ai.model}` : "Offline demo mode";
  pill.title = live ? "Responses come from IBM watsonx.ai" : "Set WATSONX_API_KEY and WATSONX_PROJECT_ID in .env to use IBM Granite";
  renderGrid(days); renderViolations(); renderRiskTop();
}

function renderGrid(days) {
  const sev = {};
  state.violations.forEach((v) => {
    if (!v.employee_id) return;
    const k = `${v.employee_id}|${v.date}`;
    if (sev[k] !== "error") sev[k] = v.severity;
  });
  const byCell = {};
  state.assignments.forEach((a) => (byCell[`${a.employee_id}|${a.date}`] ||= []).push(a.shift));

  let html = `<thead><tr><th scope="col">Team member</th>${days.map((d) => `<th scope="col">${dayShort(d)}</th>`).join("")}<th scope="col" class="hrs">Hours</th></tr></thead><tbody>`;
  const roles = [...new Set(state.employees.map((e) => e.role))];
  for (const role of roles) {
    html += `<tr class="role-row"><th colspan="9" scope="colgroup">${esc(role)}s</th></tr>`;
    for (const e of state.employees.filter((x) => x.role === role)) {
      const h = state.hours[e.id] || 0;
      const pct = Math.min(100, (h / Math.max(e.max_hours, 1)) * 100);
      const cls = h > state.rules.hard_weekly_hours_cap ? "cap" : h > e.max_hours ? "over" : "";
      html += `<tr><td class="name"><b>${esc(e.name)}</b><small>${e.max_hours}h contract</small></td>`;
      for (const d of days) {
        const shifts = (byCell[`${e.id}|${d}`] || []).sort((a, b) => SHIFT_KEYS.indexOf(a) - SHIFT_KEYS.indexOf(b));
        const s = sev[`${e.id}|${d}`];
        const label = `${e.name}, ${dayShort(d)}: ${shifts.length ? shifts.join(" and ") : "no shift"}${s ? ", " + (s === "error" ? "breaks a rule" : "needs review") : ""}`;
        html += `<td><button class="cell${isOff(e, d) ? " off" : ""}${s === "error" ? " has-err" : ""}" data-emp="${e.id}" data-day="${d}" aria-label="${esc(label)}">`
          + shifts.map((k) => { const p = state.risk.assignments[`${e.id}|${d}|${k}`]; return `<span class="chip ${k}">${state.shifts[k].label}${p == null ? "" : `<em class="rk r-${lvl(p)}" title="Predicted call-out risk">${pct(p)}</em>`}</span>`; }).join("")
          + (s ? `<span class="badge ${s === "error" ? "err" : "warn"}" aria-hidden="true">${s === "error" ? "!" : "▲"}</span>` : "")
          + `</button></td>`;
      }
      html += `<td class="hrs"><span>${h}h</span><div class="bar"><i class="${cls}" style="width:${pct}%"></i></div></td></tr>`;
    }
  }
  html += `<tr class="role-row"><th colspan="9" scope="colgroup">Coverage (staffed / needed)</th></tr>`;
  for (const k of SHIFT_KEYS) {
    html += `<tr class="cov"><th scope="row">${state.shifts[k].label}</th>`;
    for (const d of days) {
      const roles = state.coverage[d][k];
      const have = Object.values(roles).reduce((n, r) => n + Math.min(r.have, r.need), 0);
      const need = Object.values(roles).reduce((n, r) => n + r.need, 0);
      const detail = Object.entries(roles).map(([r, c]) => `${r}: ${c.have}/${c.need}`).join(", ");
      html += `<td class="${have < need ? "short" : "ok"}" title="${esc(detail)}">${have}/${need}</td>`;
    }
    html += `<td></td></tr>`;
  }
  html += `<tr class="role-row"><th colspan="9" scope="colgroup">Chance of at least one call-out (ML forecast)</th></tr>`;
  for (const k of SHIFT_KEYS) {
    html += `<tr class="cov"><th scope="row">${state.shifts[k].label}</th>`;
    for (const d of days) {
      const s = state.risk.slots[`${d}|${k}`];
      html += s ? `<td class="rkcell r-${lvl(s.p_any)}" title="Expected call-outs: ${s.expected.toFixed(2)}">${pct(s.p_any)}</td>` : `<td class="ok">–</td>`;
    }
    html += `<td></td></tr>`;
  }
  $("#grid").innerHTML = html + "</tbody>";
}

function renderViolations() {
  const errs = state.violations.filter((v) => v.severity === "error");
  const warns = state.violations.filter((v) => v.severity === "warning");
  $("#counts").innerHTML = state.violations.length
    ? `${errs.length ? `<span class="e">${errs.length} errors</span>` : ""}${warns.length ? `<span class="w">${warns.length} warnings</span>` : ""}`
    : `<span class="good">All clear</span>`;
  $("#violations").innerHTML = state.violations.length
    ? state.violations.map((v) => `<li><i class="mark ${v.severity === "error" ? "err" : "warn"}" aria-hidden="true">${v.severity === "error" ? "!" : "▲"}</i>
        <span><span class="sr">${v.severity}: </span>${esc(v.message)}</span>
        <button class="btn small" data-fix="${esc(v.id)}">Suggest fix</button></li>`).join("")
    : `<li class="empty">No problems. Every shift is covered within the rules.</li>`;
}

/* ---------- popover ---------- */
function showPopover() {
  const emp = state.employees.find((e) => e.id === open.empId);
  const cell = document.querySelector(`.cell[data-emp="${open.empId}"][data-day="${open.day}"]`);
  if (!emp || !cell) return closePopover();
  const mine = state.assignments.filter((a) => a.employee_id === emp.id && a.date === open.day).map((a) => a.shift);
  const pop = $("#popover");
  pop.innerHTML = `<h3>${esc(emp.name)}</h3><p>${dayShort(open.day)}${isOff(emp, open.day) ? " · marked unavailable" : ""}</p>`
    + SHIFT_KEYS.map((k) => `<button class="opt ${k}" data-shift="${k}" aria-pressed="${mine.includes(k)}">
        <i class="dot ${k}"></i>${state.shifts[k].label}<small>${timeRange(state.shifts[k])}<em class="rk orisk" data-risk="${k}"></em></small></button>`).join("")
    + `<div id="conflict"></div>`;
  pop.hidden = false;
  const r = cell.getBoundingClientRect();
  const left = Math.min(window.scrollX + r.left, window.scrollX + window.innerWidth - 276);
  pop.style.left = Math.max(8, left) + "px";
  pop.style.top = window.scrollY + r.bottom + 6 + "px";
  const target = { ...open };
  api("/api/ml/what-if", "POST", { employee_id: target.empId, date: target.day }).then((res) => {
    if (!res.ok || !open || open.empId !== target.empId || open.day !== target.day) return;
    for (const [k, v] of Object.entries(res.data)) {
      const el = pop.querySelector(`[data-risk="${k}"]`);
      if (el) { el.className = `rk orisk r-${v.level}`; el.textContent = pct(v.p); el.title = "Predicted call-out risk if assigned"; }
    }
  });
}
function closePopover() { open = null; $("#popover").hidden = true; }

async function toggleShift(shift, force = false) {
  const slot = { employee_id: open.empId, date: open.day, shift };
  const has = state.assignments.some((a) => a.employee_id === slot.employee_id && a.date === slot.date && a.shift === shift);
  const res = has ? await api("/api/assignments", "DELETE", slot) : await api("/api/assignments", "POST", { ...slot, force });
  if (res.status === 409) {
    $("#conflict").innerHTML = `<div class="conflict"><strong>This breaks a rule:</strong><ul>${res.data.violations.map((v) => `<li>${esc(v.message)}</li>`).join("")}</ul>
      <button class="btn small" data-force="${shift}">Assign anyway</button></div>`;
    return;
  }
  if (!res.ok) return toast(res.data.error || "Something went wrong.");
  load(res.data);
}

/* ---------- ML panel ---------- */
let mlInfo = null;
function renderModel() {
  if (!mlInfo) return;
  const m = mlInfo.metrics, d = mlInfo.data;
  $("#mlSummary").textContent = `${mlInfo.selected} was selected from 3 models. Trained on ${d.train.toLocaleString()} shifts, tested on ${d.test.toLocaleString()} unseen shifts (${pct(d.call_out_rate)} call-out rate).`;
  $("#mlMetrics").innerHTML = [["ROC-AUC", m.roc_auc.toFixed(2)], ["Recall", pct(m.recall)], ["Precision", pct(m.precision)], ["Accuracy", pct(m.accuracy)]]
    .map(([n, v]) => `<div><b>${v}</b><span>${n}</span></div>`).join("");
  const top = Math.max(...mlInfo.importance.map((x) => x.importance), 0.001);
  $("#mlImportance").innerHTML = mlInfo.importance.slice(0, 6).map((x) =>
    `<li><span>${esc(x.label)}</span><div class="bar"><i style="width:${(x.importance / top) * 100}%"></i></div><em>${x.importance.toFixed(3)}</em></li>`).join("");
  $("#mlCompare").innerHTML = mlInfo.comparison.map((c) =>
    `<li class="${c.name === mlInfo.selected ? "sel" : ""}"><span>${esc(c.name)}${c.name === mlInfo.selected ? " (selected)" : ""}</span><span>${c.cv_auc.toFixed(3)} ± ${c.cv_std.toFixed(3)}</span></li>`).join("");
}
function renderRiskTop() {
  const t = state.risk.top;
  $("#mlTop").innerHTML = t.length
    ? t.map((x) => `<li><span>${dayShort(x.date)} · ${state.shifts[x.shift].label}</span><em class="rk r-${x.level}">${pct(x.p_any)}</em></li>`).join("")
    : `<li class="note">Schedule some shifts to see forecasts.</li>`;
}
async function loadModel() { const r = await api("/api/ml/model"); if (r.ok) { mlInfo = r.data; renderModel(); } }

/* ---------- AI ---------- */
function aiShow(html) { $("#aiOut").innerHTML = html; }
const srcNote = (r) => `<span class="src">${r.source === "granite" ? "Answered by IBM Granite" : "Built-in summary" + (r.note ? " (" + esc(r.note) + ")" : "")}</span>`;

async function suggest(id, btn) {
  btn.disabled = true;
  const res = await api("/api/ai/suggest", "POST", { violation_id: id });
  btn.disabled = false;
  if (!res.ok) return toast(res.data.error || "Could not suggest a fix.");
  const { choice, candidates, add, remove, violation } = res.data;
  if (!choice.employee_id) return aiShow(`<div class="fix"><h3>No safe replacement</h3>${esc(choice.reason)}</div>`);
  const pick = candidates.find((c) => c.employee_id === choice.employee_id);
  aiShow(`<div class="fix"><h3>${esc(violation.message)}</h3>
    <strong>${esc(pick.name)}</strong> can take ${esc(state.shifts[add.shift].label.toLowerCase())} on ${dayShort(add.date)}. ${esc(choice.reason)}
    <div class="row" style="margin:8px 0 0"><button class="btn primary small" id="applyFix">Apply this fix</button></div>
    ${candidates.length > 1 ? `<ul>${candidates.filter((c) => c !== pick).map((c) => `<li>${esc(c.name)} would be at ${c.weekly_hours_after}h</li>`).join("")}</ul>` : ""}
    ${srcNote(choice)}</div>`);
  $("#applyFix").onclick = async () => {
    const r = await api("/api/assignments/swap", "POST", { add: { ...add, employee_id: choice.employee_id }, remove });
    if (r.ok) { load(r.data); aiShow(""); toast("Fix applied."); } else toast(r.data.error || "Could not apply.");
  };
}

/* ---------- events ---------- */
document.addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (cell) { open = { empId: cell.dataset.emp, day: cell.dataset.day }; return showPopover(); }
  const opt = e.target.closest(".opt");
  if (opt) return toggleShift(opt.dataset.shift);
  const force = e.target.closest("[data-force]");
  if (force) return toggleShift(force.dataset.force, true);
  const fix = e.target.closest("[data-fix]");
  if (fix) return suggest(fix.dataset.fix, fix);
  if (!e.target.closest("#popover")) closePopover();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopover(); });

async function run(btn, fn) { btn.disabled = true; try { await fn(); } finally { btn.disabled = false; } }

$("#btnAuto").onclick = (e) => run(e.target, async () => {
  const r = await api("/api/schedule/auto", "POST"); load(r.data);
  toast(r.data.unfilled.length ? `Scheduled, but ${r.data.unfilled.length} slot(s) couldn't be filled safely.` : "Week scheduled with no rule problems.");
});
$("#btnClear").onclick = async () => load((await api("/api/schedule/clear", "POST")).data);
$("#btnReset").onclick = async () => { closePopover(); load((await api("/api/reset", "POST")).data); toast("Demo data reset."); };
$("#btnRetrain").onclick = (e) => run(e.target, async () => {
  $("#mlSummary").textContent = "Training and comparing models…";
  const r = await api("/api/ml/retrain", "POST", { seed: Math.floor(Math.random() * 1000) });
  if (r.ok) { mlInfo = r.data.info; renderModel(); load(r.data.state); toast("Model retrained on fresh synthetic data."); }
});
$("#btnExplain").onclick = (e) => run(e.target, async () => {
  aiShow("Thinking…"); const r = await api("/api/ai/explain", "POST"); aiShow(`${esc(r.data.text)}${srcNote(r.data)}`);
});
$("#askForm").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#askInput").value.trim(); if (!q) return;
  aiShow("Thinking…"); const r = await api("/api/ai/ask", "POST", { question: q });
  aiShow(r.ok ? `${esc(r.data.text)}${srcNote(r.data)}` : esc(r.data.error));
};

api("/api/state").then((r) => load(r.data));
loadModel();
