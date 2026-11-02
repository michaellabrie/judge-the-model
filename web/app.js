// Starter Node app: three plain sections that read results/ and show correct numbers.
//
// Run locally:  bash run_app.sh   (your agent writes it; it runs node web/server.mjs, then open http://localhost:8766/)
// Build each section up to its required minimum (see the activity website). Keep every number
// coming from m11.js, and label every table and plot "Training" or "Test".
import * as m11 from "./m11.js";

const TRAIN = "Training";
const TEST = "Test";
const SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const int = (n) => n.toLocaleString("en-US");
const state = { section: SECTIONS[0], key: "readmission", model: "logreg" };
const cache = {};

async function load(name) {
  if (!(name in cache)) {
    // ../results/ is next to web/ on GitHub Pages; web/server.mjs serves it at the same address.
    const r = await fetch(`../results/${name}.json`);
    cache[name] = r.ok ? await r.json() : null;
  }
  return cache[name];
}

const caption = (text) => `<p class="caption">${esc(text)}</p>`;
const starterNote = (text) => `<p class="info">Starter display: ${esc(text)} Build this section up to its required minimum.</p>`;

// rows: array of objects; the first column is the row header.
function table(rows) {
  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th scope="col">${esc(c)}</th>`).join("");
  const body = rows.map((r) => "<tr>" + cols.map((c, i) => i === 0 ? `<th scope="row">${esc(r[c])}</th>`
    : `<td class="num">${esc(r[c])}</td>`).join("") + "</tr>").join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

async function recommendations() {
  const recs = (await load("decisions"))?.recommendations || {};
  const any = Object.values(recs).some(Boolean);
  $("recommendations").insertAdjacentHTML("beforeend", any
    ? m11.DATASETS.map((k) => `<p><strong>${esc(m11.DATASET_LABEL[k])}:</strong> ${recs[k] ? esc(recs[k]) : "<em>not written yet</em>"}</p>`).join("")
    : caption("No recommendations yet: fill in Recommendations in DECISIONS.md, run python -m analysis, then commit and push."));
}

// ---- 1. Beyond Accuracy ------------------------------------------------------
function beyondAccuracy(res, model) {
  const [p, y] = m11.preds(res, model);
  const c = m11.confusion(p, y, 0.5);
  return starterNote("the four counts at threshold 0.50, and accuracy.")
    + caption(`${TEST} · ${m11.MODEL_LABEL[model]}, threshold 0.50`)
    + table([{ TP: int(c.tp), FN: int(c.fn), FP: int(c.fp), TN: int(c.tn), Accuracy: m11.fmtPct(m11.metrics(c).accuracy) }]);
}

// ---- 2. A Threshold for the Job ------------------------------------------------
function threshold(res, model) {
  const [p, y] = m11.preds(res, model);
  const m = m11.metrics(m11.confusion(p, y, 0.5));
  return starterNote("metrics at 0.50 only.")
    + caption(`${TEST} · ${m11.MODEL_LABEL[model]}, threshold 0.50`)
    + table([{ Sensitivity: m11.fmtPct(m.sensitivity), Specificity: m11.fmtPct(m.specificity),
               Precision: m11.fmtPct(m.precision), "Flagged per 100": m.flagged_per_100.toFixed(1) }]);
}

// ---- 3. Training vs. Test ------------------------------------------------------
function trainVsTest(res) {
  const s = m11.aurocSummary(res);
  const rows = Object.entries(s).map(([m, v]) => ({ Model: m11.MODEL_LABEL[m],
    [`${TRAIN} AUROC`]: m11.fmtNum(v.train), [`${TEST} AUROC`]: m11.fmtNum(v.test) }));
  return starterNote("training and test AUROC of every model, in a table.")
    + caption(`${TRAIN} and ${TEST} · AUROC on the rows each model learned from, and on rows it never saw`) + table(rows);
}

const RENDER = Object.fromEntries(SECTIONS.map((s, i) => [s, [beyondAccuracy, threshold, trainVsTest][i]]));

// ---- page ----------------------------------------------------------------------
async function render() {
  const { section, key, model } = state;
  const res = await load(key);
  $("section-title").textContent = section;
  const content = $("content");
  if (!res) {
    $("context").textContent = "";
    content.innerHTML = `<p class="warn">No results for ${esc(m11.DATASET_LABEL[key])} yet: run <code>python -m analysis</code>, then commit and push <code>results/</code>.</p>`;
    return;
  }
  $("context").textContent = `${m11.DATASET_LABEL[key]}: ${int(res.train.n)} training rows (${int(res.train.n_pos)} positive), `
    + `${int(res.test.n)} test rows (${int(res.test.n_pos)} positive).`;
  [...$("model").options].forEach((o) => { o.text = m11.MODEL_LABEL[o.value] + (o.value in res.models ? "" : " (not trained yet)"); });
  content.innerHTML = model in res.models ? RENDER[section](res, model)
    : `<p class="warn">${esc(m11.MODEL_LABEL[model])} has no results yet: run <code>python -m analysis</code>, then commit and push <code>results/</code>.</p>`;
}

function init() {
  $("sections").innerHTML = SECTIONS.map((s, i) => `<label class="radio"><input type="radio" name="section" value="${i}"
    ${s === state.section ? "checked" : ""}>${esc(s)}</label>`).join("");
  $("dataset").innerHTML = m11.DATASETS.map((k) => `<option value="${k}">${esc(m11.DATASET_LABEL[k])}</option>`).join("");
  $("model").innerHTML = m11.MODELS.map((m) => `<option value="${m}">${esc(m11.MODEL_LABEL[m])}</option>`).join("");
  $("dataset").value = state.key;
  $("model").value = state.model;
  $("sections").addEventListener("change", (e) => { state.section = SECTIONS[Number(e.target.value)]; render(); });
  $("dataset").addEventListener("change", (e) => { state.key = e.target.value; render(); });
  $("model").addEventListener("change", (e) => { state.model = e.target.value; render(); });
  recommendations();
  render();
}

init();
