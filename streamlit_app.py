"""Judge the Model: three analysis modules that help a clinical lead judge whether any classifier is fit for each job.

Run locally:  bash run_app.sh  (it runs streamlit run streamlit_app.py)
The app reads only results/ and computes every number with m11.py. Every table and figure is labelled Training or Test.
The capsid, helix and sheet geometry is decoration only: the data are clinical records, not virology or biochemistry experiments.
"""

import html
import math
import random

import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]
DEFAULT_T = 0.5
STEPS = [round(i / 100, 2) for i in range(1, 100)]  # the slider's thresholds, 0.01 to 0.99
GOALS = [None, 0.5, 0.7, 0.8, 0.9]  # sensitivity goals the lead can pick in Section 2

# Plain-language wording for each job. Words only: every number comes from m11.
WORDS = {
    "readmission": {
        "job": "Predict, at discharge, which patients will be readmitted within 30 days.",
        "positive": "readmitted within 30 days",
        "framing": "Inpatients with diabetes in US hospitals, 1999–2008. Results apply to this historical cohort, "
                   "not to all discharges.",
        "person": "patient", "a_person": "a patient", "people": "patients",
        "who_pos": "were readmitted within 30 days", "who_neg": "were not readmitted within 30 days",
        "one_pos": "was readmitted within 30 days", "one_neg": "was not",
        "pos": "readmitted", "neg": "not readmitted",
        "always_no": "you will not be readmitted",
    },
    "diabetes": {
        "job": "Flag adults for follow-up testing. Finding the adults who have diabetes matters most.",
        "positive": "diabetes (a reported doctor diagnosis, or HbA1c ≥ 6.5%)",
        "framing": "US adults, NHANES 2017–2018. The label includes adults already diagnosed and describes one point in "
                   "time; it does not predict future diabetes.",
        "person": "adult", "a_person": "an adult", "people": "adults",
        "who_pos": "had diabetes", "who_neg": "did not have diabetes",
        "one_pos": "had diabetes", "one_neg": "did not",
        "pos": "diabetes", "neg": "no diabetes",
        "always_no": "you do not have diabetes",
    },
}

# Each section as an analysis module: (number, title, what the reader can do after using it).
MODULES = dict(zip(SECTIONS, [
    ("01", "Beyond Accuracy", "Name the positive class, see why accuracy alone misleads, and read a confusion matrix as four "
                              "groups of {ppl}."),
    ("02", "A Threshold for the Job", "The threshold is a decision separate from the model. See what moving it gains and gives "
                                      "up compared with {t:.2f}."),
    ("03", "Training vs. Test", "See why a model is judged on {ppl} it never saw, and spot a model that memorised its training "
                                "{ppl}."),
]))

st.set_page_config(page_title="Judge the Model", page_icon=":material/hexagon:", layout="wide")

# Teal/amber mark a right/wrong decision; cyan/amber mark a test result and its gap. Each pair passes the colour-blind
# checks on both panel surfaces. Teal and cyan are too close to tell apart, so they never share a figure.
STYLE = """<style>
:root {
  --bg-0:#0a111c; --bg-1:#101926; --bg-2:#162231; --char:#12171f;
  --line:#243346; --line-2:#344860;
  --ink-1:#e8eef5; --ink-2:#b7c4d2; --ink-3:#93a3b6;
  --cyan:#25a5c4; --cyan-ink:#8fd3e0; --teal:#22a088; --amber:#bd8422; --amber-ink:#e9c27a; --slate:#8493a6;
  --sans:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}

/* ---- page frame: a faint six-fold lattice behind everything ---- */
.stApp {
  background:
    radial-gradient(1100px 560px at 88% -8%, rgba(37,165,196,.09), transparent 62%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='38.1' height='66'%3E%3Cpath d='M19.05 0L38.1 11V33L19.05 44L0 33V11ZM19.05 44V66' fill='none' stroke='%2325a5c4' stroke-opacity='.05'/%3E%3C/svg%3E"),
    var(--bg-0);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stExpandSidebarButton"] { background: var(--bg-2); border: 1px solid var(--line-2); border-radius: 2px; }
[data-testid="stMainBlockContainer"] { padding: 1.25rem 2rem 4rem; max-width: 1560px; }
[data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4) { padding: 0; margin: 0; }
[data-testid="stMarkdownContainer"] :is(section, header, article, dl, dd, dt, ul, ol, li) p, [data-testid="stMarkdownContainer"] p { margin: 0; }
[data-testid="stMarkdownContainer"] dl { padding: 0; margin: 0; }
[data-testid="stMarkdownContainer"] dd { margin: 0; }
.stMarkdown [data-testid="stMarkdownContainer"] { margin-bottom: 0; }

/* ---- sidebar: the control column ---- */
[data-testid="stSidebar"] { background: var(--char); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font: 500 .7rem var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] > div {
  position: relative; width: 100%; padding: .5rem .7rem .52rem;
  border: 1px solid transparent; border-left: 2px solid var(--line-2); border-radius: 2px;
}
[data-testid="stSidebar"] [role="radiogroup"] > div:hover { background: rgba(255,255,255,.035); }
[data-testid="stSidebar"] [role="radiogroup"] > div:has([data-selected]) {
  background: rgba(37,165,196,.12); border-color: rgba(37,165,196,.32); border-left-color: var(--cyan);
}
[data-testid="stSidebar"] [role="radiogroup"] > div:has([data-focus-visible]) { outline: 2px solid var(--cyan-ink); outline-offset: 1px; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] { margin: 0; padding: 0; cursor: pointer; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]::after { content: ""; position: absolute; inset: 0; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font: 500 .9rem var(--sans); color: var(--ink-1); }
[data-testid="stSidebar"] [data-testid="stRadioCaption"] { padding: .1rem 0 0; pointer-events: none; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: .76rem; color: var(--ink-3); }
.brand { display: flex; align-items: center; gap: .7rem; padding-bottom: .9rem; border-bottom: 1px solid var(--line); }
.brand svg { width: 44px; height: 44px; flex: none; }
.brand b { display: block; font: 600 .86rem var(--sans); letter-spacing: .11em; color: var(--ink-1); }
.brand span { display: block; font: 400 .72rem var(--mono); color: var(--ink-3); margin-top: .12rem; }
.record { border-top: 1px solid var(--line); padding-top: .85rem; }
.stApp .record h4 { padding: 0; margin: 0 0 .45rem; font: 500 .7rem var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }
.record dl { display: grid; grid-template-columns: auto minmax(0,1fr); gap: .22rem .7rem; margin: 0; font: 400 .74rem var(--mono); }
.record dt { color: var(--ink-3); }
.record dd { margin: 0; color: var(--ink-2); text-align: right; white-space: nowrap; }
.stApp .record p { font-size: .76rem; color: var(--ink-3); margin-top: .6rem; line-height: 1.45; }

/* ---- masthead ---- */
.masthead {
  position: relative; overflow: hidden; border: 1px solid var(--line-2); border-radius: 3px;
  background: linear-gradient(100deg, #0d1827 0%, #0f1c2e 52%, #0a1320 100%);
}
.mast-grain {
  position: absolute; inset: 0; opacity: .32; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .55 0 0 0 0 .7 0 0 0 0 .8 0 0 0 .32 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.mast-field { position: absolute; inset: 0; pointer-events: none; -webkit-mask-image: linear-gradient(90deg, transparent 34%, #000 74%); mask-image: linear-gradient(90deg, transparent 34%, #000 74%); }
.mast-field svg { width: 100%; height: 100%; display: block; }
.mast-field path { fill: none; stroke: var(--cyan-ink); stroke-width: .6; stroke-opacity: .34; }
.mast-field circle { fill: none; stroke: var(--amber); stroke-width: .7; stroke-opacity: .42; }
.mast-capsid { position: absolute; right: clamp(-90px, -3vw, 0px); top: -34px; width: clamp(280px, 27vw, 410px); pointer-events: none; }
.mast-capsid::before { content: ""; position: absolute; inset: 6%; border-radius: 50%; background: radial-gradient(circle, rgba(37,165,196,.17), transparent 68%); }
.mast-capsid svg { position: relative; width: 100%; height: auto; display: block; }
.mast-body { position: relative; padding: 1.55rem 1.7rem 1rem; max-width: min(66%, 860px); }
.stApp p.eyebrow { font: 500 .74rem var(--mono); letter-spacing: .17em; text-transform: uppercase; color: var(--cyan-ink); }
.stApp .masthead h1 {
  font: 600 clamp(2.15rem, 4.5vw, 4.1rem)/1.04 var(--sans); letter-spacing: .05em; color: var(--ink-1); margin: .5rem 0 .6rem;
}
.stApp .masthead h1 .the { font-weight: 300; color: var(--cyan-ink); }
.stApp p.question { font: 400 1.08rem/1.45 var(--sans); color: var(--ink-2); max-width: 44rem; }
.mast-helix { position: relative; height: 26px; -webkit-mask-image: linear-gradient(90deg, #000 40%, rgba(0,0,0,.3)); mask-image: linear-gradient(90deg, #000 40%, rgba(0,0,0,.3)); }
.mast-helix svg { width: 100%; height: 100%; display: block; }
.mast-strip {
  position: relative; display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 0;
  border-top: 1px solid var(--line); background: #0b1421;
}
.mast-strip div { padding: .55rem 1.7rem; border-right: 1px solid var(--line); }
.mast-strip div:last-child { border-right: 0; }
.mast-strip dt { font: 500 .66rem var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }
.mast-strip dd { margin: .12rem 0 0; font: 500 .93rem var(--sans); color: var(--ink-1); }
.mast-strip dd small { font: 400 .76rem var(--mono); color: var(--ink-3); }
.stApp p.mast-note { position: relative; padding: .42rem 1.7rem; border-top: 1px solid var(--line); background: #0b1421; font: 400 .76rem/1.4 var(--sans); color: var(--ink-3); }
.stApp p.nav-hint { display: none; margin-top: .6rem; font: 500 .8rem var(--sans); color: var(--cyan-ink); }

/* ---- panels: each table or figure sits in one, with its Training or Test label ---- */
.stack { display: grid; gap: 1rem; }
.intro { display: grid; gap: .5rem; }
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr)); align-items: start; }
.fig-row { display: grid; gap: 1rem; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); align-items: stretch; }
.fig-row .panel { display: flex; flex-direction: column; }
.fig-row .panel-body { flex: 1; }
@media (max-width: 1180px) { .fig-row { grid-template-columns: minmax(0, 1fr); } .figure-roc { max-width: 460px; margin: 0 auto; } }
.panel { position: relative; min-width: 0; background: var(--bg-1); border: 1px solid var(--line); border-radius: 3px; }
.panel::before {
  content: ""; position: absolute; left: -1px; top: -1px; width: 13px; height: 13px;
  border-left: 2px solid var(--cyan); border-top: 2px solid var(--cyan); border-top-left-radius: 3px;
}
.panel-head { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .65rem; padding: .58rem .95rem; border-bottom: 1px solid var(--line); }
.stApp .panel-head h3, .stApp .deck-head h3 { font: 600 .98rem/1.3 var(--sans); color: var(--ink-1); }
.stApp .panel-head h2 { font: 600 1.02rem/1.3 var(--sans); color: var(--ink-1); }
.fig-no { font: 500 .7rem var(--mono); letter-spacing: .05em; color: var(--ink-3); }
.chips { display: inline-flex; gap: .3rem; margin-left: auto; }
.chip { font: 600 .67rem var(--mono); letter-spacing: .09em; text-transform: uppercase; padding: .13rem .46rem; border: 1px solid; border-radius: 2px; }
.chip-test { color: var(--cyan-ink); border-color: rgba(37,165,196,.6); background: rgba(37,165,196,.11); }
.chip-training { color: var(--ink-2); border-color: var(--line-2); background: rgba(132,147,166,.12); }
.panel-body { padding: .85rem .95rem; }
.stApp p.note { padding: .55rem .95rem; border-top: 1px solid var(--line); font: 400 .82rem/1.5 var(--sans); color: var(--ink-3); }
.stApp p.prose { font: 400 .97rem/1.6 var(--sans); color: var(--ink-2); max-width: 82ch; }
.prose strong, .callout strong { color: var(--ink-1); font-weight: 600; }

/* ---- module header ---- */
.module-head { display: flex; align-items: center; gap: 1rem; padding: 1.05rem 0 .15rem; }
.module-head svg { width: 52px; height: 52px; flex: none; }
.stApp p.module-no { font: 500 .72rem var(--mono); letter-spacing: .16em; text-transform: uppercase; color: var(--cyan-ink); }
.stApp .module-head h2 { font: 600 1.62rem/1.2 var(--sans); color: var(--ink-1); margin: .1rem 0 .15rem; }
.stApp .module-head p.lead { font: 400 .95rem/1.45 var(--sans); color: var(--ink-2); max-width: 80ch; }

/* ---- recommendations ---- */
.rec-grid { display: grid; gap: 0; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }
.rec { padding: .75rem .95rem .85rem; border-right: 1px solid var(--line); }
.rec:last-child { border-right: 0; }
.stApp .rec h3 { font: 500 .7rem var(--mono); letter-spacing: .11em; text-transform: uppercase; color: var(--cyan-ink); margin-bottom: .32rem; }
.stApp .rec p { font: 400 .9rem/1.55 var(--sans); color: var(--ink-1); }
.src { font: 400 .72rem var(--mono); color: var(--ink-3); margin-left: auto; }

/* ---- tiles ---- */
.tiles { display: grid; gap: 1px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 215px), 1fr)); background: var(--line); }
.tile { background: var(--bg-1); padding: .8rem .95rem .85rem; min-width: 0; }
.tile-label { display: block; font: 500 .7rem/1.35 var(--mono); letter-spacing: .05em; text-transform: uppercase; color: var(--ink-3); }
.tile-value { display: block; font: 600 1.92rem/1.2 var(--sans); color: var(--ink-1); margin: .22rem 0 .2rem; }
.tile-sub { display: block; font: 400 .82rem/1.42 var(--sans); color: var(--ink-2); }

/* ---- callouts ---- */
.callout { padding: .75rem .95rem; border-left: 2px solid var(--cyan); background: rgba(37,165,196,.07); font: 400 .95rem/1.58 var(--sans); color: var(--ink-2); }
.callout.warn { border-left-color: var(--amber); background: rgba(189,132,34,.09); font-size: .87rem; }
.callout .tag { font: 600 .68rem var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--amber-ink); margin-right: .45rem; }
.definition { display: grid; gap: 0; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); }
.definition div { padding: .68rem .95rem; border-right: 1px solid var(--line); }
.definition div:last-child { border-right: 0; }
.definition dt { font: 500 .68rem var(--mono); letter-spacing: .11em; text-transform: uppercase; color: var(--ink-3); margin-bottom: .2rem; }
.definition dd { margin: 0; font: 400 .93rem/1.45 var(--sans); color: var(--ink-1); }

/* ---- confusion matrix: rows = actual, columns = predicted, positives first ---- */
.cm { width: 100%; border-collapse: separate; border-spacing: 3px; table-layout: fixed; overflow-wrap: normal; }
.cm th { font: 500 .78rem/1.3 var(--mono); color: var(--ink-2); padding: .22rem .4rem; text-align: center; border: 0; background: none; }
.cm th.axis { font: 600 .66rem/1.2 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--cyan-ink); }
.cm th.axis-row { text-align: left; padding-left: 0; width: 23%; }
.cm tbody th { text-align: left; width: 23%; padding-left: 0; }
.cm tbody th small { display: block; font: 400 .72rem var(--mono); color: var(--ink-3); margin-top: .1rem; }
.cm td.corner { border: 0; background: none; }
.cell { background: color-mix(in srgb, var(--c) calc(var(--a) * 100%), var(--bg-2)); padding: .72rem .8rem; vertical-align: top; border: 0; border-radius: 2px; }
.cell.good { --c: var(--teal); }
.cell.bad { --c: var(--amber); }
.cell strong { display: block; font: 600 1.78rem/1.12 var(--sans); color: var(--ink-1); }
.cell .nm { display: block; font: 500 .9rem/1.3 var(--sans); color: var(--ink-1); margin-top: .22rem; }
.cell .tm { display: block; font: 400 .72rem/1.4 var(--mono); color: #d9e2ec; margin-top: .14rem; }
.minis { display: grid; gap: .6rem; grid-template-columns: repeat(auto-fit, minmax(min(100%, 186px), 1fr)); }
.mini { background: var(--bg-2); border: 1px solid var(--line); border-radius: 2px; padding: .52rem .55rem .45rem; min-width: 0; }
.mini.sel { border-color: var(--cyan); box-shadow: inset 0 0 0 1px rgba(37,165,196,.35); }
.stApp .mini h4 { padding: 0 0 .32rem .15rem; margin: 0; min-height: 2.6em; font: 600 .84rem/1.3 var(--sans); color: var(--ink-1); }
.mini h4 small { display: block; font: 400 .7rem/1.2 var(--mono); color: var(--cyan-ink); }
.cm-mini { border-spacing: 2px; }
.cm-mini th { font-size: .64rem; padding: .12rem .1rem; }
.cm-mini th.axis { font-size: .58rem; letter-spacing: .1em; }
.cm-mini th.axis-row, .cm-mini tbody th { width: 40%; padding: .12rem .3rem .12rem 0; }
.cm-mini .cell { padding: .3rem .28rem; background: color-mix(in srgb, var(--c) calc(var(--a) * 100%), var(--bg-1)); white-space: nowrap; }
.cm-mini .cell strong { font: 600 .84rem/1.15 var(--mono); font-variant-numeric: tabular-nums; }
.cm-mini .cell .tm { font-size: .62rem; margin-top: 0; }

/* ---- metric readout ---- */
.readout { margin: 0; }
.readout > div { display: grid; grid-template-columns: minmax(0,1fr) auto; column-gap: .9rem; padding: .56rem 0 .6rem; border-bottom: 1px solid var(--line); }
.readout > div:first-child { padding-top: 0; }
.readout > div:last-child { border-bottom: 0; padding-bottom: 0; }
.readout dt { font: 500 .8rem/1.35 var(--mono); color: var(--ink-1); align-self: end; }
.readout dd { margin: 0; }
.readout .v { font: 600 1.06rem/1.25 var(--mono); color: var(--ink-1); font-variant-numeric: tabular-nums; text-align: right; }
.readout .w { grid-column: 1 / -1; font: 400 .85rem/1.45 var(--sans); color: var(--ink-2); margin-top: .16rem; }
.meter { display: block; height: 3px; margin-top: .34rem; background: var(--line); border-radius: 2px; overflow: hidden; }
.meter i { display: block; height: 100%; background: var(--cyan); }
.readout .meter { grid-column: 1 / -1; }
.undef { color: var(--amber-ink) !important; font-style: italic; }

/* ---- tables ---- */
.scroll { overflow-x: auto; }
.dt { width: 100%; border-collapse: collapse; font: 400 .87rem var(--sans); }
.dt th { border: 0; border-bottom: 1px solid var(--line-2); padding: .5rem .7rem; background: none; font: 500 .7rem/1.3 var(--mono); letter-spacing: .03em; color: var(--ink-3); text-align: right; vertical-align: bottom; white-space: nowrap; }
.dt th small { display: block; font: 400 .67rem/1.3 var(--mono); letter-spacing: 0; color: var(--ink-3); white-space: normal; }
.dt td { border: 0; border-bottom: 1px solid var(--line); padding: .5rem .7rem; font: 400 .86rem var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-1); text-align: right; white-space: nowrap; }
.dt tbody th { border-bottom: 1px solid var(--line); font: 500 .88rem var(--sans); letter-spacing: 0; color: var(--ink-1); text-align: left; vertical-align: middle; }
.dt thead th:first-child { text-align: left; }
.dt tbody tr:last-child th, .dt tbody tr:last-child td { border-bottom: 0; }
.dt tr.sel td, .dt tr.sel th { background: rgba(37,165,196,.1); }
.dt tr.sep td, .dt tr.sep th { border-top: 1px solid var(--line-2); }
.dt tbody tr:hover td, .dt tbody tr:hover th { background: rgba(255,255,255,.03); }
.dt tr.sel:hover td, .dt tr.sel:hover th { background: rgba(37,165,196,.14); }
.dt .meter { width: 100%; min-width: 58px; }
.tag-sel { font: 400 .72rem var(--mono); color: var(--cyan-ink); margin-left: .3rem; }

/* ---- figures ---- */
.figure { display: block; width: 100%; height: auto; }
.scroll .figure-wide { min-width: 600px; }
.swipe { display: none; font: 400 .76rem var(--mono); color: var(--ink-3); margin-bottom: .3rem; }
.figure text { font-family: var(--mono); font-size: 13px; fill: var(--ink-3); }
.figure text.strong { fill: var(--ink-1); font-weight: 500; }
.figure text.lab { fill: var(--ink-2); font-family: var(--sans); font-size: 14px; paint-order: stroke; stroke: var(--bg-1); stroke-width: 3px; }
.figure-roc text { font-size: 12px; }
.figure-roc text.lab { font-size: 13px; }
.figure-roc text.strong { paint-order: stroke; stroke: var(--bg-1); stroke-width: 3px; }
.figure .grid-line { stroke: var(--line); stroke-width: 1; }
.figure .axis { stroke: var(--line-2); stroke-width: 1; }
.figure .b-good { fill: var(--teal); }
.figure .b-bad { fill: var(--amber); }
.figure .wash { fill: rgba(232,238,245,.035); }
.figure .t-line { stroke: var(--ink-1); stroke-width: 2; }
.figure .ref-line { stroke: var(--slate); stroke-width: 1; }
.figure .hit { fill: transparent; }
.figure .hit:hover { fill: rgba(232,238,245,.08); }
.figure .curve { fill: none; stroke: var(--cyan); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.figure .curve-area { fill: rgba(37,165,196,.1); }
.figure .op-now { fill: var(--ink-1); stroke: var(--bg-1); stroke-width: 2; }
.figure .op-ref { fill: var(--bg-1); stroke: var(--slate); stroke-width: 2; }
.stApp ul.legend { display: flex; flex-wrap: wrap; gap: .3rem 1.2rem; padding: 0; margin: 0 0 .55rem; list-style: none; font: 400 .85rem/1.4 var(--sans); color: var(--ink-2); }
.stApp .legend li { display: inline-flex; align-items: center; gap: .42rem; margin: 0; padding: 0; font: inherit; color: inherit; }
.legend b { color: var(--ink-1); font: 500 .85rem var(--mono); }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; flex: none; }
.sw.good { background: var(--teal); }
.sw.bad { background: var(--amber); }
.sw.line { width: 2px; height: 14px; border-radius: 0; background: var(--ink-1); }
.sw.ref { width: 1px; height: 14px; border-radius: 0; background: var(--slate); }
.sw.dot { border-radius: 50%; }
.sw.train { background: var(--slate); }
.sw.test { background: var(--cyan); }
.sw.gap { width: 16px; height: 2px; border-radius: 0; background: var(--amber); }

/* ---- training vs. test: a table with a dot pair in each row ---- */
.db-axis { position: relative; height: 1.15rem; margin: 0 .4rem; }
.db-axis span { position: absolute; bottom: 0; transform: translateX(-50%); font: 400 .68rem var(--mono); color: var(--ink-3); }
.dt td.db-cell { width: 42%; min-width: 280px; padding: 0 .7rem; }
.db-track { position: relative; height: 2.5rem; margin: 0 .4rem; }
.db-track i { position: absolute; display: block; }
.db-grid { top: 0; bottom: 0; width: 1px; background: var(--line); }
.db-grid.coin { background: var(--line-2); }
.db-link { top: calc(50% - 1px); height: 2px; background: var(--amber); }
.db-dot { top: calc(50% - 7px); width: 14px; height: 14px; margin-left: -7px; border-radius: 50%; border: 2px solid var(--bg-1); box-sizing: border-box; }
.db-dot.train { background: var(--slate); }
.db-dot.test { background: var(--cyan); }
.sel .db-dot { border-color: #14283a; }

/* ---- the threshold controls (Streamlit widgets inside a keyed container) ---- */
.stApp .st-key-deck { position: relative; background: var(--bg-1); border: 1px solid var(--line-2); border-radius: 3px; padding: 0 .95rem .85rem; }
.st-key-deck::before {
  content: ""; position: absolute; left: -1px; top: -1px; width: 13px; height: 13px;
  border-left: 2px solid var(--amber); border-top: 2px solid var(--amber); border-top-left-radius: 3px;
}
.deck-head { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .65rem; margin: 0 -.95rem; padding: .58rem .95rem; border-bottom: 1px solid var(--line); }
.st-key-deck [data-testid="stWidgetLabel"] p { font: 500 .88rem/1.4 var(--sans); color: var(--ink-1); }
.st-key-goal [role="radiogroup"] { gap: 0; flex-wrap: wrap; }
.st-key-goal [data-testid="stRadioOption"] { margin: 0 -1px 0 0; padding: .32rem .8rem; cursor: pointer; border: 1px solid var(--line-2); background: var(--bg-2); }
.st-key-goal [data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-goal [data-testid="stRadioOption"] p { font: 500 .84rem var(--mono); color: var(--ink-2); }
.st-key-goal [data-testid="stRadioOption"]:hover { background: #1c2c3f; }
.st-key-goal [data-testid="stRadioOption"][data-selected] { position: relative; z-index: 1; background: rgba(37,165,196,.2); border-color: var(--cyan); }
.st-key-goal [data-testid="stRadioOption"][data-selected] p { color: var(--ink-1); }
.st-key-goal [data-testid="stRadioOption"][data-focus-visible] { outline: 2px solid var(--cyan-ink); outline-offset: 1px; }
.st-key-deck [data-testid="stSliderThumbValue"], .st-key-deck [data-testid="stSliderTickBar"] { font-family: var(--mono); }
.st-key-deck [data-testid="stSliderThumbValue"] { font-weight: 600; }
.st-key-deck button[kind="secondary"] { font: 500 .84rem var(--mono); border-color: var(--cyan); background: rgba(37,165,196,.1); color: var(--ink-1); }
.st-key-deck button[kind="secondary"]:hover { background: rgba(37,165,196,.22); border-color: var(--cyan-ink); color: var(--ink-1); }
.stApp p.goal-read { font: 400 .92rem/1.55 var(--sans); color: var(--ink-2); }
.live { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 2px; }
.live div { background: var(--bg-2); padding: .45rem .6rem .5rem; min-width: 0; }
.live span { display: block; font: 500 .64rem/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--ink-3); }
.live strong { display: block; font: 600 1.28rem/1.25 var(--mono); color: var(--ink-1); margin-top: .12rem; }
.live small { display: block; font: 400 .7rem/1.3 var(--mono); color: var(--ink-3); }
.goal-read strong { color: var(--ink-1); font-weight: 600; }
.goal-read small { display: block; margin-top: .3rem; font-size: .8rem; color: var(--ink-3); }

@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding: 3rem 1rem 3rem; }
  .mast-body { max-width: 100%; padding: 1.2rem 1rem .8rem; }
  .mast-capsid { opacity: .4; right: -110px; top: -30px; }
  .mast-field { opacity: .5; }
  .mast-strip div, .stApp p.mast-note { padding-left: 1rem; padding-right: 1rem; }
  .stApp p.nav-hint { display: block; }
  .swipe { display: block; }
  .rec, .definition div { border-right: 0; border-bottom: 1px solid var(--line); }
  .rec:last-child, .definition div:last-child { border-bottom: 0; }
  .cm tbody th { width: 25%; }
  .cell { padding: .55rem .55rem; }
  .cell strong { font-size: 1.32rem; }
  .cell .nm { font-size: .8rem; }
}
</style>"""


def stamp(name: str) -> float:
    path = m11.RESULTS / f"{name}.json"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _load(name: str, mtime: float):
    return m11.load(name)


def load(name: str):
    """Reload whenever `python -m analysis` rewrites the file."""
    return _load(name, stamp(name))


# ---- decorative geometry: drawn from real capsid and protein geometry, and carrying no data ----------------
PHI = (1 + 5 ** 0.5) / 2
ICO_VERTS = [(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0), (0, -1, PHI), (0, 1, PHI),
             (0, -1, -PHI), (0, 1, -PHI), (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)]
ICO_FACES = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]


def _rotate(v, yaw: float, pitch: float, roll: float) -> tuple:
    x, y, z = v
    x, z = math.cos(yaw) * x + math.sin(yaw) * z, -math.sin(yaw) * x + math.cos(yaw) * z
    y, z = math.cos(pitch) * y - math.sin(pitch) * z, math.sin(pitch) * y + math.cos(pitch) * z
    return math.cos(roll) * x - math.sin(roll) * y, math.sin(roll) * x + math.cos(roll) * y, z


@st.cache_data(show_spinner=False)
def capsid_mesh(freq: int) -> tuple:
    """A class I geodesic icosahedron (triangulation number freq²): 12 five-fold vertices, the rest six-fold."""
    corners = [tuple(c / math.sqrt(1 + PHI ** 2) for c in v) for v in ICO_VERTS]
    index, points, edges = {}, [], set()

    def vertex(a, b, c, i, j):
        k = freq - i - j
        q = tuple((i * a[n] + j * b[n] + k * c[n]) / freq for n in range(3))
        r = math.sqrt(sum(x * x for x in q))
        p = tuple(x / r * (0.5 + 0.5 * r) for x in q)  # part-way between flat facet and sphere, so it reads as faceted
        key = tuple(round(x, 4) for x in p)
        if key not in index:
            index[key] = len(points)
            points.append(p)
        return index[key]

    for f in ICO_FACES:
        a, b, c = (corners[n] for n in f)
        grid = {(i, j): vertex(a, b, c, i, j) for i in range(freq + 1) for j in range(freq + 1 - i)}
        for (i, j), v in grid.items():
            for di, dj in ((1, 0), (0, 1), (1, -1)):
                u = grid.get((i + di, j + dj))
                if u is not None:
                    edges.add((min(u, v), max(u, v)))
    return points, sorted(edges)


@st.cache_data(show_spinner=False)
def capsid_svg(freq: int, yaw: float = 0.32, pitch: float = 0.2, roll: float = 0.16, weight: float = 1.0, dot: float = 1.0) -> str:
    """The particle, depth-cued: edges and capsomers fade towards the back; five-fold vertices are amber."""
    points, edges = capsid_mesh(freq)
    pts = [_rotate(p, yaw, pitch, roll) for p in points]
    xy = [(100 + 96 * p[0], 100 - 96 * p[1]) for p in pts]
    layers = [[], [], [], []]
    degree = [0] * len(pts)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        z = (pts[a][2] + pts[b][2]) / 2
        layers[min(3, int((z + 1) / 2 * 4))].append(f"M{xy[a][0]:.1f} {xy[a][1]:.1f}L{xy[b][0]:.1f} {xy[b][1]:.1f}")
    out = [f'<path d="{"".join(seg)}" fill="none" stroke="#5fc0d6" stroke-width="{weight * w:.2f}" stroke-opacity="{o}"/>'
           for seg, w, o in zip(layers, (.32, .4, .55, .8), (.14, .24, .46, .86))]
    for i, (x, y) in enumerate(xy):
        if pts[i][2] < -0.15:
            continue
        front = pts[i][2] > 0.35
        if degree[i] == 5:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{(2.5 if front else 1.7) * dot:.2f}" fill="#d9a03a" fill-opacity="{1 if front else .55}"/>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{(1.45 if front else 1) * dot:.2f}" fill="#a9dfe9" fill-opacity="{.9 if front else .38}"/>')
    return f'<svg viewBox="0 0 200 200" aria-hidden="true" focusable="false">{"".join(out)}</svg>'


@st.cache_data(show_spinner=False)
def particle_field() -> str:
    """Small icosahedra at random orientations, some ringed, the way picked particles look on a micrograph."""
    rng = random.Random(11)
    points, edges = capsid_mesh(1)
    placed, seg, rings = [], [], []
    for _ in range(600):
        x, y, r = rng.uniform(360, 1000), rng.uniform(8, 292), rng.uniform(6.5, 12)
        if len(placed) >= 26 or any((x - a) ** 2 + (y - b) ** 2 < (r + c + 24) ** 2 for a, b, c in placed):
            continue
        placed.append((x, y, r))
        pts = [_rotate(p, rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28)) for p in points]
        seg += [f"M{x + r * pts[a][0]:.1f} {y - r * pts[a][1]:.1f}L{x + r * pts[b][0]:.1f} {y - r * pts[b][1]:.1f}" for a, b in edges]
        if len(placed) % 3 == 0:
            rings.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 6:.1f}"/>')
    return (f'<svg viewBox="0 0 1000 300" preserveAspectRatio="xMaxYMid slice" aria-hidden="true" focusable="false">'
            f'<path d="{"".join(seg)}"/>{"".join(rings)}</svg>')


@st.cache_data(show_spinner=False)
def helix_svg(residues: int, width: float, height: float, stroke: float = 1.1, dots: bool = True) -> str:
    """The Cα trace of an ideal alpha helix, side on: 100° of rotation per residue."""
    amp = height / 2 - 2.5
    step = width / (residues - 1)
    back, front, atoms = [], [], []
    for i in range(3 * (residues - 1)):
        t0, t1 = i / 3, (i + 1) / 3
        a0, a1 = math.radians(100 * t0), math.radians(100 * t1)
        seg = f"M{t0 * step:.1f} {height / 2 + amp * math.sin(a0):.1f}L{t1 * step:.1f} {height / 2 + amp * math.sin(a1):.1f}"
        (front if math.cos((a0 + a1) / 2) > 0 else back).append(seg)
    if dots:
        for i in range(residues):
            a = math.radians(100 * i)
            atoms.append(f'<circle cx="{i * step:.1f}" cy="{height / 2 + amp * math.sin(a):.1f}" r="1.5" fill="#5fc0d6" '
                         f'fill-opacity="{.85 if math.cos(a) > 0 else .3}"/>')
    return (f'<svg viewBox="0 0 {width:g} {height:g}" preserveAspectRatio="xMinYMid slice" aria-hidden="true" focusable="false">'
            f'<path d="{"".join(back)}" fill="none" stroke="#5fc0d6" stroke-width="{stroke}" stroke-opacity=".28"/>'
            f'<path d="{"".join(front)}" fill="none" stroke="#5fc0d6" stroke-width="{stroke}" stroke-opacity=".82"/>{"".join(atoms)}</svg>')


@st.cache_data(show_spinner=False)
def sheet_svg() -> str:
    """A small antiparallel beta sheet: pleated strands, with hairlines for the backbone hydrogen bonds."""
    rows = (9, 24, 39)
    strands, bonds, atoms = [], [], []
    for n, y in enumerate(rows):
        xs = [4 + 8 * i for i in range(6)]
        ys = [y + (2.4 if (i + n) % 2 else -2.4) for i in range(6)]
        strands.append("M" + "L".join(f"{x} {v:.1f}" for x, v in zip(xs, ys)))
        atoms += [f'<circle cx="{x}" cy="{v:.1f}" r="1.7" fill="#a9dfe9"/>' for x, v in zip(xs, ys)]
        if n < 2:
            bonds += [f"M{x} {y + 3}V{rows[n + 1] - 3}" for x in xs[1::2]]
    return (f'<svg viewBox="0 0 48 48" aria-hidden="true" focusable="false"><path d="{"".join(bonds)}" stroke="#d9a03a" stroke-width=".7" stroke-opacity=".8"/>'
            f'<path d="{"".join(strands)}" fill="none" stroke="#5fc0d6" stroke-width="1.5"/>{"".join(atoms)}</svg>')


def module_glyph(number: str) -> str:
    if number == "01":
        return capsid_svg(1, 0.55, 0.3, 0.1, 7, 3.4)
    return helix_svg(11, 48, 26, 1.5) if number == "02" else sheet_svg()


# ---- shared displays -----------------------------------------------------------
def show(body: str, where=st) -> None:
    """Our own HTML. st.markdown, because st.html strips inline SVG. Every value in it is generated here or passed through esc()."""
    where.markdown(body, unsafe_allow_html=True, anchors=False)


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def fmt_metric(name: str, v) -> str:
    return m11.fmt_num(v) if name == "F1" else m11.fmt_pct(v)


def per_100(k: int, n: int):
    """A count from m11.confusion, per 100 people screened."""
    return None if n == 0 else 100 * k / n


def num(text: str) -> str:
    """A value for display; "undefined" (0 ÷ 0) is marked so it can never be read as a number."""
    return f'<span class="undef">{text}</span>' if text == "undefined" else text


def meter(share) -> str:
    return "" if share is None else f'<span class="meter"><i style="width:{share * 100:.1f}%"></i></span>'


def chip(part: str) -> str:
    return f'<span class="chip chip-{part.lower()}">{part}</span>'


def panel(title: str, body: str, parts=(TEST,), fig: str = "", note: str = "", body_class: str = "panel-body") -> str:
    """A framed table or figure. `parts` is its Training or Test label."""
    no = f'<span class="fig-no">{fig}</span>' if fig else ""
    chips = f'<span class="chips">{"".join(chip(p) for p in parts)}</span>' if parts else ""
    foot = f'<p class="note">{note}</p>' if note else ""
    return (f'<section class="panel"><header class="panel-head">{no}<h3>{title}</h3>{chips}</header>'
            f'<div class="{body_class}">{body}</div>{foot}</section>')


def tile(label: str, value: str, sub: str = "") -> str:
    return f'<div class="tile"><span class="tile-label">{label}</span><strong class="tile-value">{num(value)}</strong><span class="tile-sub">{sub}</span></div>'


def data_table(headers: list, rows: list, sel=None, seps=()) -> str:
    """rows = [(row name, [cells])]; a cell is text, or (text, share) to draw a thin meter under it."""
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = []
    for i, (name, cells) in enumerate(rows):
        tds = "".join(f"<td>{num(c[0])}{meter(c[1])}</td>" if isinstance(c, tuple) else f"<td>{num(c)}</td>" for c in cells)
        cls = " ".join(k for k, on in (("sel", i == sel), ("sep", i in seps)) if on)
        mark = '<span class="tag-sel">(selected)</span>' if i == sel else ""
        body.append(f'<tr class="{cls}"><th scope="row">{name}{mark}</th>{tds}</tr>')
    return f'<div class="scroll"><table class="dt"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def matrix(c: dict, w: dict, detailed: bool = True) -> str:
    """Rows = what actually happened, columns = what the model predicted, positives first
    (scikit-learn's confusion_matrix uses the opposite order). The tint shows the share of each row."""
    m = m11.metrics(c)
    miss = None if m["sensitivity"] is None else 1 - m["sensitivity"]
    cells = [[("tp", "good", m["sensitivity"], "caught", "true positives"), ("fn", "bad", miss, "missed", "false negatives")],
             [("fp", "bad", m["fpr"], "false alarms", "false positives"), ("tn", "good", m["specificity"], "correctly not flagged", "true negatives")]]
    rows = []
    for name, total, row in zip((w["pos"], w["neg"]), (m["positives"], m["negatives"]), cells):
        tds = []
        for k, tone, share, group, term in row:
            alpha = 0.06 + 0.36 * (share or 0)
            if detailed:
                inner = (f'<strong>{c[k]:,}</strong><span class="nm">{group}</span>'
                         f'<span class="tm">{term} · {m11.fmt_pct(share)} of row</span>')
            else:
                inner = f'<strong>{c[k]:,}</strong><span class="tm">{k.upper()}</span>'
            tds.append(f'<td class="cell {tone}" style="--a:{alpha:.3f}" title="{esc(group)} ({esc(term)}): {c[k]:,}, {m11.fmt_pct(share)} of this row">{inner}</td>')
        count = f"<small>{total:,} {w['people']}</small>" if detailed else ""
        rows.append(f'<tr><th scope="row">{name}{count}</th>{"".join(tds)}</tr>')
    return (f'<table class="cm{"" if detailed else " cm-mini"}"><thead><tr><td class="corner"></td><th class="axis" colspan="2" scope="colgroup">Predicted</th></tr>'
            '<tr><th class="axis axis-row" scope="col">Actual</th><th scope="col">flagged</th><th scope="col">not flagged</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def explained(c: dict, w: dict) -> str:
    """Each metric with its denominator in words."""
    m, ppl = m11.metrics(c), w["people"]
    rows = [
        ("Sensitivity (recall, TPR)", m["sensitivity"],
         f"Of the {m['positives']:,} {ppl} who {w['who_pos']}, the share flagged ({c['tp']:,} of {m['positives']:,})."),
        ("Specificity", m["specificity"],
         f"Of the {m['negatives']:,} {ppl} who {w['who_neg']}, the share not flagged ({c['tn']:,} of {m['negatives']:,})."),
        ("Precision", m["precision"],
         f"Of the {m['flagged']:,} {ppl} flagged, the share who {w['who_pos']} ({c['tp']:,} of {m['flagged']:,})."),
        ("False positive rate (FPR)", m["fpr"],
         f"Of the {m['negatives']:,} {ppl} who {w['who_neg']}, the share flagged anyway ({c['fp']:,} of "
         f"{m['negatives']:,}). It equals 1 − specificity."),
        ("Accuracy", m["accuracy"],
         f"Of all {m['n']:,} {ppl}, the share whose flag matched what happened ({c['tp'] + c['tn']:,} of {m['n']:,})."),
        ("F1", m["f1"], "Precision and sensitivity combined into one number from 0 to 1 (their harmonic mean). "
                        "It ignores the correctly-not-flagged group."),
    ]
    out = []
    for name, v, words in rows:
        if v is None:
            words += " That group is empty, so this is 0 ÷ 0: undefined."
        out.append(f'<div><dt>{name}</dt><dd class="v">{num(fmt_metric(name, v))}</dd>{meter(v)}<dd class="w">{words}</dd></div>')
    return f'<dl class="readout">{"".join(out)}</dl>'


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    if any(recs.values()):
        body = "".join(f'<article class="rec"><h3>{esc(m11.DATASET_LABEL[k])}</h3><p>{esc(recs.get(k) or "Not written yet.")}</p></article>'
                       for k in m11.DATASETS)
    else:
        body = ('<p class="rec">No recommendations yet: fill in <code>Recommendations</code> in DECISIONS.md, run '
                "<code>python -m analysis</code>, then commit and push.</p>")
    show(f'<section class="panel"><header class="panel-head"><h2>Recommendations</h2><span class="src">results/decisions.json</span></header>'
            f'<div class="rec-grid">{body}</div></section>')


def masthead(res: dict, key: str, model: str) -> None:
    tr, te = res["train"], res["test"]
    show(f"""<section class="masthead">
  <div class="mast-grain"></div>
  <div class="mast-field">{particle_field()}</div>
  <div class="mast-capsid">{capsid_svg(4)}</div>
  <div class="mast-body">
    <p class="eyebrow">Classifier evaluation console</p>
    <h1>JUDGE <span class="the">THE</span> MODEL</h1>
    <p class="question">Which classifier would you trust for this job, at what threshold, and with what limits?</p>
    <p class="nav-hint">Open the sidebar (arrow at the top left) to change module, dataset or model.</p>
  </div>
  <div class="mast-helix">{helix_svg(120, 1500, 26)}</div>
  <dl class="mast-strip">
    <div><dt>Dataset</dt><dd>{esc(m11.DATASET_LABEL[key])}</dd></div>
    <div><dt>Model</dt><dd>{esc(m11.MODEL_LABEL[model])}</dd></div>
    <div><dt>{TRAIN} rows</dt><dd>{tr["n"]:,} <small>{tr["n_pos"]:,} positive</small></dd></div>
    <div><dt>{TEST} rows</dt><dd>{te["n"]:,} <small>{te["n_pos"]:,} positive</small></dd></div>
  </dl>
  <p class="mast-note">Interface styled after cryo-EM and structural-virology analysis software. The theme is visual only: the data are
  hospital discharge records and a US health survey, not virology or biochemistry experiments.</p>
</section>""")


def module_header(section: str, w: dict) -> None:
    number, title, lead = MODULES[section]
    show(f'<header class="module-head">{module_glyph(number)}<div><p class="module-no">Analysis module {number}</p>'
            f'<h2>{title}</h2><p class="lead">{lead.format(ppl=w["people"], t=DEFAULT_T)}</p></div></header>')


# ---- 1. Beyond Accuracy ------------------------------------------------------
def beyond_accuracy(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, DEFAULT_T)
    m = m11.metrics(c)
    always = m11.majority_accuracy(y)

    definition = (f'<dl class="definition"><div><dt>The job</dt><dd>{w["job"]}</dd></div>'
                  f'<div><dt>Positive class (what the models look for)</dt><dd>{w["positive"][0].upper() + w["positive"][1:]}.</dd></div>'
                  f'<div><dt>Flagged</dt><dd>{w["a_person"].capitalize()} is flagged when the model\'s probability is at least {DEFAULT_T:.2f}.</dd></div></dl>')

    tiles = "".join([
        tile("Prevalence", m11.fmt_pct(m["prevalence"]), f'Of the {m["n"]:,} {ppl}, the share who {w["who_pos"]}.'),
        tile('"Always no" accuracy', m11.fmt_pct(always),
             f'Telling every {w["person"]} "{w["always_no"]}": the most common outcome for everyone.'),
        tile(f"{esc(label)} accuracy at {DEFAULT_T:.2f}", m11.fmt_pct(m["accuracy"]),
             f'Of all {m["n"]:,} {ppl}, the share whose flag matched what happened.'),
        tile(f"{esc(label)} catches at {DEFAULT_T:.2f}", f'{c["tp"]:,} of {m["positives"]:,}', f'Of the {ppl} who {w["who_pos"]}, the number flagged.'),
    ])
    story = (f'<div class="callout">Telling every {w["person"]} "{w["always_no"]}" is right for {m11.fmt_pct(always)} of them, and catches '
             f'<strong>none</strong> of the {m["positives"]:,} who {w["who_pos"]}. {label} at {DEFAULT_T:.2f} is right for '
             f'{m11.fmt_pct(m["accuracy"])}, and catches <strong>{c["tp"]:,} of {m["positives"]:,}</strong>. Because most {ppl} '
             f'{w["who_neg"]}, a model can score a high accuracy while missing most of the {ppl} who {w["who_pos"]}. '
             "Count who is caught and who is missed, not just accuracy.</div>")

    rows, minis, undefined = [], [], False
    names = [n for n in m11.MODELS if n in res["models"]]
    for name in names:
        cc = m11.confusion(*m11.preds(res, name), DEFAULT_T)
        mm = m11.metrics(cc)
        minis.append(f'<div class="mini{" sel" if name == model else ""}"><h4>{esc(m11.MODEL_LABEL[name])}'
                     f'{"<small>(selected)</small>" if name == model else ""}</h4>{matrix(cc, w, detailed=False)}</div>')
        cells = [f'{cc["tp"]:,} of {mm["positives"]:,}', f'{mm["flagged"]:,}']
        for metric, k in [("Accuracy", "accuracy"), ("Sensitivity (recall, TPR)", "sensitivity"), ("Specificity", "specificity"),
                          ("FPR", "fpr"), ("Precision", "precision"), ("F1", "f1")]:
            text = fmt_metric(metric, mm[k])
            cells.append((text, mm[k]) if k in ("accuracy", "sensitivity") else text)
            undefined |= mm[k] is None
        rows.append((esc(m11.MODEL_LABEL[name]), cells))
    note = f'"Always no" accuracy on the same {ppl}, for comparison: {m11.fmt_pct(always)}.'
    if undefined:
        note += f' "undefined" means 0 ÷ 0: for example, a model that flags no one has no flagged {ppl} to take a share of.'
    table = data_table(["Model", "Caught", "Flagged", "Accuracy", "Sensitivity<small>recall, TPR</small>", "Specificity", "FPR", "Precision", "F1"],
                       rows, sel=names.index(model) if model in names else None)

    show('<div class="stack">'
            + panel("The job", definition, parts=(), body_class="", note=w["framing"])
            + panel("Why accuracy alone misleads", f'<div class="tiles">{tiles}</div>{story}', fig="Fig. 1.1", body_class="")
            + '<div class="grid">'
            + panel(f"{esc(label)} at {DEFAULT_T:.2f}: four groups of {ppl}", matrix(c, w), fig="Fig. 1.2",
                    note="Rows = what actually happened. Columns = what the model predicted. Teal marks a right decision and amber "
                         "a wrong one; a stronger tint means a larger share of that row.")
            + panel(f"{esc(label)} at {DEFAULT_T:.2f}: the metrics in words", explained(c, w), fig="Tab. 1.1")
            + "</div>"
            + panel(f"Every model at {DEFAULT_T:.2f}: confusion matrices", f'<div class="minis">{"".join(minis)}</div>', fig="Fig. 1.3",
                    note=f"{m['n']:,} {ppl}. In each matrix, rows = actual and columns = predicted. TP caught, FN missed, FP false alarm, TN correctly not flagged.")
            + panel(f"Every model at threshold {DEFAULT_T:.2f}", table, fig="Tab. 1.2", body_class="", note=note)
            + "</div>")


# ---- 2. A Threshold for the Job ------------------------------------------------
# (label, how to read it from the counts and metrics, format)
TRADE_ROWS = [
    ("Flagged per 100 {ppl}", lambda c, m: m["flagged_per_100"], "per100"),
    ("Caught per 100 {ppl}", lambda c, m: per_100(c["tp"], m["n"]), "per100"),
    ("False alarms per 100 {ppl}", lambda c, m: per_100(c["fp"], m["n"]), "per100"),
    ("Caught (true positives)", lambda c, m: c["tp"], "count"),
    ("Missed (false negatives)", lambda c, m: c["fn"], "count"),
    ("False alarms (false positives)", lambda c, m: c["fp"], "count"),
    ("Sensitivity (recall, TPR)", lambda c, m: m["sensitivity"], "pct"),
    ("Specificity", lambda c, m: m["specificity"], "pct"),
    ("Precision", lambda c, m: m["precision"], "pct"),
    ("FPR", lambda c, m: m["fpr"], "pct"),
    ("Accuracy", lambda c, m: m["accuracy"], "pct"),
    ("F1", lambda c, m: m["f1"], "f1"),
]


def fmt_trade(v, kind: str) -> str:
    if v is None:
        return "undefined"
    return {"per100": f"{v:.1f}", "count": f"{v:,}", "pct": m11.fmt_pct(v), "f1": m11.fmt_num(v)}[kind]


def fmt_change(a, b, kind: str) -> str:
    if a is None or b is None:
        return "undefined"
    d = b - a
    return {"per100": f"{d:+.1f}", "count": f"{d:+,}", "pct": f"{d * 100:+.1f} points", "f1": f"{d:+.3f}"}[kind]


def more_fewer(d: float, unit: str, decimals: int = 0) -> str:
    return f"{abs(d):,.{decimals}f} {'more' if d > 0 else 'fewer'} {unit}" if d else f"the same number of {unit}"


@st.cache_data(show_spinner=False)
def probability_bands(key: str, model: str, mtime: float) -> tuple:
    """How many test rows sit in each 0.01-wide band of probability, for each actual outcome. Counted with m11.confusion
    at every slider step, so the bars either side of a threshold always add up to its confusion matrix."""
    p, y = m11.preds(m11.load(key), model)
    at = [m11.confusion(p, y, i / 100) for i in range(101)]  # flagged at 0.00, 0.01, ... 1.00
    pos = [at[i]["tp"] - at[i + 1]["tp"] for i in range(100)]
    neg = [at[i]["fp"] - at[i + 1]["fp"] for i in range(100)]
    pos[99] += at[100]["tp"]  # the last band includes a probability of exactly 1
    neg[99] += at[100]["fp"]
    return pos, neg


def share_ticks(top: float) -> list:
    step = next((s for s in (.01, .02, .05, .1, .2, .25, .5) if top / s <= 3), 1)
    return [step * i for i in range(1, int(top / step + 1e-9) + 1)]


def threshold_figure(pos: list, neg: list, t: float, w: dict) -> str:
    """One bar per band of probability, mirrored around the centre line: above it, the people with the outcome; below it,
    those without. Moving the threshold recolours the bars into the four groups of the confusion matrix; it never moves them."""
    width, height, left, right, top, bottom = 760, 490, 54, 28, 30, 56
    mid = (top + height - bottom) / 2
    half = mid - top - 2
    n_pos, n_neg = sum(pos) or 1, sum(neg) or 1
    tallest = max(max(pos) / n_pos, max(neg) / n_neg) or 1
    x = lambda v: left + v * (width - left - right)
    cut = round(t * 100)
    band = (width - left - right) / 100
    parts = [f'<rect class="wash" x="{x(t):.1f}" y="{top}" width="{x(1) - x(t):.1f}" height="{height - top - bottom}"/>']
    for tick in share_ticks(tallest):
        for sign in (-1, 1):
            yy = mid + sign * half * tick / tallest
            parts.append(f'<line class="grid-line" x1="{left}" x2="{width - right}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                         f'<text x="{left - 7}" y="{yy + 4:.1f}" text-anchor="end">{m11.fmt_pct(tick, 0)}</text>')
    for i in range(100):
        for count, total, sign, tone_flagged in ((pos[i], n_pos, -1, "good"), (neg[i], n_neg, 1, "bad")):
            if count:
                tone = tone_flagged if i >= cut else ("bad" if tone_flagged == "good" else "good")
                h = max(1.2, half * (count / total) / tallest)
                yy = mid - h - .6 if sign < 0 else mid + .6
                parts.append(f'<rect class="b-{tone}" x="{x(i / 100) + .5:.1f}" y="{yy:.1f}" width="{band - 1:.2f}" height="{h:.1f}" rx=".8"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{mid}" y2="{mid}"/>')
    for i in range(100):
        if pos[i] or neg[i]:
            parts.append(f'<rect class="hit" x="{x(i / 100):.1f}" y="{top}" width="{band:.2f}" height="{height - top - bottom}">'
                         f'<title>Probability {i / 100:.2f} to {(i + 1) / 100:.2f}: {pos[i]:,} {esc(w["pos"])}, {neg[i]:,} {esc(w["neg"])}</title></rect>')
    base = height - bottom
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for v in (0, .25, .5, .75, 1):
        parts.append(f'<text x="{x(v):.1f}" y="{base + 16}" text-anchor="middle">{v:.2f}</text>')

    # labels: the threshold and the 0.50 reference sit on opposite sides so they never collide
    tx = x(t)
    if t != DEFAULT_T:
        ref_side = "start" if t < DEFAULT_T else "end"
        parts.append(f'<line class="ref-line" x1="{x(DEFAULT_T):.1f}" x2="{x(DEFAULT_T):.1f}" y1="{top - 6}" y2="{base}"/>'
                     f'<text x="{x(DEFAULT_T) + (5 if t < DEFAULT_T else -5):.1f}" y="{top - 11}" text-anchor="{ref_side}">{DEFAULT_T:.2f}</text>')
    anchor = "end" if (t < DEFAULT_T and tx - left > 120) or tx > width - right - 120 else "start"
    parts.append(f'<line class="t-line" x1="{tx:.1f}" x2="{tx:.1f}" y1="{top - 6}" y2="{base}"/>'
                 f'<text class="strong" x="{tx + (-6 if anchor == "end" else 6):.1f}" y="{top - 11}" text-anchor="{anchor}">threshold {t:.2f}</text>')

    # the row under the ticks: which side is flagged, and the axis title on whichever side is free
    row = base + 38
    if tx - left > 90:
        parts.append(f'<text class="lab" x="{tx - 6:.1f}" y="{row}" text-anchor="end">‹ not flagged</text>')
    if width - right - tx > 70:
        parts.append(f'<text class="lab" x="{tx + 6:.1f}" y="{row}">flagged ›</text>')
    title = "probability from the model"
    parts.append(f'<text x="{width - right if t < .45 else left}" y="{row}" text-anchor="{"end" if t < .45 else "start"}">{title}</text>')

    parts.append(f'<text class="lab" x="{width - right - 6}" y="{top + 15}" text-anchor="end">Actual: {esc(w["pos"])} ↑</text>')
    parts.append(f'<text class="lab" x="{width - right - 6}" y="{base - 7}" text-anchor="end">Actual: {esc(w["neg"])} ↓</text>')
    return (f'<div class="swipe">Swipe sideways to see the whole figure.</div><div class="scroll"><svg class="figure figure-wide" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Test probabilities for each actual outcome, split at threshold {t:.2f}">{"".join(parts)}</svg></div>')


def roc_figure(p, y, t: float, m_now: dict, m_ref: dict) -> str:
    """The ROC curve from m11.roc. The curve belongs to the model; the threshold only picks a point on it."""
    size, left, right, top, bottom = 400, 56, 18, 14, 50
    plot_w, plot_h = size - left - right, size - top - bottom
    fx = lambda v: left + v * plot_w
    fy = lambda v: top + (1 - v) * plot_h
    roc = m11.roc(p, y)
    kept = [(roc["fpr"][0], roc["tpr"][0])]
    for a, b in zip(roc["fpr"][1:], roc["tpr"][1:]):  # one point per screen pixel is enough
        if abs(a - kept[-1][0]) >= .004 or abs(b - kept[-1][1]) >= .004:
            kept.append((a, b))
    kept.append((roc["fpr"][-1], roc["tpr"][-1]))
    line = "L".join(f"{fx(a):.1f} {fy(b):.1f}" for a, b in kept)
    parts = []
    for v in (0, .25, .5, .75, 1):
        parts.append(f'<line class="grid-line" x1="{fx(0)}" x2="{fx(1)}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/>'
                     f'<line class="grid-line" x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{fy(0)}" y2="{fy(1)}"/>'
                     f'<text x="{fx(0) - 7}" y="{fy(v) + 4:.1f}" text-anchor="end">{m11.fmt_pct(v, 0)}</text>'
                     f'<text x="{fx(v):.1f}" y="{fy(0) + 16}" text-anchor="middle">{m11.fmt_pct(v, 0)}</text>')
    parts.append(f'<line class="axis" x1="{fx(0)}" y1="{fy(0)}" x2="{fx(1)}" y2="{fy(1)}"/>'
                 f'<text x="{fx(.6):.1f}" y="{fy(.53):.1f}" transform="rotate(-44 {fx(.6):.1f} {fy(.53):.1f})" text-anchor="middle">coin flip</text>')
    parts.append(f'<path class="curve-area" d="M{line}L{fx(1)} {fy(0)}Z"/><path class="curve" d="M{line}"/>')

    def point(m: dict, cls: str, text: str, above: bool) -> str:
        a, b = m["fpr"], m["sensitivity"]
        if a is None or b is None:
            return ""
        px, py = fx(a), fy(b)
        if py < top + 16:
            above = False
        elif py > fy(0) - 16:
            above = True
        anchor, dx = ("end", -10) if a > .62 else ("start", 10)
        return (f'<g><circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="13"><title>{esc(text)}: sensitivity {m11.fmt_pct(b)}, '
                f'false positive rate {m11.fmt_pct(a)}</title></circle><circle class="{cls}" cx="{px:.1f}" cy="{py:.1f}" r="5.5" pointer-events="none"/>'
                f'<text class="strong" x="{px + dx:.1f}" y="{py + (-9 if above else 17):.1f}" text-anchor="{anchor}" pointer-events="none">{esc(text)}</text></g>')

    same = t == DEFAULT_T or (m_now["fpr"] == m_ref["fpr"] and m_now["sensitivity"] == m_ref["sensitivity"])
    if not same:
        now_above = (m_now["sensitivity"] or 0) >= (m_ref["sensitivity"] or 0)
        parts.append(point(m_ref, "op-ref", f"{DEFAULT_T:.2f}", not now_above))
        parts.append(point(m_now, "op-now", f"{t:.2f}", now_above))
    else:
        parts.append(point(m_now, "op-now", f"{t:.2f}", True))
    parts.append(f'<text class="lab" x="{fx(.5):.1f}" y="{size - 8}" text-anchor="middle">False positive rate (FPR)</text>'
                 f'<text class="lab" x="14" y="{fy(.5):.1f}" text-anchor="middle" transform="rotate(-90 14 {fy(.5):.1f})">Sensitivity (TPR)</text>')
    return f'<svg class="figure figure-roc" viewBox="0 0 {size} {size}" role="img" aria-label="Test ROC curve with the threshold marked">{"".join(parts)}</svg>'


def goal_threshold(p, y, goal: float):
    """The highest slider threshold whose test sensitivity reaches the goal, or None.
    Sensitivity only falls as the threshold rises, so the first match from the top is the highest."""
    return next((t for t in reversed(STEPS) if m11.metrics(m11.confusion(p, y, t))["sensitivity"] >= goal), None)


def set_threshold(t: float) -> None:
    st.session_state["threshold"] = t


def move_slider(p, y) -> None:
    goal = st.session_state["goal"]
    t = goal_threshold(p, y, goal) if goal is not None else None
    if t is not None:
        set_threshold(t)


def sensitivity_goal(p, y, w: dict, label: str) -> None:
    """Design choice: the lead picks how many of the people with the outcome to find, and sees what that costs."""
    ppl = w["people"]
    goal = st.radio(f"Sensitivity goal: find at least this share of the {ppl} who {w['who_pos']}", GOALS, key="goal",
                    horizontal=True, format_func=lambda g: "No goal" if g is None else m11.fmt_pct(g, 0),
                    on_change=move_slider, args=(p, y))
    if goal is None:
        return
    t = goal_threshold(p, y, goal)
    if t is None:
        st.warning(f"No threshold from 0.01 to 0.99 lets {label} find {m11.fmt_pct(goal, 0)} of the {ppl} who {w['who_pos']}.")
        return
    c, base = m11.confusion(p, y, t), m11.confusion(p, y, DEFAULT_T)
    m, mb = m11.metrics(c), m11.metrics(base)
    show(f'<p class="goal-read"><strong>To find at least {m11.fmt_pct(goal, 0)}</strong>, the highest threshold that works for {esc(label)} is '
            f'<strong>{t:.2f}</strong>. It catches {c["tp"]:,} of the {m["positives"]:,} {ppl} who {w["who_pos"]} '
            f'({m11.fmt_pct(m["sensitivity"])}), flags {m["flagged_per_100"]:.1f} of every 100 {ppl}, and gives '
            f'<strong>{per_100(c["fp"], m["n"]):.1f} false alarms per 100 {ppl}</strong> '
            f'(at {DEFAULT_T:.2f}: {per_100(base["fp"], mb["n"]):.1f}). Of the {ppl} it flags, '
            f'{num(m11.fmt_pct(m["precision"]))} {w["who_pos"]}.'
            f'<small>{TEST} · {esc(label)} · a goal met on the {TEST.lower()} {ppl} may not be met on new {ppl}</small></p>')
    if round(st.session_state.get("threshold", DEFAULT_T), 2) != t:
        st.button(f"Move the slider to {t:.2f}", on_click=set_threshold, args=(t,))


def live_readout(base: dict, now: dict, mb: dict, mn: dict, ppl: str) -> str:
    """The three numbers the slider trades, right beside it."""
    items = [("Sensitivity", m11.fmt_pct(mn["sensitivity"]), m11.fmt_pct(mb["sensitivity"])),
             (f"Flagged per 100 {ppl}", fmt_trade(mn["flagged_per_100"], "per100"), fmt_trade(mb["flagged_per_100"], "per100")),
             (f"False alarms per 100 {ppl}", fmt_trade(per_100(now["fp"], mn["n"]), "per100"), fmt_trade(per_100(base["fp"], mb["n"]), "per100"))]
    return '<div class="live">' + "".join(f'<div><span>{name}</span><strong>{num(value)}</strong><small>at {DEFAULT_T:.2f}: {ref}</small></div>'
                                          for name, value, ref in items) + "</div>"


def threshold(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    p, y = m11.preds(res, model)
    show(f'<p class="prose">The model gives each {w["person"]} a probability. The <strong>threshold</strong> is a separate decision: where to draw '
            f'the line. {ppl.capitalize()} at or above it are flagged. Moving the threshold changes who is flagged; the '
            f'model, its probabilities and how it ranks {ppl} stay the same ({TEST} AUROC '
            f'{m11.fmt_num(m11.auroc(p, y))} at every threshold).</p>')
    with st.container(key="deck"):
        show(f'<div class="deck-head"><span class="fig-no">Controls</span><h3>Threshold controls</h3><span class="chips">{chip(TEST)}</span></div>')
        left, right = st.columns([5, 6], gap="large")
        with right:
            sensitivity_goal(p, y, w, label)
        with left:
            st.session_state.setdefault("threshold", DEFAULT_T)
            t = round(st.slider("Threshold: flag when the probability is at least", 0.01, 0.99, step=0.01, format="%.2f", key="threshold"), 2)
            base, now = m11.confusion(p, y, DEFAULT_T), m11.confusion(p, y, t)
            mb, mn = m11.metrics(base), m11.metrics(now)
            show(live_readout(base, now, mb, mn, ppl))

    rows = []
    for name, get, kind in TRADE_ROWS:
        a, b = get(base, mb), get(now, mn)
        rows.append((name.format(ppl=ppl), [fmt_trade(a, kind), fmt_trade(b, kind), fmt_change(a, b, kind)]))
    table = data_table(["", f"At {DEFAULT_T:.2f} ({TEST})", f"At {t:.2f} ({TEST})", "Change"], rows, seps=(3, 6))

    if now == base:
        verdict = f"Move the slider to see what a different threshold gains and gives up compared with {DEFAULT_T:.2f}."
    else:
        flagged = more_fewer(mn["flagged_per_100"] - mb["flagged_per_100"], f"{ppl} per 100", 1)
        caught = more_fewer(now["tp"] - base["tp"], f"of the {mb['positives']:,} who {w['who_pos']}")
        alarms = more_fewer(now["fp"] - base["fp"], "false alarms")
        alarms_100 = fmt_change(per_100(base["fp"], mb["n"]), per_100(now["fp"], mn["n"]), "per100")
        verdict = (f"<strong>Compared with {DEFAULT_T:.2f}, a threshold of {t:.2f}</strong> flags {flagged}, catches {caught}, "
                   f"and gives {alarms} ({alarms_100} per 100 {ppl}).")
    same = ""
    if len(set(p)) == 1:
        same = (f'<div class="callout warn"><span class="tag">Note</span>{esc(label)} gives every {w["person"]} the same probability '
                f'({m11.fmt_num(p[0])}), so any threshold flags either everyone or no one.</div>')
    caution = (f'<div class="callout warn"><span class="tag">Caution</span>You are choosing this threshold while looking at the {TEST.lower()} {ppl}, '
               f'so treat its {TEST.lower()} sensitivity as likely optimistic for new {ppl}.</div>')

    pos, neg = probability_bands(key, model, stamp(key))
    legend = (f'<ul class="legend"><li><i class="sw good"></i><span>Right decision: caught <b>{now["tp"]:,}</b>, correctly not flagged <b>{now["tn"]:,}</b></span></li>'
              f'<li><i class="sw bad"></i><span>Wrong decision: missed <b>{now["fn"]:,}</b>, false alarms <b>{now["fp"]:,}</b></span></li></ul>')
    bands = panel(f"Who falls on each side of the threshold ({esc(label)})", legend + threshold_figure(pos, neg, t, w), fig="Fig. 2.1",
                  note=f"Each bar is a 0.01-wide band of probability. Above the centre line: the {mb['positives']:,} {ppl} who {w['who_pos']}. "
                       f"Below it: the {mb['negatives']:,} {ppl} who {w['who_neg']}. Bar height is the share of that group, so the two can be "
                       "compared although one is larger. Moving the threshold recolours the bars; it never moves them.")
    curve = panel("Every threshold this model offers", roc_figure(p, y, t, mn, mb), fig="Fig. 2.2",
                  note=f"Each point on this ROC curve is one threshold. The curve belongs to the model and does not change; the threshold only picks a "
                       f"point on it. {TEST} AUROC {m11.fmt_num(m11.auroc(p, y))} is how often {w['a_person']} who {w['one_pos']} got a higher probability "
                       f"than one who {w['one_neg']}. It describes ranking, not accuracy, and it does not choose a threshold.")
    matrices = "".join(panel(f"At {tt:.2f}: {esc(label)}", matrix(cc, w), fig=no, note="Rows = actual. Columns = predicted.")
                       for tt, cc, no in ((DEFAULT_T, base, "Fig. 2.3"), (t, now, "Fig. 2.4")))
    show('<div class="stack">'
            + same
            + f'<div class="fig-row">{bands}{curve}</div>'
            + f'<div class="grid">{matrices}</div>'
            + f'<div class="callout">{verdict}</div>'
            + panel(f"What moving from {DEFAULT_T:.2f} to {t:.2f} gains and gives up ({esc(label)})", table, fig="Tab. 2.1", body_class="",
                    note=f"{mb['n']:,} {ppl}, {mb['positives']:,} of whom {w['who_pos']}.")
            + caution
            + "</div>")


# ---- 3. Training vs. Test ------------------------------------------------------
def auroc_table(s: dict, model: str, tr: dict, te: dict, ppl: str) -> str:
    """Training and test AUROC side by side: each row is both a dot pair and the numbers, all from m11.auroc_summary."""
    lo, hi = .45, 1.0
    at = lambda v: (v - lo) / (hi - lo) * 100
    ticks = [.5, .6, .7, .8, .9, 1.0]
    grid = "".join(f'<i class="db-grid{" coin" if v == .5 else ""}" style="left:{at(v):.2f}%"></i>' for v in ticks)
    axis = "".join(f'<span style="left:{at(v):.2f}%">{v:.1f}</span>' for v in ticks)
    rows = []
    for m, v in s.items():
        name, sel = esc(m11.MODEL_LABEL[m]), m == model
        marks = ""
        if v["train"] is not None and v["test"] is not None:
            a, b = sorted((at(v["train"]), at(v["test"])))
            marks = (f'<i class="db-link" style="left:{a:.2f}%;width:{b - a:.2f}%"></i>'
                     f'<i class="db-dot train" style="left:{at(v["train"]):.2f}%" title="{name}: {TRAIN} AUROC {m11.fmt_num(v["train"])}"></i>'
                     f'<i class="db-dot test" style="left:{at(v["test"]):.2f}%" title="{name}: {TEST} AUROC {m11.fmt_num(v["test"])}"></i>')
        mark = '<span class="tag-sel">(selected)</span>' if sel else ""
        rows.append(f'<tr class="{"sel" if sel else ""}"><th scope="row">{name}{mark}</th><td class="db-cell"><div class="db-track">{grid}{marks}</div></td>'
                    f'<td>{num(m11.fmt_num(v["train"]))}</td><td>{num(m11.fmt_num(v["test"]))}</td><td>{num(m11.fmt_num(v["gap"]))}</td></tr>')
    return (f'<div class="scroll"><table class="dt"><thead><tr><th scope="col">Model</th><th scope="col"><div class="db-axis">{axis}</div></th>'
            f'<th scope="col">{TRAIN} AUROC<small>{tr["n"]:,} {ppl} it learned from</small></th>'
            f'<th scope="col">{TEST} AUROC<small>{te["n"]:,} {ppl} it never saw</small></th>'
            f'<th scope="col">Gap<small>training − test</small></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def train_vs_test(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    s = m11.auroc_summary(res)
    tr, te = res["train"], res["test"]

    def best(part: str) -> str:
        return max((m for m in s if s[m][part] is not None), key=lambda m: s[m][part])

    bt, bs = best("train"), best("test")
    wide = max((m for m in s if s[m]["gap"] is not None), key=lambda m: s[m]["gap"])
    v = s[model]
    legend = (f'<ul class="legend"><li><i class="sw dot train"></i>{TRAIN} AUROC</li><li><i class="sw dot test"></i>{TEST} AUROC</li>'
              '<li><i class="sw gap"></i>Gap between them</li></ul>')
    intro = (f'<p class="prose">Each model learned from the {TRAIN.lower()} {ppl}. It is judged on the {TEST.lower()} {ppl} it never '
             f'saw, because those are the closest thing to new {ppl}. A model that memorised its {TRAIN.lower()} '
             f'{ppl} looks almost perfect on them and much worse on the {TEST.lower()} {ppl}: a large gap between '
             'the two is the warning sign.</p>'
             f'<p class="prose"><strong>AUROC</strong> is the chance that a randomly chosen {w["person"]} who {w["one_pos"]} got a higher probability '
             f'than one who {w["one_neg"]} (0.5 is no better than a coin flip; 1.0 is perfect ranking). It describes '
             'ranking, not accuracy, and it does not choose a threshold.</p>')
    cohort = "".join([
        tile(f"{TRAIN} {ppl}: the models learned from these", f"{tr['n']:,}", f"{tr['n_pos']:,} of them {w['who_pos']}."),
        tile(f"{TEST} {ppl}: never seen by the models", f"{te['n']:,}", f"{te['n_pos']:,} of them {w['who_pos']}."),
    ])
    verdicts = "".join([
        tile(f"Best on its {TRAIN.lower()} {ppl}", esc(m11.MODEL_LABEL[bt]), f"{TRAIN} AUROC {m11.fmt_num(s[bt]['train'])}"),
        tile(f"Best on the {TEST.lower()} {ppl}", esc(m11.MODEL_LABEL[bs]), f"{TEST} AUROC {m11.fmt_num(s[bs]['test'])}"),
        tile("Largest gap", esc(m11.MODEL_LABEL[wide]), f"Gap {m11.fmt_num(s[wide]['gap'])} (training − test)"),
    ])
    reading =(f'<div class="callout"><strong>{esc(label)}:</strong> on the {tr["n"]:,} {TRAIN.lower()} {ppl} it learned from, its {TRAIN} AUROC is '
               f'{m11.fmt_num(v["train"])}; on the {te["n"]:,} {TEST.lower()} {ppl} it never saw, its {TEST} AUROC is '
               f'{m11.fmt_num(v["test"])} (gap {m11.fmt_num(v["gap"])}). Only the {TEST.lower()} result shows what to '
               f'expect on new {ppl}.</div>')
    show('<div class="stack">'
            + f'<div class="intro">{intro}</div>'
            + panel("How much each AUROC rests on", f'<div class="tiles">{cohort}</div>', parts=(TRAIN, TEST), fig="Tab. 3.1", body_class="")
            + panel("AUROC of every model", legend + auroc_table(s, model, tr, te, ppl), parts=(TRAIN, TEST), fig="Fig. 3.1",
                    note="The axis starts just below 0.5, a coin flip. A long amber line is the warning sign: the model ranks the "
                         f"{ppl} it learned from far better than {ppl} it never saw.")
            + panel("What to take from this", f'<div class="tiles">{verdicts}</div>{reading}', parts=(TRAIN, TEST), fig="Tab. 3.2", body_class="")
            + "</div>")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.html(STYLE)
show(where=st.sidebar, body=f'<div class="brand">{capsid_svg(2, 0.32, 0.2, 0.16, 3.2, 2.1)}<div><b>JUDGE THE MODEL</b><span>evaluation console</span></div></div>')
section = st.sidebar.radio("Analysis module", SECTIONS,
                           captions=["Accuracy, and who is really caught", "What moving the line gains and gives up", "Spot a model that memorised"])
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
if res is None:
    st.error(f"results/{key}.json is missing: run `python -m analysis`, then commit and push `results/`.")
    st.stop()
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
versions = res.get("versions") or {}
created = res.get("created")
show(where=st.sidebar, body=f'<div class="record"><h4>Run record</h4><dl><dt>Exported</dt><dd>{esc(created[:16].replace("T", " ") + " UTC" if created else "not recorded")}</dd>'
                f'<dt>scikit-learn</dt><dd>{esc(versions.get("scikit-learn", "not recorded"))}</dd><dt>LightGBM</dt><dd>{esc(versions.get("lightgbm", "not recorded"))}</dd></dl>'
                '<p>Every number is computed with m11.py from the exported results/ files. This app never retrains a model.</p></div>')

masthead(res, key, model)
recommendations()
module_header(section, WORDS[key])
if model in res["models"]:
    RENDER[section](res, key, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
