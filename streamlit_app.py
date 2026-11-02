"""Judge the Model: three sections that help a clinical lead judge whether any classifier is fit for each job.

Run locally:  bash run_app.sh  (it runs streamlit run streamlit_app.py)
The app reads only results/ and computes every number with m11.py. Every table is labelled Training or Test.
"""

import pandas as pd
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
        "person": "patient", "people": "patients",
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
        "person": "adult", "people": "adults",
        "who_pos": "had diabetes", "who_neg": "did not have diabetes",
        "one_pos": "had diabetes", "one_neg": "did not",
        "pos": "diabetes", "neg": "no diabetes",
        "always_no": "you do not have diabetes",
    },
}

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def _load(name: str, mtime: float):
    return m11.load(name)


def load(name: str):
    """Reload whenever `python -m analysis` rewrites the file."""
    path = m11.RESULTS / f"{name}.json"
    return _load(name, path.stat().st_mtime if path.exists() else 0.0)


# ---- shared displays -----------------------------------------------------------
def fmt_metric(name: str, v) -> str:
    return m11.fmt_num(v) if name == "F1" else m11.fmt_pct(v)


def per_100(k: int, n: int):
    """A count from m11.confusion, per 100 people screened."""
    return None if n == 0 else 100 * k / n


def matrix(c: dict, w: dict, detailed: bool = True) -> pd.DataFrame:
    """Rows = what actually happened, columns = what the model predicted, positives first
    (scikit-learn's confusion_matrix uses the opposite order)."""
    if detailed:
        cells = [[f"{c['tp']:,} caught (true positives)", f"{c['fn']:,} missed (false negatives)"],
                 [f"{c['fp']:,} false alarms (false positives)", f"{c['tn']:,} correctly not flagged (true negatives)"]]
    else:
        cells = [[f"TP {c['tp']:,}", f"FN {c['fn']:,}"], [f"FP {c['fp']:,}", f"TN {c['tn']:,}"]]
    return pd.DataFrame(cells, index=[f"Actual: {w['pos']}", f"Actual: {w['neg']}"],
                        columns=["Predicted: flagged", "Predicted: not flagged"])


def explained(c: dict, w: dict) -> pd.DataFrame:
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
        out.append({"Metric": name, "Value": fmt_metric(name, v), "What it means": words})
    return pd.DataFrame(out).set_index("Metric")


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ------------------------------------------------------
def beyond_accuracy(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, DEFAULT_T)
    m = m11.metrics(c)
    always = m11.majority_accuracy(y)

    st.markdown(f"**The job:** {w['job']}  \n**Positive class** (what the models look for): {w['positive']}.  \n"
                f"A {w['person']} is **flagged** when the model's probability is at least {DEFAULT_T:.2f}.")
    st.caption(w["framing"])

    st.subheader("Why accuracy alone misleads")
    st.caption(f"{TEST} · {m['n']:,} {ppl} the models never saw")
    a, b, d = st.columns(3)
    a.metric(f"Prevalence ({TEST})", m11.fmt_pct(m["prevalence"]),
             help=f"Of the {m['n']:,} {ppl}, the share who {w['who_pos']}.")
    b.metric(f'"Always no" accuracy ({TEST})', m11.fmt_pct(always),
             help=f'Accuracy of telling every {w["person"]} "{w["always_no"]}": the most common outcome for everyone.')
    d.metric(f"{label} accuracy at {DEFAULT_T:.2f} ({TEST})", m11.fmt_pct(m["accuracy"]))
    st.markdown(f'Telling every {w["person"]} "{w["always_no"]}" is right for {m11.fmt_pct(always)} of them, and catches '
                f"**none** of the {m['positives']:,} who {w['who_pos']}. {label} at {DEFAULT_T:.2f} is right for "
                f"{m11.fmt_pct(m['accuracy'])}, and catches **{c['tp']:,} of {m['positives']:,}**. Because most {ppl} "
                f"{w['who_neg']}, a model can score a high accuracy while missing most of the {ppl} who {w['who_pos']}. "
                "Count who is caught and who is missed, not just accuracy.")

    st.subheader(f"{label} at {DEFAULT_T:.2f}: four groups of {ppl}")
    st.caption(f"{TEST} · rows = what actually happened, columns = what the model predicted")
    st.table(matrix(c, w))
    st.caption(f"{TEST} · {label}, threshold {DEFAULT_T:.2f}")
    st.table(explained(c, w))

    st.subheader(f"Every model at {DEFAULT_T:.2f}")
    st.caption(f"{TEST} · {m['n']:,} {ppl} · each matrix: rows = actual, columns = predicted")
    rows, undefined = [], False
    names = [n for n in m11.MODELS if n in res["models"]]
    for col, name in zip(st.columns(len(names)), names):
        cc = m11.confusion(*m11.preds(res, name), DEFAULT_T)
        mm = m11.metrics(cc)
        with col:
            st.markdown(f"**{m11.MODEL_LABEL[name]}** ({TEST})")
            st.table(matrix(cc, w, detailed=False))
        row = {"Model": m11.MODEL_LABEL[name] + (" (selected)" if name == model else ""),
               "Caught": f"{cc['tp']:,} of {mm['positives']:,}", "Flagged": f"{mm['flagged']:,}"}
        for metric, k in [("Accuracy", "accuracy"), ("Sensitivity (recall, TPR)", "sensitivity"), ("Specificity", "specificity"),
                          ("FPR", "fpr"), ("Precision", "precision"), ("F1", "f1")]:
            row[metric] = fmt_metric(metric, mm[k])
            undefined |= mm[k] is None
        rows.append(row)
    st.caption(f"{TEST} · every model at threshold {DEFAULT_T:.2f}")
    st.table(pd.DataFrame(rows).set_index("Model"))
    note = f'"Always no" accuracy on the same {ppl}, for comparison: {m11.fmt_pct(always)}.'
    if undefined:
        note += f' "undefined" means 0 ÷ 0: for example, a model that flags no one has no flagged {ppl} to take a share of.'
    st.caption(note)


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
    st.markdown(f"**To find at least {m11.fmt_pct(goal, 0)}**, the highest threshold that works for {label} is **{t:.2f}**. "
                f"It catches {c['tp']:,} of the {m['positives']:,} {ppl} who {w['who_pos']} "
                f"({m11.fmt_pct(m['sensitivity'])}), flags {m['flagged_per_100']:.1f} of every 100 {ppl}, and gives "
                f"**{per_100(c['fp'], m['n']):.1f} false alarms per 100 {ppl}** "
                f"(at {DEFAULT_T:.2f}: {per_100(base['fp'], mb['n']):.1f}). Of the {ppl} it flags, "
                f"{m11.fmt_pct(m['precision'])} {w['who_pos']}.")
    st.caption(f"{TEST} · {label} · a goal met on the {TEST.lower()} {ppl} may not be met on new {ppl}")
    if round(st.session_state.get("threshold", DEFAULT_T), 2) != t:
        st.button(f"Move the slider to {t:.2f}", on_click=set_threshold, args=(t,))


def threshold(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    p, y = m11.preds(res, model)
    st.markdown(f"The model gives each {w['person']} a probability. The **threshold** is a separate decision: where to draw "
                f"the line. {ppl.capitalize()} at or above it are flagged. Moving the threshold changes who is flagged; the "
                f"model, its probabilities and how it ranks {ppl} stay the same ({TEST} AUROC "
                f"{m11.fmt_num(m11.auroc(p, y))} at every threshold).")
    sensitivity_goal(p, y, w, label)
    st.session_state.setdefault("threshold", DEFAULT_T)
    t = st.slider("Threshold: flag when the probability is at least", 0.01, 0.99, step=0.01, format="%.2f", key="threshold")
    if len(set(p)) == 1:
        st.info(f"{label} gives every {w['person']} the same probability ({m11.fmt_num(p[0])}), so any threshold "
                "flags either everyone or no one.")

    base, now = m11.confusion(p, y, DEFAULT_T), m11.confusion(p, y, t)
    mb, mn = m11.metrics(base), m11.metrics(now)
    for col, tt, cc in zip(st.columns(2), (DEFAULT_T, t), (base, now)):
        with col:
            st.markdown(f"**At {tt:.2f}**")
            st.caption(f"{TEST} · {label} · rows = actual, columns = predicted")
            st.table(matrix(cc, w))

    rows = []
    for name, get, kind in TRADE_ROWS:
        a, b = get(base, mb), get(now, mn)
        rows.append({"": name.format(ppl=ppl), f"At {DEFAULT_T:.2f} ({TEST})": fmt_trade(a, kind),
                     f"At {t:.2f} ({TEST})": fmt_trade(b, kind), "Change": fmt_change(a, b, kind)})
    st.caption(f"{TEST} · {label} · {mb['n']:,} {ppl}, {mb['positives']:,} of whom {w['who_pos']}")
    st.table(pd.DataFrame(rows).set_index(""))

    if now == base:
        st.markdown(f"Move the slider to see what a different threshold gains and gives up compared with {DEFAULT_T:.2f}.")
    else:
        flagged = more_fewer(mn["flagged_per_100"] - mb["flagged_per_100"], f"{ppl} per 100", 1)
        caught = more_fewer(now["tp"] - base["tp"], f"of the {mb['positives']:,} who {w['who_pos']}")
        alarms = more_fewer(now["fp"] - base["fp"], "false alarms")
        alarms_100 = fmt_change(per_100(base["fp"], mb["n"]), per_100(now["fp"], mn["n"]), "per100")
        st.markdown(f"**Compared with {DEFAULT_T:.2f}, a threshold of {t:.2f}** flags {flagged}, catches {caught}, "
                    f"and gives {alarms} ({alarms_100} per 100 {ppl}).")
    st.caption(f"You are choosing this threshold while looking at the {TEST.lower()} {ppl}, so treat its {TEST.lower()} "
               f"sensitivity as likely optimistic for new {ppl}.")


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, key: str, model: str) -> None:
    w, ppl, label = WORDS[key], WORDS[key]["people"], m11.MODEL_LABEL[model]
    s = m11.auroc_summary(res)
    tr, te = res["train"], res["test"]
    st.markdown(f"Each model learned from the {TRAIN.lower()} {ppl}. It is judged on the {TEST.lower()} {ppl} it never "
                f"saw, because those are the closest thing to new {ppl}. A model that memorised its {TRAIN.lower()} "
                f"{ppl} looks almost perfect on them and much worse on the {TEST.lower()} {ppl}: a large gap between "
                "the two is the warning sign.")
    st.markdown(f"**AUROC** is the chance that a randomly chosen {w['person']} who {w['one_pos']} got a higher probability "
                f"than one who {w['one_neg']} (0.5 is no better than a coin flip; 1.0 is perfect ranking). It describes "
                "ranking, not accuracy, and it does not choose a threshold.")

    a, b = st.columns(2)
    a.metric(f"{TRAIN} {ppl} (the models learned from these)", f"{tr['n']:,}",
             help=f"{tr['n_pos']:,} of them {w['who_pos']}.")
    b.metric(f"{TEST} {ppl} (never seen by the models)", f"{te['n']:,}",
             help=f"{te['n_pos']:,} of them {w['who_pos']}.")
    st.caption(f"{TRAIN}: {tr['n_pos']:,} of {tr['n']:,} {w['who_pos']} · "
               f"{TEST}: {te['n_pos']:,} of {te['n']:,} {w['who_pos']}")

    rows = [{"Model": m11.MODEL_LABEL[m] + (" (selected)" if m == model else ""),
             f"{TRAIN} AUROC ({tr['n']:,} {ppl} it learned from)": m11.fmt_num(v["train"]),
             f"{TEST} AUROC ({te['n']:,} {ppl} it never saw)": m11.fmt_num(v["test"]),
             "Gap (training − test)": m11.fmt_num(v["gap"])} for m, v in s.items()]
    st.caption(f"{TRAIN} and {TEST} · AUROC of every model")
    st.table(pd.DataFrame(rows).set_index("Model"))

    def best(part: str) -> str:
        return max((m for m in s if s[m][part] is not None), key=lambda m: s[m][part])

    bt, bs = best("train"), best("test")
    wide = max((m for m in s if s[m]["gap"] is not None), key=lambda m: s[m]["gap"])
    st.markdown(f"Best on its {TRAIN.lower()} {ppl}: **{m11.MODEL_LABEL[bt]}** ({TRAIN} AUROC {m11.fmt_num(s[bt]['train'])}).  \n"
                f"Best on the {TEST.lower()} {ppl}: **{m11.MODEL_LABEL[bs]}** ({TEST} AUROC {m11.fmt_num(s[bs]['test'])}).  \n"
                f"Largest gap: **{m11.MODEL_LABEL[wide]}** ({m11.fmt_num(s[wide]['gap'])}).")

    v = s[model]
    st.markdown(f"**{label}:** on the {tr['n']:,} {TRAIN.lower()} {ppl} it learned from, its {TRAIN} AUROC is "
                f"{m11.fmt_num(v['train'])}; on the {te['n']:,} {TEST.lower()} {ppl} it never saw, its {TEST} AUROC is "
                f"{m11.fmt_num(v['test'])} (gap {m11.fmt_num(v['gap'])}). Only the {TEST.lower()} result shows what to "
                f"expect on new {ppl}.")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
if res is None:
    st.error(f"results/{key}.json is missing: run `python -m analysis`, then commit and push `results/`.")
    st.stop()
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} {TRAIN.lower()} rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} {TEST.lower()} rows ({res['test']['n_pos']:,} positive).")
if model in res["models"]:
    RENDER[section](res, key, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
