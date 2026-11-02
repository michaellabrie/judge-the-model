"""Starter app for Streamlit Community Cloud: three plain sections that read results/ and show correct numbers.

Run locally:  bash run_app.sh  (your agent writes it; it runs streamlit run streamlit_app.py)
Build each section up to its required minimum (see the activity website). Keep every number
coming from m11.py, and label every table and plot "Training" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def load(name: str):
    return m11.load(name)


def starter_note(text: str) -> None:
    st.info(f"Starter display: {text} Build this section up to its required minimum.")


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
def beyond_accuracy(res: dict, model: str) -> None:
    starter_note("the four counts at threshold 0.50, and accuracy.")
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, 0.5)
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{**{k.upper(): v for k, v in c.items()}, "Accuracy": m11.fmt_pct(m11.metrics(c)["accuracy"])}]))


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(res: dict, model: str) -> None:
    starter_note("metrics at 0.50 only.")
    p, y = m11.preds(res, model)
    m = m11.metrics(m11.confusion(p, y, 0.5))
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{"Sensitivity": m11.fmt_pct(m["sensitivity"]), "Specificity": m11.fmt_pct(m["specificity"]),
                            "Precision": m11.fmt_pct(m["precision"]), "Flagged per 100": f"{m['flagged_per_100']:.1f}"}]))


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    starter_note("training and test AUROC of every model, in a table.")
    s = m11.auroc_summary(res)
    rows = [{"Model": m11.MODEL_LABEL[m], f"{TRAIN} AUROC": m11.fmt_num(v["train"]), f"{TEST} AUROC": m11.fmt_num(v["test"])}
            for m, v in s.items()]
    st.caption(f"{TRAIN} and {TEST} · AUROC on the rows each model learned from, and on rows it never saw")
    st.table(pd.DataFrame(rows).set_index("Model"))


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"]:
    RENDER[section](res, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
