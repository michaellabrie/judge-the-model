"""Shared metric helpers for the M11 "Judge the Model" activity.

Every number in your app and in the analysis comes from these functions. They are ported from
the M11 lecture app (web/src/data.ts).

    import m11
    res = m11.load("diabetes")                 # results/diabetes.json, or None if missing
    p, y = m11.preds(res, "logreg")            # test-set probabilities and labels
    p, y = m11.preds(res, "logreg", "train")   # training-set probabilities and labels
    c = m11.confusion(p, y, 0.5)               # {"tp", "fn", "fp", "tn"}; flag when p >= threshold
    m11.metrics(c)                             # sensitivity, specificity, ... (None = undefined)
    m11.roc(p, y)                              # ROC points and AUROC
    m11.calibration(p, y)                      # 10 equal-size bins
    m11.auroc_summary(res)                     # training and test AUROC per model, and the gap

Undefined values (for example precision when no one is flagged, 0 ÷ 0) are None here and
null in JSON. Show them as "undefined", never as 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DATASETS = ["readmission", "diabetes"]
MODELS = ["baseline", "logreg", "knn", "rf", "lgbm"]
FITTED = ["logreg", "knn", "rf", "lgbm"]
MODEL_LABEL = {
    "baseline": "Prevalence baseline",
    "logreg": "Logistic regression",
    "knn": "k-nearest neighbours",
    "rf": "Random forest",
    "lgbm": "Gradient boosting",
}
DATASET_LABEL = {"readmission": "30-day readmission", "diabetes": "Diabetes screening"}
PARTITION_LABEL = {"train": "Training", "test": "Test"}

# Thresholds can come from sliders (0.1 + 0.2 ≠ 0.3), so p within 1e-9 below t still counts as flagged.
EPS = 1e-9


# ---------------------------------------------------------------- loading
def load(name: str, folder: Path | str | None = None) -> dict | None:
    """results/<name>.json, e.g. "diabetes" or "decisions". None if it doesn't exist yet."""
    path = Path(folder or RESULTS) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def preds(res: dict, model: str, part: str = "test") -> tuple[list[float], list[int]]:
    """One model's probabilities and the true labels, on the test set (default) or the training set."""
    return res["models"][model][f"{part}_p"], res[part]["y"]


# ---------------------------------------------------------------- the confusion matrix and its metrics
def confusion(p, y, threshold: float) -> dict:
    """Counts at one threshold. A patient is flagged (predicted positive) when p >= threshold."""
    tp = fn = fp = tn = 0
    for pi, yi in zip(p, y):
        flag = pi >= threshold - EPS
        if yi == 1:
            if flag:
                tp += 1
            else:
                fn += 1
        elif flag:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _ratio(a: float, b: float):
    return None if b == 0 else a / b


def metrics(c: dict) -> dict:
    """Every metric from the four counts. Denominators, in words:
    sensitivity (TPR) = TP ÷ (TP + FN): of the patients who had the outcome, the share flagged
    specificity      = TN ÷ (TN + FP): of the patients who did not, the share not flagged
    precision        = TP ÷ (TP + FP): of the patients flagged, the share who had the outcome
    FPR              = FP ÷ (FP + TN): of the patients who did not, the share flagged (1 − specificity)
    F1               = 2TP ÷ (2TP + FP + FN): the harmonic mean of precision and sensitivity
    """
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    n = tp + fn + fp + tn
    flagged = tp + fp
    return {
        "n": n,
        "positives": tp + fn,
        "negatives": fp + tn,
        "flagged": flagged,
        "flagged_per_100": _ratio(100 * flagged, n),
        "prevalence": _ratio(tp + fn, n),
        "accuracy": _ratio(tp + tn, n),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "precision": _ratio(tp, tp + fp),
        "fpr": _ratio(fp, fp + tn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def majority_accuracy(y) -> float | None:
    """Accuracy of "the most common class for everyone"."""
    n = len(y)
    if n == 0:
        return None
    pos = sum(1 for v in y if v == 1)
    return max(pos, n - pos) / n


# ---------------------------------------------------------------- ROC and AUROC
def _desc_order(p) -> list[int]:
    """Indices by descending probability; ties by index, so every run gives the same order."""
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def auroc(p, y) -> float | None:
    """AUROC: the chance that a random positive gets a higher probability than a random negative (ties count ½).
    It describes ranking, not accuracy. None if only one class is present."""
    n = len(p)
    idx = sorted(range(n), key=lambda i: (p[i], i))
    n_pos = 0
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and p[idx[j + 1]] == p[idx[i]]:
            j += 1
        mid = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            if y[idx[k]] == 1:
                n_pos += 1
                rank_sum += mid
        i = j + 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc(p, y) -> dict:
    """The ROC curve: one (FPR, TPR) point per distinct threshold, from flagging no one (0, 0) to everyone (1, 1).
    thr[k] is the threshold for point k (flag p >= thr[k]); thr[0] is infinity."""
    order = _desc_order(p)
    n_pos = sum(1 for v in y if v == 1)
    n_neg = len(y) - n_pos
    fpr, tpr, thr = [0.0], [0.0], [math.inf]
    tp = fp = 0
    k = 0
    while k < len(order):
        v = p[order[k]]
        while k < len(order) and p[order[k]] == v:
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
            k += 1
        fpr.append(fp / n_neg if n_neg else 0.0)
        tpr.append(tp / n_pos if n_pos else 0.0)
        thr.append(v)
    return {"fpr": fpr, "tpr": tpr, "thr": thr, "auroc": auroc(p, y)}


# ---------------------------------------------------------------- calibration
def calibration(p, y, bins: int = 10) -> list[dict]:
    """Calibration plot data: patients sorted by probability and cut into `bins` equal-size groups.
    For each: the mean predicted probability and the observed share with the outcome.
    On a well-calibrated model the two are close (the points sit near the diagonal)."""
    n = len(p)
    idx = sorted(range(n), key=lambda i: (p[i], i))
    groups: list[list[int]] = [[] for _ in range(bins)]
    for r, i in enumerate(idx):
        groups[min(bins - 1, (r * bins) // n)].append(i)
    out = []
    for g in groups:
        if not g:
            continue
        sp = sy = 0.0
        lo, hi = math.inf, -math.inf
        for i in g:
            sp += p[i]
            sy += y[i]
            lo = min(lo, p[i])
            hi = max(hi, p[i])
        out.append({"lo": lo, "hi": hi, "mean_pred": sp / len(g), "observed": sy / len(g), "n": len(g)})
    return out


# ---------------------------------------------------------------- training vs test
def auroc_summary(res: dict) -> dict:
    """For every model in a results file: AUROC on the training rows it learned from, AUROC on the
    test rows it never saw, and the gap (training − test). A big gap means the model memorised its
    training patients (overfitting); the test AUROC is the one to expect on new patients."""
    out = {}
    for m in res["models"]:
        train, test = auroc(*preds(res, m, "train")), auroc(*preds(res, m, "test"))
        out[m] = {"train": train, "test": test, "gap": None if train is None or test is None else train - test}
    return out


# ---------------------------------------------------------------- formatting
def fmt_pct(v, d: int = 1) -> str:
    """0.1234 → "12.3%"; None → "undefined"."""
    return "undefined" if v is None else f"{v * 100:.{d}f}%"


def fmt_num(v, d: int = 3) -> str:
    return "undefined" if v is None else f"{v:.{d}f}"
