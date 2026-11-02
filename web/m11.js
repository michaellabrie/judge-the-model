// Shared metric helpers for the Node app, a line-for-line port of m11.py. Every number in the app comes from these.
// Undefined values (precision when no one is flagged, 0 ÷ 0) are null: show them as "undefined", never as 0.

export const DATASETS = ["readmission", "diabetes"];
export const MODELS = ["baseline", "logreg", "knn", "rf", "lgbm"];
export const FITTED = ["logreg", "knn", "rf", "lgbm"];
export const MODEL_LABEL = {
  baseline: "Prevalence baseline",
  logreg: "Logistic regression",
  knn: "k-nearest neighbours",
  rf: "Random forest",
  lgbm: "Gradient boosting",
};
export const DATASET_LABEL = { readmission: "30-day readmission", diabetes: "Diabetes screening" };

// Thresholds come from sliders (0.1 + 0.2 ≠ 0.3), so p within 1e-9 below t still counts as flagged.
const EPS = 1e-9;

// ---------------------------------------------------------------- loading
export function preds(res, model, part = "test") {
  return [res.models[model][`${part}_p`], res[part].y];
}

// ---------------------------------------------------------------- the confusion matrix and its metrics
export function confusion(p, y, threshold) {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (let i = 0; i < p.length; i++) {
    const flag = p[i] >= threshold - EPS;
    if (y[i] === 1) flag ? tp++ : fn++;
    else flag ? fp++ : tn++;
  }
  return { tp, fn, fp, tn };
}

const ratio = (a, b) => (b === 0 ? null : a / b);

export function metrics(c) {
  const { tp, fn, fp, tn } = c;
  const n = tp + fn + fp + tn;
  const flagged = tp + fp;
  return {
    n,
    positives: tp + fn,
    negatives: fp + tn,
    flagged,
    flagged_per_100: ratio(100 * flagged, n),
    prevalence: ratio(tp + fn, n),
    accuracy: ratio(tp + tn, n),
    sensitivity: ratio(tp, tp + fn),
    specificity: ratio(tn, tn + fp),
    precision: ratio(tp, tp + fp),
    fpr: ratio(fp, fp + tn),
    f1: ratio(2 * tp, 2 * tp + fp + fn),
  };
}

export function majorityAccuracy(y) {
  if (y.length === 0) return null;
  const pos = y.filter((v) => v === 1).length;
  return Math.max(pos, y.length - pos) / y.length;
}

// ---------------------------------------------------------------- ROC and AUROC
const ascOrder = (p) => [...p.keys()].sort((a, b) => p[a] - p[b] || a - b);
const descOrder = (p) => [...p.keys()].sort((a, b) => p[b] - p[a] || a - b);

export function auroc(p, y) {
  const n = p.length;
  const idx = ascOrder(p);
  let nPos = 0, rankSum = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && p[idx[j + 1]] === p[idx[i]]) j++;
    const mid = (i + j) / 2 + 1; // ranks are 1-based
    for (let k = i; k <= j; k++) if (y[idx[k]] === 1) { nPos++; rankSum += mid; }
    i = j + 1;
  }
  const nNeg = n - nPos;
  if (nPos === 0 || nNeg === 0) return null;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function roc(p, y) {
  const order = descOrder(p);
  const nPos = y.filter((v) => v === 1).length;
  const nNeg = y.length - nPos;
  const fpr = [0], tpr = [0], thr = [Infinity];
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; ) {
    const v = p[order[k]];
    while (k < order.length && p[order[k]] === v) {
      y[order[k]] === 1 ? tp++ : fp++;
      k++;
    }
    fpr.push(nNeg ? fp / nNeg : 0);
    tpr.push(nPos ? tp / nPos : 0);
    thr.push(v);
  }
  return { fpr, tpr, thr, auroc: auroc(p, y) };
}

// ---------------------------------------------------------------- calibration
export function calibration(p, y, bins = 10) {
  const n = p.length;
  const idx = ascOrder(p);
  const groups = Array.from({ length: bins }, () => []);
  idx.forEach((i, r) => groups[Math.min(bins - 1, Math.floor((r * bins) / n))].push(i));
  return groups.filter((g) => g.length).map((g) => {
    let sp = 0, sy = 0, lo = Infinity, hi = -Infinity;
    for (const i of g) {
      sp += p[i]; sy += y[i];
      lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]);
    }
    return { lo, hi, mean_pred: sp / g.length, observed: sy / g.length, n: g.length };
  });
}

// ---------------------------------------------------------------- training vs test
export function aurocSummary(res) {
  const out = {};
  for (const m of Object.keys(res.models)) {
    const train = auroc(...preds(res, m, "train"));
    const test = auroc(...preds(res, m, "test"));
    out[m] = { train, test, gap: train === null || test === null ? null : train - test };
  }
  return out;
}

// ---------------------------------------------------------------- formatting
export const fmtPct = (v, d = 1) => (v === null || v === undefined ? "undefined" : `${(v * 100).toFixed(d)}%`);
export const fmtNum = (v, d = 3) => (v === null || v === undefined ? "undefined" : v.toFixed(d));
