"""The analysis workflow. Run from the starter folder:

    python -m analysis

For each dataset it fits every model on the training rows, predicts the training and test rows,
and writes results/<dataset>.json. It also copies your DECISIONS.md answers to
results/decisions.json, so rerun it after you edit DECISIONS.md. The models come from
analysis/models.py, which your AI agent writes to the spec in AGENT_BRIEF.md.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import m11
from analysis import data, evaluate, export

RESULTS = export.RESULTS


def stop(msg: str) -> None:
    print(f"Stopped: {msg}")
    sys.exit(1)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def versions() -> dict:
    return {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm"]}}


def write(name: str, obj: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")


def train_and_test(key: str, models: dict, created: str | None = None, versions_: dict | None = None) -> dict:
    """Fit each model on the training rows of one dataset and score it on both sets."""
    X_train, y_train = data.rows(key, "train")
    X_test, y_test = data.rows(key, "test")
    out = {
        "dataset": key, "created": created, "versions": versions_, "features": data.predictors(key),
        "train": {"n": len(y_train), "n_pos": sum(y_train), "row_id": data.ids(key, "train"), "y": y_train},
        "test": {"n": len(y_test), "n_pos": sum(y_test), "row_id": data.ids(key, "test"), "y": y_test},
        "models": {},
    }
    for name, build in models.items():
        out["models"][name] = r = evaluate.fit_and_score(build, X_train, y_train, X_test, y_test)
        print(f"  {key:<12} {name:<9} training AUROC {m11.fmt_num(r['train_auroc'])}   test AUROC {m11.fmt_num(r['test_auroc'])}", flush=True)
    return out


def models() -> dict:
    """MODELS from analysis/models.py: name → a function that builds that model's pipeline."""
    try:
        from analysis.models import MODELS
    except ModuleNotFoundError as e:
        if e.name != "analysis.models":
            raise
        stop("analysis/models.py is missing; ask your agent to write it to the spec in AGENT_BRIEF.md.")
    if list(MODELS) != m11.MODELS:
        stop(f"MODELS in analysis/models.py must have exactly these keys, in this order: {', '.join(m11.MODELS)}.")
    return MODELS


def main() -> None:
    MODELS = models()
    try:
        decisions = export.read()
    except export.DecisionsError as e:
        stop(str(e))
    if not decisions["before_results"]["complete"]:
        print('Note: "Before results" in DECISIONS.md is still empty; fill it in before you look at the results.')
    for key in data.DATASETS:
        write(key, train_and_test(key, MODELS, now(), versions()))
    out = export.export()
    print(f"Wrote results/<dataset>.json for {len(data.DATASETS)} datasets, and results/decisions.json.")
    if out["words"] > export.WORD_LIMIT:
        print(f"Note: {out['words']} words in Before results, over the {export.WORD_LIMIT}-word limit.")
    print("Next: python check.py, then commit and push results/.")


if __name__ == "__main__":
    main()
