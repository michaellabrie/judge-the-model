"""Integrity checks for the M11 "Judge the Model" activity.

    python check.py                   everything (run this before you submit)
    python check.py --stage setup     the data and the starter results (step 1)
    python check.py --stage results   + analysis/models.py and the results of `python -m analysis` (step 3)

Each stage includes the ones before it. Every failure says what to fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import m11  # noqa: E402

# Each classifier's test AUROC in the instructor's reference run (reproduce/reference.json).
REFERENCE_TEST_AUROC = {"readmission": {"logreg": 0.6396, "knn": 0.5832, "rf": 0.651, "lgbm": 0.6322},
                        "diabetes": {"logreg": 0.8266, "knn": 0.8124, "rf": 0.8232, "lgbm": 0.8069}}
TOLERANCE = 0.005
STAGES = ["setup", "results", "all"]
AUROC_TOL = 1e-9


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")

    def fail(self, msg: str) -> None:
        print(f"  FAIL  {msg}")
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        print(f"  note  {msg}")
        self.warns.append(msg)

    def check(self, cond: bool, ok: str, fix: str) -> bool:
        (self.ok if cond else self.fail)(ok if cond else fix)
        return cond


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=AUROC_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ---------------------------------------------------------------- the supplied data, read directly
# check.py never relies on analysis/data.py (which you could change): it reads the CSVs itself.
@lru_cache(maxsize=None)
def dictionary() -> dict:
    return json.loads((ROOT / "data" / "datasets.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def table(key: str):
    import pandas as pd
    return pd.read_csv(ROOT / "data" / f"{key}.csv", dtype={"split": str}, keep_default_na=False, na_values=[""])


def supplied_predictors(key: str) -> list[str]:
    return [c["name"] for c in dictionary()[key]["columns"] if c["kind"] != "id"]


def rows(key: str, split: str) -> tuple[list[int], list[int]]:
    """row_id and target of the supplied training or test rows."""
    part = table(key)[table(key).split == split]
    return part.row_id.astype(int).tolist(), part.target.astype(int).tolist()


# ---------------------------------------------------------------- setup
def check_setup(r: Report) -> bool:
    print("Setup")
    if sys.version_info < (3, 11):
        r.fail(f"Python {sys.version.split()[0]} is too old; recreate your environment with uv and Python 3.11 or newer.")
    try:
        import lightgbm, numpy, pandas, sklearn  # noqa: F401, E401
        r.ok("Python packages import")
    except ImportError as e:
        r.fail(f"Package {e.name} is missing; run `uv pip install -r requirements.txt` in your environment.")
        return False
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    good = True
    for key in m11.DATASETS:
        m = manifest["datasets"][key]
        path = ROOT / "data" / m["file"]
        if not r.check(path.exists() and sha256(path) == m["sha256"], f"data/{m['file']} matches its fingerprint",
                       f"data/{m['file']} differs from the supplied file; restore it from the starter zip (never edit the data)."):
            good = False
            continue
        df = table(key)
        train, test = df[df.split == "train"], df[df.split == "test"]
        r.check(len(train) + len(test) == len(df) and df.row_id.is_unique and len(train) == m["n_train"] and len(test) == m["n_test"],
                f"{key}: every row is a training or a test row, never both",
                f"{key}: the split column is wrong; restore data/{m['file']} from the starter zip.")
        excluded = {c["name"] for c in dictionary()[key]["excluded"]}
        try:
            from analysis import data
            used = set(data.predictors(key))
        except Exception:  # noqa: BLE001  (a broken analysis/data.py is reported below)
            used = {"<analysis/data.py does not load>"}
        bad = sorted(used - set(supplied_predictors(key)) | (used & ({"row_id", "split", "target"} | excluded)))
        r.check(not bad, f"{key}: no ID, outcome or leaked column is a predictor",
                f"{key}: {', '.join(bad)} must not be a predictor; restore analysis/data.py from the starter.")
    for key in m11.DATASETS:
        res = m11.load(key)
        if r.check(res is not None, f"results/{key}.json exists",
                   f"results/{key}.json is missing; restore it from the starter zip or run `python -m analysis`."):
            good &= check_results_file(r, key, res, require_all=False)
    from analysis import export
    if export.is_repo():
        r.ok("this folder is a git repository with at least one commit")
    else:
        r.warn("This folder isn't a git repository with a commit yet; follow step 1 on the website (git init, commit, push).")
    return good


def check_results_file(r: Report, key: str, res: dict, require_all: bool) -> bool:
    """Rows and labels match the supplied data; every recomputed AUROC matches the export."""
    ok = True
    for part in ("train", "test"):
        ids, y = rows(key, part)
        ok &= r.check(res.get(part, {}).get("row_id") == ids and res[part].get("y") == y,
                      f"{key}: {m11.PARTITION_LABEL[part].lower()} results cover exactly the {part} rows",
                      f"{key}: results/{key}.json does not match the {part} rows in data/{key}.csv; use the supplied `split` "
                      "column (never reshuffle, never train on test rows) and rerun `python -m analysis`.")
    bad = sorted(set(res.get("features", [])) - set(supplied_predictors(key)))
    ok &= r.check(not bad, f"{key}: only the supplied predictors were used",
                  f"{key}: {', '.join(bad)} were used as predictors; restore analysis/data.py and rerun `python -m analysis`.")
    missing = [m for m in m11.MODELS if m not in res["models"]]
    if require_all:
        ok &= r.check(not missing, f"{key}: results for all {len(m11.MODELS)} models",
                      f"{key}: no results for {', '.join(missing)}; run `python -m analysis`.")
    if not ok:
        return False
    problems = []
    for name, entry in res["models"].items():
        for part in ("train", "test"):
            p, y = m11.preds(res, name, part)
            if len(p) != len(y) or any(v is None or not 0 <= v <= 1 for v in p):
                problems.append(f"{name} does not have exactly one {part} probability per row")
            elif not close(m11.auroc(p, y), entry[f"{part}_auroc"]):
                problems.append(f"{name} {part} AUROC")
    return r.check(not problems, f"{key}: every exported AUROC matches the one recomputed with m11",
                   f"{key}: {'; '.join(problems[:3])} differ from the recomputed values; never edit results by hand, "
                   "rerun `python -m analysis`.")


# ---------------------------------------------------------------- results
SPEC = "see Models in AGENT_BRIEF.md"
# The fixed spec: the steps before the classifier, then the classifier's class and the settings that must not change.
EXPECTED = {
    "baseline": ([], "DummyClassifier", {"strategy": "prior"}),
    "logreg": (["encode", "SimpleImputer", "StandardScaler"], "LogisticRegression", {"max_iter": 5000, "C": 1.0}),
    "knn": (["encode", "SimpleImputer", "StandardScaler"], "KNeighborsClassifier", {"n_neighbors": 25}),
    "rf": (["encode", "SimpleImputer"], "RandomForestClassifier", {"n_estimators": 300, "min_samples_leaf": 3, "random_state": 0}),
    "lgbm": (["encode"], "LGBMClassifier", {"random_state": 0, "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31}),
}


def step_name(step) -> str:
    """A step's class name; the supplied encode() step (a ColumnTransformer with its OneHotEncoder) is "encode"."""
    if type(step).__name__ == "ColumnTransformer":
        enc = [t for _, t, _ in step.transformers if type(t).__name__ == "OneHotEncoder"]
        if enc and enc[0].get_params().get("min_frequency") == 10 and enc[0].get_params().get("handle_unknown") == "infrequent_if_exist":
            return "encode"
    if type(step).__name__ == "SimpleImputer" and step.get_params().get("strategy") != "median":
        return "SimpleImputer(not median)"
    return type(step).__name__


def check_pipelines(r: Report) -> bool:
    from sklearn.pipeline import Pipeline
    try:
        from analysis.models import MODELS
    except ModuleNotFoundError as e:
        if e.name != "analysis.models":
            raise
        r.fail(f"analysis/models.py is missing; ask your agent to write it ({SPEC}).")
        return False
    if not r.check(list(MODELS) == m11.MODELS, f"MODELS lists the {len(m11.MODELS)} models in order",
                   f"MODELS in analysis/models.py must have exactly the keys {', '.join(m11.MODELS)}, in that order ({SPEC})."):
        return False
    ok = True
    for name, (before, cls, params) in EXPECTED.items():
        model = MODELS[name]()
        steps = [s for _, s in model.steps] if isinstance(model, Pipeline) else [model]
        names = [step_name(s) for s in steps[:-1]]
        final = steps[-1]
        got = final.get_params()
        same = names == before and type(final).__name__ == cls and all(got.get(k) == v for k, v in params.items())
        want = " → ".join(["encode()" if b == "encode" else (f'{b}(strategy="median")' if b == "SimpleImputer" else f"{b}()") for b in before]
                          + [f"{cls}({', '.join(f'{k}={v!r}' for k, v in params.items())})"])
        ok &= r.check(same, f"{name}: {' → '.join(before + [cls])} with the fixed settings",
                      f"{name}() must build {want} as one Pipeline, with no other steps or tuning ({SPEC}).")
    return ok


def check_results(r: Report) -> bool:
    print("Results")
    ok = check_pipelines(r)
    for key in m11.DATASETS:
        res = m11.load(key)
        if res is None:
            r.fail(f"results/{key}.json is missing; run `python -m analysis`.")
            ok = False
            continue
        ok &= check_results_file(r, key, res, require_all=True)
        for name, want in REFERENCE_TEST_AUROC[key].items():
            if name not in res["models"]:
                continue
            got = res["models"][name]["test_auroc"]
            ok &= r.check(abs(got - want) <= TOLERANCE,
                          f"{key}: {name} test AUROC {got:.4f} matches the reference {want:.4f} (±{TOLERANCE})",
                          f"{key}: {name} test AUROC {got:.4f} is not within ±{TOLERANCE} of the reference {want:.4f}; "
                          f"make {name}() match the spec and use the supplied data, then rerun `python -m analysis`.")
    return ok


# ---------------------------------------------------------------- everything else
def check_all(r: Report) -> bool:
    print("Submission")
    from analysis import export
    parsed = export.read()
    ok = True
    for key, s in parsed.items():
        ok &= r.check(s["complete"], f'"{export.LABELS[key]}" in DECISIONS.md is complete',
                      f'Fill in "{export.LABELS[key]}" in DECISIONS.md (missing: {", ".join(s["missing"])}).')
    words = export.word_count(parsed)
    if words > export.WORD_LIMIT:
        r.warn(f"Before results in DECISIONS.md has {words} words; the limit is {export.WORD_LIMIT}.")
    d = m11.load("decisions")
    ok &= r.check(d is not None and d.get("sections") == {k: v["answers"] for k, v in parsed.items()},
                  "results/decisions.json matches DECISIONS.md",
                  "results/decisions.json is out of date; run `python -m analysis` again, then commit and push.")
    if not export.is_repo():
        r.fail("This folder isn't a git repository; follow step 1 on the website (git init, commit, push).")
        return False
    dirty = export.git("status", "--porcelain", "--", "results", "DECISIONS.md")
    ok &= r.check(dirty == "", "results/ and DECISIONS.md are committed",
                  "Commit your latest results/ and DECISIONS.md (git add, git commit, git push).")
    if export.unpushed():
        r.warn("Some commits are not on GitHub yet; run `git push`.")
    launchers = [p.name for p in sorted(ROOT.glob("*.sh")) if re.search(r"streamlit run|server\.mjs|npm start", p.read_text(encoding="utf-8"))]
    tracked = [n for n in launchers if export.git("ls-files", "--error-unmatch", n)]
    ok &= r.check(bool(tracked), f"{(tracked or launchers or ['-'])[0]} launches your app and is committed",
                  "Commit a Bash launcher in the top folder (for example run_app.sh) that starts your app; ask your agent to write it "
                  "(see AGENT_BRIEF.md), then run it yourself.")
    first = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0] if (ROOT / "README.md").exists() else ""
    if "http" not in first or "paste your" in first:
        r.warn("Put your public app link on the first line of README.md.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES[:-1], help="check up to this stage only (default: everything)")
    level = STAGES.index(ap.parse_args().stage or "all")
    r = Report()
    from analysis import export
    try:
        check_setup(r)
        if level >= 1:
            check_results(r)
        if level >= 2:
            check_all(r)
    except export.DecisionsError as e:
        r.fail(str(e))
    print()
    if r.fails:
        print(f"{len(r.fails)} check(s) failed. Fix the first one, then run python check.py again.")
        sys.exit(1)
    print("All checks passed." + (f" {len(r.warns)} note(s) above." if r.warns else ""))


if __name__ == "__main__":
    main()
