# Brief for the AI coding agent

Read this before changing any code. **You write and debug the code**: the environment setup, `analysis/models.py`, the launcher and the app. **The student makes every decision**, checks your work, and writes their own answers. Build what they ask for without breaking the rules below.

## The project
A take-home activity. A Python workflow trains four classifiers plus a baseline on two health datasets, and the student builds **one app with three sections**, in Streamlit (Python) or Node.js (JavaScript), that helps **a clinical lead** judge whether any model is fit for each job. The clinical lead knows the clinical problem but not the metrics, and must decide whether to use a model.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

| Dataset | Job | Positive class |
|---|---|---|
| `readmission` | Predict readmission within 30 days, at discharge | Readmitted within 30 days |
| `diabetes` | Flag adults for follow-up testing; finding positives matters most | Doctor diagnosis or HbA1c ≥ 6.5% |

| Section | After using it, the reader can… |
|---|---|
| 1. Beyond Accuracy | name the positive class, see why accuracy alone misleads, read a confusion matrix as four groups of patients, and compare every model at 0.50 (a confusion matrix each; accuracy, sensitivity/recall/TPR, specificity, FPR, precision, F1) |
| 2. A Threshold for the Job | see that the threshold is a decision separate from the model, and what moving it gains and gives up compared with 0.50 |
| 3. Training vs. Test | see why a model is judged on patients it never saw, and spot a model that memorised its training patients |

Every section has a dataset selector (both datasets) and a model selector (the four classifiers plus a baseline). A **Recommendations** panel at the top shows the lines from `results/decisions.json`. The activity website lists each section's required minimum and reviewer tasks.

## Files
- `analysis/`: the workflow (`python -m analysis`). `data.py` loads the data and supplies `encode()`; `evaluate.py` fits each model on the training rows and scores both sets; `export.py` reads `DECISIONS.md`; `__main__.py` runs it all and writes `results/`. **Don't change these files.** The starter has no `analysis/models.py`: you write it (see Models).
- `m11.py`: the shared metric helpers. `load`, `preds`, `confusion`, `metrics`, `majority_accuracy`, `auroc`, `roc`, `calibration`, `auroc_summary`.
- `web/m11.js`: the same helpers ported to JavaScript for the Node app (`preds`, `confusion`, `metrics`, `majorityAccuracy`, `auroc`, `roc`, `calibration`, `aurocSummary`). Undefined values are `null`.
- `results/<dataset>.json`: for each model, the training and test probabilities and AUROCs. `results/decisions.json`: the exported `DECISIONS.md`.
- The app, in **one** of two forms. Ask the student which one they chose and change only that one:
  - **Streamlit:** `streamlit_app.py`, plain starter sections that already read the results. Hosted on Streamlit Community Cloud.
  - **Node.js:** `web/app.js` (with `web/index.html` and `web/style.css`), the same plain sections. `web/server.mjs` serves `web/` and `results/` for a local preview. Hosted on GitHub Pages as static files from the repository root (the top-level `index.html` redirects to `web/`), so the app must work with no server code: fetch results with relative paths (`../results/<name>.json`) and keep root-relative paths (`/…`) out. No npm packages: Pages has no install step. If the student wants a library, save its browser ES module in `web/` (or import it from a CDN).
- `data/`: the CSVs and the dictionary (`data/README.md`), with a "When is it known?" entry for every column.
- `check.py`: the integrity checks (`python check.py`, or `--stage setup` / `--stage results`).

## What you build
1. **Environment.** In the starter folder, create a virtual environment with `uv` (Python 3.11 or newer) from `requirements.txt`, then run `python check.py --stage setup`. Explain each command before you run it.
2. **Launcher: `run_app.sh`** in the top folder, a Bash script that starts **only the student's chosen app** on their computer:
   - Streamlit: `streamlit run streamlit_app.py`, with the project's Python environment (`.venv/bin/python`, or `.venv/Scripts/python.exe` on Windows; fall back to the active `python`). Node: `node web/server.mjs` (Node 20 or newer; no `npm install`).
   - It moves to its own folder first, checks its prerequisites with a clear message, prints the address to open, and stops with Ctrl+C. It installs nothing and never runs git or publishes anything.
   - Then tell the student: where to run it (a terminal in the project folder; on Windows, Git Bash or WSL, not PowerShell or Command Prompt), the prerequisites, the exact command (`bash run_app.sh`), the address to open, and how to stop it. **Don't run it for them: the student runs the launch command themselves.** Make clear it runs the app only on their computer; publishing is separate (`SETUP.md`).
3. **Models: `analysis/models.py`**, the complete module the workflow imports (see Models below). Explain it, and how to run it, in a few sentences.
4. **Analysis.** Once the student has committed `Before results` in `DECISIONS.md`, run `python -m analysis` and `python check.py --stage results`, and fix any error at its cause. Then explain what was exported to `results/`, which AUROCs are training and which are test, and show the student the check output so they can see whether it passes.
5. **The three app sections**, built to the required minimums on the activity website, in the chosen app only.

## Models: the spec for `analysis/models.py`
Write one function per model that returns a fresh, unfitted `make_pipeline(...)`, and a `MODELS` dict mapping each key to its function, in exactly this order:

| Key | Steps, in order |
|---|---|
| `baseline` | `DummyClassifier(strategy="prior")` (a pipeline is optional) |
| `logreg` | `encode()` → `SimpleImputer(strategy="median")` → `StandardScaler()` → `LogisticRegression(max_iter=5000)` |
| `knn` | `encode()` → `SimpleImputer(strategy="median")` → `StandardScaler()` → `KNeighborsClassifier(n_neighbors=25)` |
| `rf` | `encode()` → `SimpleImputer(strategy="median")` → `RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)` |
| `lgbm` | `encode()` → `LGBMClassifier(random_state=0, verbose=-1, n_jobs=8)` (it handles missing values itself) |

- `MODELS = {"baseline": baseline, "logreg": logreg, "knn": knn, "rf": rf, "lgbm": lgbm}`.
- Import `encode` from `analysis.data`; don't write your own encoder. No other steps, settings or tuning.
- The module only builds models. `analysis/evaluate.py` and `analysis/__main__.py` already fit, score and export them; don't write a second training loop, script or output format.
- `python check.py --stage results` checks every pipeline's steps and settings, and compares each classifier's test AUROC with the instructor's reference run (±0.005).

## Rules for the analysis (do not break these)
1. **Use the supplied `split` column.** Never reshuffle, never make new splits, and never fit a model on `test` rows.
2. **Keep all preprocessing inside a `Pipeline`.** Encoding, imputation and scaling are steps of the model's pipeline, so they learn from the training rows only. Never scale or impute the whole dataset first.
3. **Build the models exactly to the spec above.** Don't change any model or setting, and don't tune anything.
4. Never add a column as a predictor, and never use `row_id`, `split` or `target` as one.

## Rules for the numbers
5. **Compute every number through `m11`** (`m11.py` in Streamlit, `web/m11.js` in Node). Never type, hard-code, estimate or invent a number. If a number isn't in `results/`, say so instead of making one up.
6. The app reads **only** the files in `results/`. It never retrains a model.
7. Undefined values (0 ÷ 0) are `None` (Python) or `null` (JavaScript). Show them as "undefined", never as 0.
8. Label every table and plot **Training** or **Test**. A training result is never a test result.

## Rules for decisions
9. **Never write or edit `DECISIONS.md`, and never write Canvas answers.** These are the student's. Don't choose a model, threshold, recommendation or design choice for them: when one is needed, ask the student.
10. Don't run the analysis until the student says `Before results` is committed.
11. Don't change `results/` by hand. Rerun the workflow instead.

## Wording (do not break these)
- AUROC describes **ranking**: how often a patient with the outcome gets a higher probability than one without. It is **not** accuracy, and it doesn't choose a threshold.
- Training results flatter every model. Never present a training result as what to expect on new patients.
- Describe associations, not causes. Never say a feature causes the outcome, or that changing it would change a patient's risk.
- Plain language for a clinical lead: say each metric's denominator in words ("of the patients who were readmitted…").

## How to work with the student
- **Offer options, not one answer.** When the student asks how to show something, give two or three different designs with a sentence on each, and let them choose. Their design choice must be their own idea, and they record one of your suggestions they turned down.
- After writing app code, explain it in **three sentences**: what it shows, where its numbers come from, and how to check one of them. Then ask the student to run `bash run_app.sh` and try the section's reviewer task themselves.
- Streamlit: list every Python import in `requirements.txt`, so Streamlit Community Cloud can install it. Node: GitHub Pages serves files as they are, so every file the page loads must be committed under `web/` or come from a CDN.
- Never commit passwords, tokens or API keys.

## Before you say you're done
Run `python check.py` and fix anything it reports that is yours to fix (`DECISIONS.md` is the student's). Then remind the student to commit and push, and to open the published app in a private window.
