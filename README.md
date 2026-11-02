**App:** _paste your public app link here_

# Judge the Model

An app that helps a clinical lead judge whether any classifier is fit for each job: 30-day readmission and diabetes screening. It reads the results my Python analysis exported to `results/`.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

Submit the link to **this repository** on Canvas, plus your four Canvas answers.

## Quick start
Your AI agent writes the code; you make the decisions, check the results and write your own answers. Open this folder in your agent and give it the starting prompt from the activity website (it begins "Read AGENT_BRIEF.md").

| Step | Your agent | You |
|---|---|---|
| 1 | Sets up the environment; writes `run_app.sh` | Run `python check.py --stage setup` and `bash run_app.sh` |
| 2 | — | Fill in `Before results` in `DECISIONS.md`; commit and push |
| 3 | Writes `analysis/models.py`; runs `python -m analysis` | Run `python check.py --stage results` and read the output |
| 4 | Builds the three app sections | Run the app and do each reviewer task |
| 5 | Reruns `python -m analysis` once you have written them | Fill in `Recommendations` and `Design choice and AI use`; commit and push |
| any | — | `python check.py` |

## What's here
| File | What it is |
|---|---|
| `DECISIONS.md` | Your decisions: what matters for each job, your recommendations, and one design choice |
| `AGENT_BRIEF.md` | What your AI agent builds, and its rules (it reads this first) |
| `SETUP.md` | How to run and publish on Streamlit Community Cloud or GitHub Pages |
| `analysis/` | The workflow; your agent writes `analysis/models.py` |
| `m11.py` | The shared metric helpers: every number comes from these |
| `streamlit_app.py`, `requirements.txt` | Your app if you choose Streamlit, hosted on Streamlit Community Cloud |
| `web/`, `package.json` | Your app if you choose Node.js (`web/app.js`, with the metric helpers in `web/m11.js`), hosted on GitHub Pages (`index.html` and `.nojekyll` at the top level are for Pages) |
| `results/` | What the analysis exports; the app reads only these files |
| `data/` | The two datasets, the dictionary (`data/README.md`) and `manifest.json` |
| `check.py` | Integrity checks: `python check.py` must pass before you submit |
| `run_app.sh` | Starts your app on your computer (your agent writes it; you run `bash run_app.sh`) |
| `prompts/` | Optional: your AI prompts, one file per session |
