"""Read DECISIONS.md and export your decisions. You don't need to change this file.

DECISIONS.md has three fixed sections. Each holds labelled prompts; your answer is the text
after each label. Keep the headings and labels exactly as they are.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "DECISIONS.md"
RESULTS = ROOT / "results"
WORD_LIMIT = 150

# section key → (heading, [(answer key, label, template text after the label)])
SECTIONS = {
    "before_results": ("Before results", [
        ("readmission", "- readmission:", ""),
        ("diabetes", "- diabetes:", ""),
    ]),
    "recommendations": ("Recommendations", [
        ("readmission", "- readmission:", '<model and threshold, or "none is adequate for this job"> because ...'),
        ("diabetes", "- diabetes:", "..."),
    ]),
    "design": ("Design choice and AI use", [
        ("design_choice", "- design choice:", "<one thing you chose to show, or leave out, for your clinical lead> because ..."),
        ("ai_turned_down", "- AI suggestion I turned down:",
         '<what an AI tool suggested, and why you didn\'t use it, or "no AI tools used">'),
    ]),
}
LABELS = {k: heading for k, (heading, _) in SECTIONS.items()}


class DecisionsError(Exception):
    pass


# ---------------------------------------------------------------- parsing
def _sections(text: str) -> dict[str, str]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1)
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _clean(answer: str, template: str) -> str:
    answer = answer.strip()
    if template and answer.startswith(template):
        answer = answer[len(template):].strip()
    return " ".join(answer.split())


def _filled(answer: str) -> bool:
    if not answer or answer in ("...", "…"):
        return False
    if re.search(r"<(model and threshold|one thing you chose|what an AI tool)", answer):
        return False
    return bool(re.search(r"\w", answer.replace("because ...", "")))


def parse(text: str) -> dict:
    """{section key: {"answers": {key: text}, "complete": bool, "missing": [labels]}}."""
    raw = _sections(text)
    out = {}
    for key, (heading, slots) in SECTIONS.items():
        if heading not in raw:
            raise DecisionsError(f'DECISIONS.md has no "## {heading}" heading; restore it from the template.')
        lines = raw[heading].splitlines()
        starts = []
        for skey, label, template in slots:
            hit = next((i for i, ln in enumerate(lines) if ln.strip().startswith(label)), None)
            if hit is None:
                raise DecisionsError(f'"## {heading}" in DECISIONS.md is missing the label "{label}"; restore it from the template.')
            starts.append(hit)
        answers, missing = {}, []
        for n, (skey, label, template) in enumerate(slots):
            i = starts[n]
            later = [s for s in starts if s > i]
            end = min(later) if later else len(lines)
            first = lines[i].strip()[len(label):]
            ans = _clean("\n".join([first] + lines[i + 1:end]), template)
            answers[skey] = ans if _filled(ans) else ""
            if not answers[skey]:
                missing.append(label.lstrip("- ").rstrip(":"))
        out[key] = {"answers": answers, "complete": not missing, "missing": missing}
    return out


def read(path: Path = DECISIONS) -> dict:
    if not path.exists():
        raise DecisionsError("DECISIONS.md is missing; restore it from the starter.")
    return parse(path.read_text(encoding="utf-8"))


def word_count(parsed: dict) -> int:
    """Words you wrote in Before results (the section with the limit)."""
    return sum(len(a.split()) for a in parsed["before_results"]["answers"].values())


# ---------------------------------------------------------------- git (check.py's reminders only)
def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def is_repo() -> bool:
    return git("rev-parse", "--is-inside-work-tree") == "true" and git("rev-parse", "--verify", "HEAD") is not None


def unpushed() -> bool:
    """True when the branch has commits that aren't on GitHub yet (or has no upstream)."""
    ahead = git("rev-list", "--count", "@{upstream}..HEAD")
    return ahead is None or ahead != "0"


# ---------------------------------------------------------------- export
def export() -> dict:
    parsed = read()
    out = {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "words": word_count(parsed),
        "word_limit": WORD_LIMIT,
        "sections": {k: v["answers"] for k, v in parsed.items()},
        "complete": {k: v["complete"] for k, v in parsed.items()},
        "recommendations": parsed["recommendations"]["answers"],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "decisions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
