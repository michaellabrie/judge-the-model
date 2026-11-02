"""The five models, built to the fixed spec in AGENT_BRIEF.md. Each function returns a fresh, unfitted pipeline.

analysis/evaluate.py fits each one on the training rows and scores it on both sets; this module only builds them.
Every preprocessing step (encoding, imputation, scaling) sits inside the pipeline, so it learns from the training rows only.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.data import encode


def baseline():
    """Gives every patient the training prevalence: the reference every real model must beat."""
    return make_pipeline(DummyClassifier(strategy="prior"))


def logreg():
    return make_pipeline(encode(), SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))


def knn():
    return make_pipeline(encode(), SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=25))


def rf():
    return make_pipeline(encode(), SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))


def lgbm():
    """LightGBM handles missing values itself, so it needs no imputer."""
    return make_pipeline(encode(), LGBMClassifier(random_state=0, verbose=-1, n_jobs=8))


MODELS = {"baseline": baseline, "logreg": logreg, "knn": knn, "rf": rf, "lgbm": lgbm}
