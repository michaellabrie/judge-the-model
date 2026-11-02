"""Fit a model on the training rows and score it on both sets. You don't need to change this file.

Probabilities are rounded to 4 decimals before any metric is computed, so every number
your app recomputes with m11 matches the exported one exactly.
"""

from __future__ import annotations

import warnings

import numpy as np

import m11

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*X has feature names.*")


def rounded(p) -> list[float]:
    return [float(v) for v in np.round(np.asarray(p, dtype=float), 4)]


def proba(model, X) -> list[float]:
    """The model's estimated probability of the positive class, rounded to 4 decimals."""
    return rounded(model.predict_proba(X)[:, 1])


def fit_and_score(build, X_train, y_train: list[int], X_test, y_test: list[int]) -> dict:
    """Fit a fresh model on the training rows only, then predict both the training and the test rows.

    Training AUROC is scored on the rows the model learned from, so it flatters the model.
    Test AUROC is scored on rows it never saw: the one to expect on new patients."""
    model = build().fit(X_train, np.asarray(y_train))
    p_train, p_test = proba(model, X_train), proba(model, X_test)
    return {"train_p": p_train, "test_p": p_test,
            "train_auroc": m11.auroc(p_train, y_train), "test_auroc": m11.auroc(p_test, y_test)}
