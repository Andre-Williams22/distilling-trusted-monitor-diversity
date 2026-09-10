"""Shared fixtures. Everything here runs on the laptop — no GPU, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated classes — AUROC should come back near 1.0.

    Returns:
        ``(scores, labels)`` with 200 clean and 200 backdoored items.
    """
    rng = np.random.default_rng(0)
    clean = rng.normal(0.2, 0.1, 200)
    backdoored = rng.normal(0.8, 0.1, 200)
    scores = np.concatenate([clean, backdoored])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    return scores, labels


@pytest.fixture
def random_scores() -> tuple[np.ndarray, np.ndarray]:
    """Pure noise — AUROC and normalised pAUC should both sit near 0.5.

    The pAUC check is the important one: an unnormalised implementation returns
    roughly ``max_fpr / 2`` here instead of 0.5, so this fixture catches the
    single most likely bug in the metric.

    Returns:
        ``(scores, labels)`` with 200 of each class.
    """
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 400)
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    return scores, labels


@pytest.fixture
def tied_scores() -> tuple[np.ndarray, np.ndarray]:
    """Heavily tied integer scores, imitating the text readout.

    A 7B monitor concentrates its written scores on a handful of values, so
    ties are the normal case for the secondary readout, not an edge case.

    Returns:
        ``(scores, labels)`` drawn from only {0, 3, 5, 8}.
    """
    rng = np.random.default_rng(2)
    clean = rng.choice([0, 3, 5, 8], 200, p=[0.5, 0.3, 0.15, 0.05])
    backdoored = rng.choice([0, 3, 5, 8], 200, p=[0.15, 0.25, 0.35, 0.25])
    scores = np.concatenate([clean, backdoored]).astype(float)
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    return scores, labels
