"""Validate the paired tests against independent references.

DeLong is the one piece of numerics in this project that is easy to get subtly
wrong and impossible to eyeball. These tests pin it two ways: its AUROCs must
match scikit-learn exactly, and its analytic interval must agree with a paired
bootstrap that shares none of its code.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.stats import (
    Interval,
    _delong_components,
    bootstrap_ci,
    delong_auroc_test,
    paired_bootstrap,
    recovered_fraction,
)

N_PER_CLASS = 400


@pytest.fixture
def labels() -> np.ndarray:
    """Balanced labels, matching this project's 50% attack rate."""
    return np.concatenate(
        [np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)]
    )


@pytest.fixture
def arms(labels) -> dict[str, np.ndarray]:
    """Three arms: a strong one, a correlated slightly-worse one, a weak one."""
    rng = np.random.default_rng(0)
    strong = np.concatenate(
        [rng.normal(0, 1, N_PER_CLASS), rng.normal(1.0, 1, N_PER_CLASS)]
    )
    correlated = strong + rng.normal(0, 0.6, 2 * N_PER_CLASS)
    weak = np.concatenate(
        [rng.normal(0, 1, N_PER_CLASS), rng.normal(0.2, 1, N_PER_CLASS)]
    )
    return {"strong": strong, "correlated": correlated, "weak": weak}


@pytest.fixture
def tied_scores(labels) -> np.ndarray:
    """Integer 0-10 scores, imitating the text readout's heavy ties."""
    rng = np.random.default_rng(3)
    raw = np.concatenate(
        [rng.normal(0, 1, N_PER_CLASS), rng.normal(1.0, 1, N_PER_CLASS)]
    )
    return np.round(np.clip(raw, -3, 3) * 1.5 + 5).clip(0, 10)


def test_delong_auroc_matches_sklearn(arms, labels):
    """DeLong's AUROC must equal scikit-learn's to floating-point precision."""
    for scores in arms.values():
        aucs, _ = _delong_components(np.vstack([scores, scores]), labels)
        assert aucs[0] == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_delong_auroc_matches_sklearn_under_ties(tied_scores, labels):
    """Midranks must handle ties the same way scikit-learn does.

    The text readout produces only a handful of distinct values, so ties are
    the normal case there, not an edge case.
    """
    aucs, _ = _delong_components(np.vstack([tied_scores, tied_scores]), labels)
    assert len(np.unique(tied_scores)) <= 11
    assert aucs[0] == pytest.approx(roc_auc_score(labels, tied_scores), abs=1e-12)


def test_delong_detects_a_large_difference(arms, labels):
    """A clearly better arm should separate decisively."""
    p_value, interval = delong_auroc_test(arms["strong"], arms["weak"], labels)
    assert p_value < 1e-6
    assert interval.estimate > 0
    assert interval.excludes_zero


def test_delong_on_identical_scores_is_not_significant(arms, labels):
    """Zero variance must not become a division, and p must be 1."""
    p_value, interval = delong_auroc_test(arms["strong"], arms["strong"], labels)
    assert p_value == 1.0
    assert interval.estimate == pytest.approx(0.0)
    assert not interval.excludes_zero


def test_bootstrap_interval_agrees_with_delong(arms, labels):
    """Two independent methods must land on the same interval.

    The bootstrap shares no code with the analytic estimator, so agreement here
    is real cross-validation rather than a tautology.
    """
    statistic = lambda scores, lab: roc_auc_score(lab, scores)  # noqa: E731
    boot = paired_bootstrap(
        arms["strong"], arms["weak"], labels, statistic, n_resamples=2000
    )
    _, delong = delong_auroc_test(arms["strong"], arms["weak"], labels)

    assert boot.estimate == pytest.approx(delong.estimate, abs=1e-12)
    assert boot.low == pytest.approx(delong.low, abs=0.01)
    assert boot.high == pytest.approx(delong.high, abs=0.01)


def test_bootstrap_is_deterministic(arms, labels):
    """The same seed must give the same interval, or results are not reproducible."""
    statistic = lambda scores, lab: roc_auc_score(lab, scores)  # noqa: E731
    first = bootstrap_ci(arms["strong"], labels, statistic, n_resamples=500)
    second = bootstrap_ci(arms["strong"], labels, statistic, n_resamples=500)
    assert first == second


def test_bootstrap_preserves_class_balance(arms, labels):
    """Every replicate must keep both classes populated.

    Stratified resampling is what guarantees enough clean items for the pAUC
    region to exist in each draw; a naive resample can empty it.
    """

    def counts(scores, lab):
        assert int((lab == 1).sum()) == N_PER_CLASS
        assert int((lab == 0).sum()) == N_PER_CLASS
        return roc_auc_score(lab, scores)

    bootstrap_ci(arms["strong"], labels, counts, n_resamples=50)


def test_mismatched_lengths_raise(arms, labels):
    """Silently truncating misaligned arrays would corrupt every paired test."""
    with pytest.raises(ValueError, match="length"):
        delong_auroc_test(arms["strong"][:-1], arms["weak"], labels)


def test_single_class_raises(arms):
    """A split with no clean items has no ROC curve to speak of."""
    with pytest.raises(ValueError, match="both classes"):
        delong_auroc_test(
            arms["strong"], arms["weak"], np.ones(2 * N_PER_CLASS, dtype=int)
        )


def test_recovered_fraction_values():
    """Half, over-recovery and below-baseline must all pass through unclamped."""
    assert recovered_fraction(0.60, 0.70, 0.65) == pytest.approx(0.5)
    assert recovered_fraction(0.60, 0.70, 0.74) == pytest.approx(1.4)
    assert recovered_fraction(0.60, 0.70, 0.58) == pytest.approx(-0.2)


def test_recovered_fraction_rejects_a_flat_ceiling():
    """"M2 doesn't beat M1" must be reported, not divided through."""
    with pytest.raises(ValueError, match="does not clear baseline"):
        recovered_fraction(0.70, 0.7005, 0.72)


def test_interval_excludes_zero():
    """The significance shorthand used throughout the analysis."""
    assert Interval(0.05, 0.01, 0.09, 0.95).excludes_zero
    assert Interval(-0.05, -0.09, -0.01, 0.95).excludes_zero
    assert not Interval(0.05, -0.01, 0.11, 0.95).excludes_zero
