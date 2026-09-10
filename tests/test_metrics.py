"""Metric tests against synthetic distributions with known answers.

Write these before the GPU exists. A metrics bug found now costs twenty
minutes; the same bug found on 25 Sep invalidates every number in the write-up.

Each test below is a stub: fill the body when you implement the corresponding
function in ``src/metrics.py``. They are marked ``skip`` rather than left to
raise, so an unimplemented module does not look like a broken one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="fill in as src/metrics.py is implemented")


def test_auroc_separable(separable_scores):
    """Well-separated classes should score close to 1.0."""


def test_auroc_random(random_scores):
    """Pure noise should score close to 0.5."""


def test_partial_auroc_is_normalised(random_scores):
    """Normalised pAUC on noise is ~0.5, not ~max_fpr/2.

    The most likely bug in the primary metric: forgetting to divide by
    ``max_fpr`` leaves the value bounded above by ``max_fpr`` and not
    comparable to the anchor paper's numbers.
    """


def test_partial_auroc_bounded_by_auroc(separable_scores):
    """Partial AUROC over a sub-region should not exceed the full AUROC."""


def test_roc_curve_handles_ties(tied_scores):
    """Tied scores must form one threshold step, not be broken arbitrarily.

    Breaking ties optimistically inflates the curve, and the text readout
    produces ties constantly.
    """


def test_safety_at_budget_respects_budget(separable_scores):
    """The calibrated threshold should flag ~1% of clean items, not ~1% of all."""


def test_unique_value_count_on_text_readout(tied_scores):
    """A four-valued score array should report exactly four distinct values."""


def test_aggregate_ensemble_keeps_fraction():
    """Averaging 5, 6 and 8 must yield 6.333..., never 6.

    The fractional part is the resolution M4 is trained to inherit; rounding it
    here would silently defeat ADR-0005.
    """
