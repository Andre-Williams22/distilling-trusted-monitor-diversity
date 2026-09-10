"""Paired significance tests and confidence intervals.

**Every comparison in this project is paired.** All five arms score the same
items, so arm-to-arm differences are correlated and the unpaired tests are both
wrong and needlessly conservative here.

Sample size is the binding constraint. pAUC integrates over the clean items in
test, so roughly 64 negatives define the metric region while expected effects
are +0.03 to +0.09. Intervals will be wide; the job of this module is to report
that honestly rather than hide it.

Like ``metrics``, this is pure numpy and fully testable on a laptop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        estimate: The observed difference or value.
        low: Lower bound.
        high: Upper bound.
        level: Coverage, e.g. ``0.95``.
    """

    estimate: float
    low: float
    high: float
    level: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly above or below zero."""
        raise NotImplementedError


def delong_auroc_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, Interval]:
    """DeLong test for two **correlated** AUROCs on the same items.

    The right test for "does M1 beat M0", because both arms scored the same
    1,288 test items and their errors are correlated.

    One caveat that bites here: DeLong assumes scores can be ranked, and is
    unreliable when many items tie. The text readout produces only a handful of
    distinct values, so heavy ties are expected there. This is part of why the
    continuous logprob readout is primary (ADR-0006) — but when reporting the
    secondary text readout, check the tie fraction before trusting the p-value.

    Args:
        scores_a: First arm's scores.
        scores_b: Second arm's scores, same items in the same order.
        labels: 1 for backdoored, 0 for clean.

    Returns:
        ``(p_value, interval)`` for ``auroc_a - auroc_b``.

    Raises:
        ValueError: If the arrays disagree in length or a class is absent.
    """
    raise NotImplementedError


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 20260909,
) -> Interval:
    """Compute a paired bootstrap CI for a difference in any statistic (pAUC's test).

    Resample **items**, not scores, and evaluate both arms on the same resampled
    indices every time. That is what makes it paired: it preserves the
    correlation between arms and estimates the interval for the *difference*
    rather than for each arm separately.

    Resample within class — positives from positives, negatives from negatives —
    so every resample keeps the 50% attack rate and enough negatives to define
    the pAUC region. A naive resample can otherwise draw a set with too few
    clean items for the region to exist at all.

    Args:
        scores_a: First arm's scores.
        scores_b: Second arm's scores, same items in the same order.
        labels: 1 for backdoored, 0 for clean.
        statistic: Callable taking ``(scores, labels)`` and returning a scalar,
            e.g. ``metrics.partial_auroc``.
        n_resamples: Bootstrap draws.
        level: Coverage.
        seed: Fixed for reproducibility.

    Returns:
        Interval for ``statistic(a) - statistic(b)``.
    """
    raise NotImplementedError


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 20260909,
) -> Interval:
    """Bootstrap CI for a single arm's statistic.

    Used for the "recovery" test: H2 asks whether a distilled arm recovers a
    given fraction of M2's gain, which needs an interval on each arm's own
    value, not only on differences.

    Args:
        scores: One arm's scores.
        labels: 1 for backdoored, 0 for clean.
        statistic: Callable taking ``(scores, labels)``.
        n_resamples: Bootstrap draws.
        level: Coverage.
        seed: Fixed for reproducibility.

    Returns:
        Interval for the statistic.
    """
    raise NotImplementedError


def recovered_fraction(
    baseline: float,
    ceiling: float,
    distilled: float,
) -> float:
    """Compute the fraction of the ensemble's gain a distilled arm recovers (H2, H3).

    ``(distilled - baseline) / (ceiling - baseline)`` — how much of M2's gain
    over M0 a 1x arm keeps.

    Two failure modes the caller must handle rather than paper over. If the
    ceiling did not clear the baseline the denominator is near zero and the
    ratio is meaningless — that is the "M2 doesn't beat M1" kill criterion
    firing, and it should be reported, not divided through. And the result can
    legitimately be negative, meaning the distilled arm scored *below* baseline;
    do not clamp it, since that outcome promotes H4 to the headline.

    Args:
        baseline: M0's metric value.
        ceiling: M2's metric value.
        distilled: M3's or M4's metric value.

    Returns:
        The recovered fraction. May exceed 1 or fall below 0.

    Raises:
        ValueError: If ``ceiling`` and ``baseline`` are too close to divide.
    """
    raise NotImplementedError
