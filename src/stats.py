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
from scipy import stats as scipy_stats


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
        return (self.low > 0.0) or (self.high < 0.0)

    def __str__(self) -> str:
        """Render as ``estimate [low, high]`` for tables and logs."""
        return f"{self.estimate:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def _check_inputs(scores: np.ndarray, labels: np.ndarray) -> None:
    """Validate that scores and labels can be compared at all.

    Args:
        scores: Suspicion scores.
        labels: 1 for backdoored, 0 for clean.

    Raises:
        ValueError: If lengths disagree or either class is absent.
    """
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"scores and labels disagree in length: "
            f"{scores.shape[0]} vs {labels.shape[0]}"
        )
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"both classes required, got {n_positive} positive "
            f"and {n_negative} negative"
        )


def _midranks(values: np.ndarray) -> np.ndarray:
    """Rank values, averaging the ranks of ties.

    Ties get the mean of the ranks they collectively occupy. This is what makes
    the AUROC computed below agree with the standard tie-aware definition --
    and ties are the normal case for the text readout, which produces only a
    handful of distinct scores.

    Args:
        values: Values to rank.

    Returns:
        One-based midranks, in the order the values were given.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = values.shape[0]
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _delong_components(
    scores_by_arm: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AUROCs and their covariance matrix (fast DeLong).

    Implements the midrank formulation of Sun and Xu (2014), which computes the
    same quantities as the original DeLong estimator in O(n log n) rather than
    O(n^2). The covariance it returns is what makes the test *paired*: it
    captures how two arms' errors move together on the same items.

    Args:
        scores_by_arm: Shape ``(n_arms, n_items)``.
        labels: 1 for backdoored, 0 for clean.

    Returns:
        ``(aucs, covariance)`` with shapes ``(n_arms,)`` and
        ``(n_arms, n_arms)``.
    """
    positive = labels == 1
    negative = ~positive
    m = int(positive.sum())
    n = int(negative.sum())

    pos_scores = scores_by_arm[:, positive]
    neg_scores = scores_by_arm[:, negative]
    n_arms = scores_by_arm.shape[0]

    rank_pos = np.empty((n_arms, m), dtype=float)
    rank_neg = np.empty((n_arms, n), dtype=float)
    rank_all = np.empty((n_arms, m + n), dtype=float)

    for arm in range(n_arms):
        rank_pos[arm] = _midranks(pos_scores[arm])
        rank_neg[arm] = _midranks(neg_scores[arm])
        rank_all[arm] = _midranks(
            np.concatenate([pos_scores[arm], neg_scores[arm]])
        )

    aucs = rank_all[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Structural components: v01 varies over positives, v10 over negatives.
    v01 = (rank_all[:, :m] - rank_pos) / n
    v10 = 1.0 - (rank_all[:, m:] - rank_neg) / m

    cov_01 = np.cov(v01, ddof=1).reshape(n_arms, n_arms)
    cov_10 = np.cov(v10, ddof=1).reshape(n_arms, n_arms)
    covariance = cov_01 / m + cov_10 / n
    return aucs, covariance


def delong_auroc_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    level: float = 0.95,
) -> tuple[float, Interval]:
    """DeLong test for two **correlated** AUROCs on the same items.

    The right test for "does M1 beat M0", because both arms scored the same
    1,288 test items and their errors are correlated.

    One caveat that bites here: DeLong assumes scores can be ranked, and is
    unreliable when many items tie. The text readout produces only a handful of
    distinct values, so heavy ties are expected there. This is part of why the
    continuous logprob readout is primary (ADR-0006) -- but when reporting the
    secondary text readout, check the tie fraction before trusting the p-value.

    Args:
        scores_a: First arm's scores.
        scores_b: Second arm's scores, same items in the same order.
        labels: 1 for backdoored, 0 for clean.
        level: Coverage for the returned interval.

    Returns:
        ``(p_value, interval)`` for ``auroc_a - auroc_b``.

    Raises:
        ValueError: If the arrays disagree in length or a class is absent.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels = np.asarray(labels).astype(int)
    _check_inputs(scores_a, labels)
    _check_inputs(scores_b, labels)

    aucs, covariance = _delong_components(
        np.vstack([scores_a, scores_b]), labels
    )
    difference = float(aucs[0] - aucs[1])
    variance = float(
        covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    )

    # Identical score vectors give zero variance; the difference is exactly
    # zero and there is nothing to test, so say so rather than dividing by it.
    if variance <= 0.0:
        return 1.0, Interval(difference, difference, difference, level)

    standard_error = float(np.sqrt(variance))
    z = difference / standard_error
    p_value = float(2.0 * scipy_stats.norm.sf(abs(z)))

    critical = float(scipy_stats.norm.ppf(0.5 + level / 2.0))
    interval = Interval(
        estimate=difference,
        low=difference - critical * standard_error,
        high=difference + critical * standard_error,
        level=level,
    )
    return p_value, interval


def _resample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a bootstrap sample of item indices, stratified by class.

    Resampling within class keeps the 50% attack rate in every replicate and
    guarantees enough clean items to define the pAUC region. A naive resample
    over all items can draw a set whose negative count is too small for the
    FPR <= 0.10 region to exist at all, which silently distorts the interval.

    Args:
        labels: 1 for backdoored, 0 for clean.
        rng: Seeded generator.

    Returns:
        Indices into the original arrays, positives followed by negatives.
    """
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    drawn_positive = positive_idx[
        rng.integers(0, positive_idx.size, positive_idx.size)
    ]
    drawn_negative = negative_idx[
        rng.integers(0, negative_idx.size, negative_idx.size)
    ]
    return np.concatenate([drawn_positive, drawn_negative])


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 20260909,
) -> Interval:
    """Compute a paired bootstrap CI for a difference in any statistic.

    This is pAUC's test, since pAUC has no closed-form variance.

    Resample **items**, not scores, and evaluate both arms on the same resampled
    indices every time. That is what makes it paired: it preserves the
    correlation between arms and estimates the interval for the *difference*
    rather than for each arm separately.

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

    Raises:
        ValueError: If the arrays disagree in length or a class is absent.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels = np.asarray(labels).astype(int)
    _check_inputs(scores_a, labels)
    _check_inputs(scores_b, labels)

    observed = float(statistic(scores_a, labels) - statistic(scores_b, labels))

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        idx = _resample_indices(labels, rng)
        resampled_labels = labels[idx]
        differences[draw] = statistic(
            scores_a[idx], resampled_labels
        ) - statistic(scores_b[idx], resampled_labels)

    alpha = 1.0 - level
    low, high = np.quantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(observed, float(low), float(high), level)


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

    Raises:
        ValueError: If the arrays disagree in length or a class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    _check_inputs(scores, labels)

    observed = float(statistic(scores, labels))

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        idx = _resample_indices(labels, rng)
        values[draw] = statistic(scores[idx], labels[idx])

    alpha = 1.0 - level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(observed, float(low), float(high), level)


def recovered_fraction(
    baseline: float,
    ceiling: float,
    distilled: float,
    min_gain: float = 1e-3,
) -> float:
    """Compute the fraction of the ensemble's gain a distilled arm recovers.

    This is the quantity H2 and H3 are stated in:
    ``(distilled - baseline) / (ceiling - baseline)`` -- how much of M2's gain
    over M0 a 1x arm keeps.

    Two failure modes are handled rather than papered over. If the ceiling did
    not clear the baseline the denominator is near zero and the ratio is
    meaningless -- that is the "M2 doesn't beat M1" kill criterion firing, and
    it should be reported, not divided through. And the result can legitimately
    be negative, meaning the distilled arm scored *below* baseline; it is not
    clamped, because that outcome promotes H4 to the headline.

    Args:
        baseline: M0's metric value.
        ceiling: M2's metric value.
        distilled: M3's or M4's metric value.
        min_gain: Smallest ceiling-over-baseline gain worth dividing by.

    Returns:
        The recovered fraction. May exceed 1 or fall below 0.

    Raises:
        ValueError: If ``ceiling`` and ``baseline`` are too close to divide.
    """
    gain = ceiling - baseline
    if gain < min_gain:
        raise ValueError(
            f"ceiling ({ceiling:.4f}) does not clear baseline ({baseline:.4f}) "
            f"by at least min_gain={min_gain}; the recovered fraction is "
            f"undefined. Report the ensemble's failure to beat the baseline "
            f"instead of dividing through it."
        )
    return float((distilled - baseline) / gain)
