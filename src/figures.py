"""Charts for the results report, drawn with matplotlib.

Two kinds of figure:

- **Per arm** -- a low-FPR ROC chart in the style of the anchor ensemble
  paper (FPR 0-0.10, the arm highlighted, an ensemble's individual members in
  light grey, a dashed random line), and a chart of how its scores split
  between clean and backdoored items.
- **All arms** -- the same ROC chart with every arm together, pAUC with 95%
  intervals per arm, and pAUC against inference cost.

Colours follow the arm, never its rank, so M2 is the same aqua in every chart
and a newly scored arm never repaints the others. The six arm colours pass a
colour-vision-deficiency check as adjacent pairs; three of them are light
against the white background, so every chart also names each arm in text.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from src.config import METRICS  # noqa: E402
from src.metrics import roc_curve  # noqa: E402

if TYPE_CHECKING:
    from src.analysis import ArmScores

ARM_TITLES = {
    "m0": "M0 · single monitor",
    "m1": "M1 · 3 identical samples",
    "m2": "M2 · 3 persona prompts",
    "m3": "M3 · distilled from M1",
    "m4": "M4 · two-term SFT",
    "m5": "M5 · MACA (MV-DPO)",
    "m6": "M6 · MACA (MV-SFT)",
    "m7": "M7 · MACA (MV-KTO)",
    "m8": "M8 · debate, graded target",
}

#: Fixed per arm, in run order of the palette's validated categorical slots.
#: Six slots pass the colour-vision check as adjacent pairs.
ARM_COLORS = {
    "m0": "#2a78d6",
    "m1": "#eb6834",
    "m2": "#1baf7a",
    "m3": "#eda100",
    "m4": "#e87ba4",
    "m5": "#7b52d6",
    "m6": "#8b5a2b",
    "m7": "#c2185b",
    "m8": "#00739e",
}

#: The benchmark every trained arm is measured against. In the faceted ROC it
#: appears in both panels, so it is drawn as a neutral reference rather than a
#: series -- which also keeps each panel's hue set inside the validated size.
BENCHMARK_ARM = "m4"
BENCHMARK_INK = "#4a4a48"

#: Which arms each ROC panel carries. Nine overlapping curves cannot be read:
#: eight of the nine arms sit inside a 0.12-wide pAUC band and three pairs are
#: closer than the line width, so the chart is faceted by question instead.
#: Every set here was checked with the palette validator (CVD, chroma,
#: normal-vision separation and contrast), light and dark.
ROC_PANELS = (
    ("Can a 3x ensemble be distilled to 1x?", ("m0", "m2")),
    ("Does MACA's vote survive distillation?", ("m5", "m7", "m8")),
)

#: The frontier's one job is "1x reaches the 3x ceiling". These arms make that
#: comparison: both ensembles at 3x, the floor, the two distilled arms and the
#: published recipe at 1x. M3 earns its place against M4 -- same recipe, a
#: redundant teacher instead of a diverse one -- so the diversity claim is
#: visible here and not only in the ingredient chart. The remaining arms belong
#: to later sections that have their own figures.
HEADLINE_ARMS = ("m0", "m1", "m2", "m3", "m4", "m5")

#: Diverging pair for the ingredient-cost chart: gains and losses need opposite
#: poles with a neutral zero, never one hue ramped by size.
COST_GAIN = "#00739e"
COST_LOSS = "#eb6834"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MEMBER = "#d3d1ca"
RANDOM = "#b3b1aa"

DPI = 200


def apply_style() -> None:
    """Set the shared look: quiet axes, hairline grid, sans type, light surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": True,
            "legend.edgecolor": AXIS,
            "legend.facecolor": SURFACE,
            "legend.framealpha": 1.0,
            "legend.fontsize": 9,
        }
    )


def _heading(ax: plt.Axes, title: str, subtitle: str) -> None:
    """Put a left-aligned title and a one-line subtitle above a chart.

    Args:
        ax: The axes to title.
        title: Main heading.
        subtitle: Context line in secondary ink.
    """
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.text(
        0, 1.025, subtitle, transform=ax.transAxes,
        fontsize=9, color=INK_SECONDARY, va="bottom",
    )


def _figure_heading(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Title a multi-panel figure, reserving space so nothing overlaps.

    Positions are set in inches from the top edge, so the gap between title,
    subtitle and panels stays the same whatever the figure's height.

    Args:
        fig: The figure.
        title: Main heading.
        subtitle: Context line in secondary ink.
    """
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.22 / height, title, fontsize=13, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.52 / height, subtitle, fontsize=9,
             color=INK_SECONDARY, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.82 / height))


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure to disk and release its memory.

    Args:
        fig: The figure.
        path: Destination ``.png``.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _low_fpr_axes(ax: plt.Axes, max_fpr: float) -> None:
    """Frame an axes as a low-FPR ROC chart with a dashed random line.

    Args:
        ax: The axes.
        max_fpr: Right edge of the chart.
    """
    ax.plot(
        [0, max_fpr], [0, max_fpr], color=RANDOM, linewidth=1.6,
        linestyle=(0, (5, 4)), label="Random", zorder=1,
    )
    ax.set_xlim(0, max_fpr)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(max_fpr / 5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")


def _roc_line(
    ax: plt.Axes, result: ArmScores, **style: Any
) -> None:
    """Draw one ROC curve.

    Args:
        ax: Target axes.
        result: Scores and labels to draw.
        **style: Passed to ``ax.plot``.
    """
    fpr, tpr = roc_curve(result.scores, result.labels)
    ax.plot(fpr, tpr, solid_capstyle="round", solid_joinstyle="round", **style)


def _member_label(result: ArmScores) -> str:
    """Name the grey member lines in a legend."""
    names = list(result.members)
    kind = "personas" if not names[0].startswith("sample") else "samples"
    return f"Individual {kind} (n={len(names)})"


def plot_arm_roc(
    result: ArmScores, summary: dict[str, Any], path: Path, context: str
) -> Path:
    """Draw one arm's low-FPR ROC chart, with its ensemble members in grey.

    Args:
        result: The arm's primary-readout scores.
        summary: Its ``summarize`` row.
        path: Destination file.
        context: Setup and slice description for the subtitle.

    Returns:
        ``path``.
    """
    apply_style()
    max_fpr = METRICS.pauc_max_fpr
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    _low_fpr_axes(ax, max_fpr)

    for index, member in enumerate(result.members.values()):
        _roc_line(
            ax, member, color=MEMBER, linewidth=1.3, zorder=2,
            label=_member_label(result) if index == 0 else None,
        )

    arm = result.arm
    _roc_line(
        ax, result, color=ARM_COLORS[arm], linewidth=2.6, zorder=3,
        label=f"{ARM_TITLES[arm]}   pAUC {summary['pauc']:.3f}",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "Random")
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower right", borderpad=0.8, handlelength=2.6,
    )
    _heading(
        ax,
        f"{ARM_TITLES[arm]} — ROC at low false-positive rates",
        f"AUROC {summary['auroc']:.3f} · pAUC@10% "
        f"{summary['pauc']:.3f} [{summary['pauc_low']:.2f}, {summary['pauc_high']:.2f}]"
        f" · {context}",
    )
    return _save(fig, path)


def _mirrored_histogram(
    ax: plt.Axes,
    scores: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    color: str,
) -> None:
    """Draw backdoored items as bars up and clean items as bars down.

    Heights are the share of each class, so the two halves stay comparable
    even when the classes differ in size.

    Args:
        ax: Target axes.
        scores: Suspicion scores.
        labels: 1 for backdoored, 0 for clean.
        edges: Histogram bin edges.
        color: The arm's colour, used for the backdoored half.
    """
    widths = np.diff(edges)
    centres = edges[:-1] + widths / 2
    halves = ((1, color, 1), (0, MUTED, -1))

    peak = 0.0
    for label, fill, direction in halves:
        counts, _ = np.histogram(scores[labels == label], bins=edges)
        share = 100 * counts / max(counts.sum(), 1)
        peak = max(peak, float(share.max()))
        ax.bar(
            centres, direction * share, width=widths * 0.86, color=fill,
            edgecolor=SURFACE, linewidth=1.0, zorder=2,
        )

    limit = peak * 1.18 + 2
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
    ax.text(0.01, 0.97, "backdoored (above the line)", transform=ax.transAxes, va="top",
            fontsize=9, color=INK_SECONDARY)
    ax.text(0.01, 0.03, "clean (below the line)", transform=ax.transAxes, va="bottom",
            fontsize=9, color=INK_SECONDARY)


def plot_arm_scores(
    primary: ArmScores,
    secondary: ArmScores | None,
    summaries: dict[str, dict[str, Any]],
    path: Path,
    context: str,
) -> Path:
    """Draw how an arm's scores split between clean and backdoored items.

    One panel per readout. The written 0-10 score is shown beside P(yes)
    because the contrast is the point: few distinct written scores mean heavy
    ties, and ties are what cap its usefulness at low false-positive rates.

    Args:
        primary: P(yes) scores.
        secondary: Written 0-10 scores, if loaded.
        summaries: ``summarize`` rows keyed by readout.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    panels = 2 if secondary is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.4), squeeze=False)
    color = ARM_COLORS[primary.arm]

    ax = axes[0][0]
    _mirrored_histogram(ax, primary.scores, primary.labels,
                        np.linspace(0, 1, 11), color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(yes) score")
    ax.set_ylabel("Share of items in class")
    ax.set_title(
        f"P(yes) · pAUC {summaries['logprob']['pauc']:.3f} · "
        f"{summaries['logprob']['distinct_values']} distinct values",
        loc="left", fontsize=10, color=INK, pad=8,
    )

    if secondary is not None:
        ax = axes[0][1]
        _mirrored_histogram(ax, secondary.scores, secondary.labels,
                            np.arange(-0.5, 11.5, 1.0), color)
        ax.set_xlim(-0.6, 10.6)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Written 0–10 score")
        ax.set_title(
            f"0–10 score · pAUC {summaries['text']['pauc']:.3f} · "
            f"{summaries['text']['distinct_values']} distinct values",
            loc="left", fontsize=10, color=INK, pad=8,
        )

    _figure_heading(
        fig, f"{ARM_TITLES[primary.arm]} — how scores separate the classes", context
    )
    return _save(fig, path)


def plot_all_arms_roc(
    results: dict[str, ArmScores],
    summaries: dict[str, dict[str, Any]],
    path: Path,
    context: str,
) -> Path:
    """Draw every arm on one low-FPR ROC chart, members in grey behind them.

    Args:
        results: Primary-readout scores keyed by arm.
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    max_fpr = METRICS.pauc_max_fpr
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    _low_fpr_axes(ax, max_fpr)

    first_member = True
    for result in results.values():
        for member in result.members.values():
            _roc_line(ax, member, color=MEMBER, linewidth=1.1, zorder=2,
                      label="Individual monitors" if first_member else None)
            first_member = False

    for arm, result in results.items():
        _roc_line(ax, result, color=ARM_COLORS[arm], linewidth=2.4, zorder=3,
                  label=f"{ARM_TITLES[arm]}   pAUC {summaries[arm]['pauc']:.3f}")

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] == "Random",
                                  labels[i] == "Individual monitors"))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", borderpad=0.8, handlelength=2.6)
    _heading(ax, "All arms — ROC at low false-positive rates", context)
    return _save(fig, path)


def plot_roc_facets(
    results: dict[str, ArmScores],
    summaries: dict[str, dict[str, Any]],
    path: Path,
    context: str,
) -> Path:
    """Draw the ROC as two small multiples, one question per panel.

    One chart cannot hold nine arms: eight of them sit inside a 0.12-wide pAUC
    band, M1/M5 are 0.003 apart and M2/M3/M8 span 0.011, so those curves
    overlap whatever colours they are given. Each panel therefore asks one
    question and carries three or four curves, and M4 -- the benchmark both
    questions are measured against -- is drawn in both as a neutral dashed
    reference rather than a series.

    Every curve is directly labelled at its right edge, which is also the
    secondary encoding the palette check requires.

    Args:
        results: Primary-readout scores keyed by arm.
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    max_fpr = METRICS.pauc_max_fpr
    # A panel with none of its arms scored is dropped rather than drawn empty,
    # so a partial run (or a --limit smoke run) still renders.
    panels = [(title, present) for title, arms in ROC_PANELS
              if (present := [a for a in arms if a in results])]
    if not panels:
        raise ValueError("no panel has any of its arms scored")
    fig, axes = plt.subplots(1, len(panels), figsize=(11.4, 4.9), squeeze=False)

    for ax, (question, arms) in zip(axes[0], panels, strict=True):
        _low_fpr_axes(ax, max_fpr)

        labels: list[tuple[float, str, str]] = []
        if BENCHMARK_ARM in results:
            _roc_line(ax, results[BENCHMARK_ARM], color=BENCHMARK_INK,
                      linewidth=1.8, linestyle=(0, (5, 2)), zorder=3)
            labels.append((
                _curve_end(results[BENCHMARK_ARM], max_fpr), BENCHMARK_INK,
                f"{ARM_TITLES[BENCHMARK_ARM].split(' · ')[0]} (benchmark)",
            ))

        for arm in arms:
            _roc_line(ax, results[arm], color=ARM_COLORS[arm], linewidth=2.2,
                      zorder=4)
            labels.append((
                _curve_end(results[arm], max_fpr), ARM_COLORS[arm],
                f"{ARM_TITLES[arm].split(' · ')[0]}  "
                f"{summaries[arm]['pauc']:.3f}",
            ))
        _place_labels(ax, labels, max_fpr)
        ax.set_title(question, fontsize=11, color=INK, pad=10, loc="left")

    _figure_heading(
        fig,
        "Detection at low false-positive rates",
        f"{context} · pAUC beside each arm · dashed line is M4, the benchmark",
    )
    return _save(fig, path)


def _curve_end(result: ArmScores, max_fpr: float) -> float:
    """True-positive rate where a curve leaves the plotted region.

    Args:
        result: The arm's scores and labels.
        max_fpr: Right edge of the plotted region.

    Returns:
        The final in-region TPR, or 0.0 if the curve never enters it.
    """
    fpr, tpr = roc_curve(result.scores, result.labels)
    inside = fpr <= max_fpr
    return float(tpr[inside][-1]) if inside.any() else 0.0


def _place_labels(
    ax: plt.Axes, labels: Sequence[tuple[float, str, str]], max_fpr: float
) -> None:
    """Write each curve's name at its right edge, nudged apart where they clash.

    Direct labelling is what lets a reader tell two curves apart without
    matching colours to a legend box, and it is the secondary encoding the
    palette validator asks for when a pair sits in the 6-8 CVD band. Arms whose
    curves finish within a line-height of each other -- M5 and M8 do -- would
    otherwise overprint, so the text is pushed apart while a leader dot stays
    on the true endpoint.

    Args:
        ax: Target axes.
        labels: ``(y, colour, text)`` per curve, any order.
        max_fpr: Right edge of the plotted region, where the dots sit.
    """
    if not labels:
        return
    gap = 0.052
    ordered = sorted(labels, key=lambda item: item[0])
    placed: list[float] = []
    for y, _colour, _text in ordered:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    # Keep the stack inside the axes if pushing up overflowed the top.
    overflow = placed[-1] - 1.0
    if overflow > 0:
        placed = [y - overflow for y in placed]

    for (true_y, colour, text), text_y in zip(ordered, placed, strict=True):
        ax.plot([max_fpr], [true_y], marker="o", markersize=4.5, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6,
                clip_on=False)
        ax.annotate(
            text, xy=(max_fpr, text_y), xytext=(7, 0),
            textcoords="offset points", va="center", ha="left", fontsize=9,
            color=INK_SECONDARY, zorder=6, annotation_clip=False,
        )


def plot_ingredient_costs(
    comparisons: Sequence[dict[str, Any]], path: Path, context: str
) -> Path:
    """Draw what each ingredient costs, as differences with 95% intervals.

    A difference of 0.02 pAUC cannot be eyeballed off two overlapping ROC
    curves, so the claims that rest on differences get the form that shows
    them: one interval per comparison, a neutral zero rule, and opposite poles
    for a gain and a loss.

    Args:
        comparisons: Rows from ``analysis.compare_arms`` with ``arm``,
            ``baseline``, ``delta_pauc`` and its bootstrap interval.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    wanted = {
        ("m4", "m3"): "Teacher diversity\nM4 \u2212 M3: diverse vs identical ensemble",
        ("m8", "m5"): "Dropping the vote\nM8 \u2212 M5: graded target vs MACA's vote",
        ("m8", "m4"): "Adding the debate\nM8 \u2212 M4: debated vs undebated teacher",
    }
    rows = [(wanted[(c["arm"], c["baseline"])], c) for c in comparisons
            if (c["arm"], c["baseline"]) in wanted]
    if not rows:
        raise ValueError("none of the ingredient comparisons were computed")
    rows.reverse()

    fig, ax = plt.subplots(figsize=(8.6, 0.78 * len(rows) + 1.7))
    ax.axvline(0.0, color=MUTED, linewidth=1.2, zorder=2)

    for index, (_label, row) in enumerate(rows):
        delta = row["delta_pauc"]
        low, high = row["delta_pauc_low"], row["delta_pauc_high"]
        color = COST_GAIN if delta >= 0 else COST_LOSS
        ax.plot([low, high], [index, index], color=color, linewidth=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([delta], [index], marker="o", markersize=9, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        crosses = low <= 0 <= high
        ax.annotate(
            f"{delta:+.3f}  [{low:+.3f}, {high:+.3f}]"
            + ("  spans zero" if crosses else ""),
            xy=(high, index), xytext=(10, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9.5,
            color=INK if not crosses else MUTED, zorder=5,
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=9.5,
                       color=INK_SECONDARY)
    ax.set_xlabel("Change in pAUC@10% FPR", color=INK_SECONDARY)
    ax.margins(x=0.34, y=0.42)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _heading(ax, "What each ingredient is worth",
             f"{context} · 95% paired-bootstrap intervals")
    return _save(fig, path)


def plot_roc_small_multiples(
    results: dict[str, ArmScores],
    summaries: dict[str, dict[str, Any]],
    path: Path,
    context: str,
    arms: Sequence[str] = HEADLINE_ARMS,
) -> Path:
    """Draw one ROC panel per arm, each against its peers in grey.

    Six curves cannot share one axes: crossing lines are an all-pairs case, and
    the categorical palette validates all-pairs for **three** series at most --
    past that no ordering of any hue set clears the separation floors, and the
    documented fix is to facet. Small multiples do exactly that while still
    showing every arm in a single image: each panel colours one arm and greys
    the rest, so a colour never has to be told apart from another colour.

    The grey backdrop is what makes the panels comparable -- each arm is read
    against the same five peers, in the same place, at the same scale.

    Args:
        results: Primary-readout scores keyed by arm.
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.
        arms: Which arms get a panel, in reading order.

    Returns:
        ``path``.
    """
    apply_style()
    max_fpr = METRICS.pauc_max_fpr
    present = [arm for arm in arms if arm in results]
    if not present:
        raise ValueError("none of the requested arms were scored")

    columns = 3
    rows = -(-len(present) // columns)  # ceil without importing math
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.7 * columns, 3.25 * rows),
        squeeze=False, sharex=True, sharey=True,
    )
    flat = [ax for row in axes for ax in row]

    for ax, arm in zip(flat, present, strict=False):
        _low_fpr_axes(ax, max_fpr)
        for other in present:
            if other != arm:
                _roc_line(ax, results[other], color=GRIDLINE, linewidth=1.3,
                          zorder=2)
        _roc_line(ax, results[arm], color=ARM_COLORS[arm], linewidth=2.4,
                  zorder=4)
        ax.set_title(
            f"{ARM_TITLES[arm]}   pAUC {summaries[arm]['pauc']:.3f}",
            fontsize=9.5, color=INK, pad=7, loc="left",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in flat[len(present):]:
        ax.set_visible(False)

    fig.supxlabel("False positive rate", color=INK_SECONDARY, fontsize=10)
    fig.supylabel("True positive rate", color=INK_SECONDARY, fontsize=10)
    _figure_heading(
        fig, "Each arm against the others",
        f"{context} · the highlighted arm in colour, its peers in grey",
    )
    return _save(fig, path)


def plot_metric_intervals(
    summaries: dict[str, dict[str, Any]], path: Path, context: str
) -> Path:
    """Draw pAUC and AUROC per arm with 95% intervals.

    Args:
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    arms = list(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.62 * len(arms) + 2.4),
                             sharey=True)
    rows = range(len(arms))[::-1]

    for ax, metric, name in ((axes[0], "pauc", "pAUC at FPR ≤ 10%"),
                             (axes[1], "auroc", "AUROC")):
        lows = [summaries[a][f"{metric}_low"] for a in arms]
        ax.axvline(0.5, color=AXIS, linewidth=1.0, zorder=1)
        for y, arm in zip(rows, arms, strict=True):
            row = summaries[arm]
            ax.hlines(y, row[f"{metric}_low"], row[f"{metric}_high"],
                      color=ARM_COLORS[arm], linewidth=2.2, zorder=2)
            ax.plot(row[metric], y, "o", markersize=8, color=ARM_COLORS[arm],
                    markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
            ax.text(1.02, y, f"{row[metric]:.3f}  [{row[f'{metric}_low']:.2f}, "
                    f"{row[f'{metric}_high']:.2f}]",
                    transform=ax.get_yaxis_transform(), va="center",
                    fontsize=9, color=INK_SECONDARY)
        ax.set_xlim(min(0.45, min(lows) - 0.03), 1.0)
        ax.set_ylim(-0.7, len(arms) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        ax.text(0.5, -0.62, "chance", fontsize=8, color=MUTED, ha="center",
                va="bottom")

    axes[0].set_yticks(list(rows), [ARM_TITLES[a] for a in arms])
    axes[0].tick_params(axis="y", length=0)
    _figure_heading(fig, "Detection per arm, with 95% intervals", context)
    fig.subplots_adjust(wspace=0.55)
    return _save(fig, path)


def plot_frontier(
    summaries: dict[str, dict[str, Any]],
    path: Path,
    context: str,
    arms: Sequence[str] | None = None,
) -> Path:
    """Draw pAUC against inference cost -- the experiment's main figure.

    Each arm sits at its cost (1x or 3x). The M0 line marks the single-monitor
    floor and the M2 line the diversity ceiling; the research question is
    whether a 1x trained arm lands near the ceiling.

    Args:
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.
        arms: Restrict to these arms, in ``HEADLINE_ARMS`` style. ``None``
            draws every scored arm, which is the full-detail view for the
            report -- nine points put six in the 1x column and the chart stops
            making its one comparison legible.

    Returns:
        ``path``.
    """
    apply_style()
    if arms is not None:
        shown = {a: summaries[a] for a in arms if a in summaries}
        # Say on the figure itself that arms were left out. It travels into
        # write-ups on its own, where a silent subset would read as selection.
        if len(shown) < len(summaries):
            context = (f"{context} · {len(shown)} of {len(summaries)} arms shown"
                       " · full chart in the report")
        summaries = shown
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positions = {1: 0.0, 3: 1.0}
    by_cost: dict[int, list[str]] = {}
    for arm, row in summaries.items():
        by_cost.setdefault(row["inference_cost"], []).append(arm)

    for reference, name in (("m0", "floor"), ("m2", "ceiling")):
        if reference in summaries:
            level = summaries[reference]["pauc"]
            ax.axhline(level, color=AXIS, linewidth=1.0, zorder=1)
            ax.text(-0.47, level, f"{reference.upper()} {name}", fontsize=8,
                    color=MUTED, va="center", ha="left",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})

    for cost, arms in by_cost.items():
        offsets = np.linspace(-0.2, 0.2, len(arms)) if len(arms) > 1 else [0.0]
        for index, (offset, arm) in enumerate(zip(offsets, arms, strict=True)):
            row = summaries[arm]
            x = positions.get(cost, float(cost)) + offset
            ax.vlines(x, row["pauc_low"], row["pauc_high"], color=ARM_COLORS[arm],
                      linewidth=2.2, zorder=2)
            ax.plot(x, row["pauc"], "o", markersize=9, color=ARM_COLORS[arm],
                    markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3,
                    label=ARM_TITLES[arm])
            # With several arms at one cost, the leftmost labels to its left so
            # neighbouring labels never run into each other's markers.
            leftward = len(arms) > 1 and index == 0
            ax.text(x - 0.05 if leftward else x + 0.05, row["pauc"],
                    f"{arm.upper()}  {row['pauc']:.3f}", fontsize=9, color=INK,
                    va="center", ha="right" if leftward else "left",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})

    lows = [r["pauc_low"] for r in summaries.values()]
    highs = [r["pauc_high"] for r in summaries.values()]
    ax.set_ylim(max(0.0, min(lows) - 0.05), min(1.0, max(highs) + 0.05))
    ax.set_xlim(-0.5, 1.6)
    ax.set_xticks([0, 1], ["1× inference cost", "3× inference cost"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("pAUC at FPR ≤ 10%")
    # No legend: every point is labelled where it sits, so a colour key would
    # repeat the labels and cost a quarter of the plotting area.
    _heading(ax, "Detection vs. inference cost", context)
    return _save(fig, path)


def _tint(color: str, amount: float) -> str:
    """Mix a colour toward the chart surface.

    Args:
        color: Hex colour.
        amount: 0 keeps the colour, 1 gives the surface.

    Returns:
        The mixed hex colour.
    """
    base = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    surface = np.array([int(SURFACE[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    mixed = base + (surface - base) * amount
    return "#" + "".join(f"{round(channel):02x}" for channel in mixed)


def _relative_luminance(color: str) -> float:
    """WCAG relative luminance of a hex colour."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _label_ink(fill: str) -> str:
    """Pick white or ink for text on a fill, whichever contrasts more.

    Args:
        fill: The bar's hex colour.

    Returns:
        ``"#ffffff"`` or the ink colour.
    """
    fill_luminance = _relative_luminance(fill)
    against_white = 1.05 / (fill_luminance + 0.05)
    against_ink = (fill_luminance + 0.05) / (_relative_luminance(INK) + 0.05)
    return "#ffffff" if against_white >= against_ink else INK


def plot_pauc_bars(
    monitors: Sequence[dict[str, Any]], path: Path, context: str
) -> Path:
    """Draw pAUC for every monitor as ranked bars with 95% intervals.

    Follows the anchor paper's individual-monitor figure. Each arm keeps its
    colour from every other chart; the individual monitors inside an ensemble
    are the same colour, lighter, so an ensemble reads directly against its
    own members.

    Args:
        monitors: Rows from ``analysis.monitor_rows``, highest pAUC first.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
    count = len(monitors)
    fig, ax = plt.subplots(figsize=(8.6, 0.46 * count + 1.9))
    rows = list(range(count))[::-1]

    fills = []
    for y, row in zip(rows, monitors, strict=True):
        base = ARM_COLORS[row["arm"]]
        fill = base if row["kind"] == "arm" else _tint(base, 0.55)
        fills.append(fill)
        ax.barh(y, row["pauc"], height=0.78, color=fill, edgecolor=SURFACE,
                linewidth=1.5, zorder=2)
        ax.errorbar(row["pauc"], y,
                    xerr=[[row["pauc"] - row["pauc_low"]],
                          [row["pauc_high"] - row["pauc"]]],
                    fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, count - 0.1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        "pAUC at FPR ≤ 10%  (McClish-standardised: 0.5 = random, 1.0 = perfect)"
    )

    # Name each bar inside it when the name fits; otherwise place it after the
    # value, so no label is ever clipped by its own bar.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for y, row, fill in zip(rows, monitors, fills, strict=True):
        value = f"{row['pauc']:.3f}"
        name = ax.text(0.012, y, row["monitor"], va="center", ha="left",
                       fontsize=9.5, fontweight="bold", color=_label_ink(fill),
                       zorder=4)
        name_right = ax.transData.inverted().transform(
            (name.get_window_extent(renderer).x1, 0)
        )[0]
        if name_right + 0.01 > row["pauc_low"]:
            name.remove()
            value = f"{value}   {row['monitor']}"
        ax.text(row["pauc_high"] + 0.012, y, value, va="center", ha="left",
                fontsize=9.5, color=INK, zorder=4)

    # Derived, never hardcoded: a literal list here silently dropped M6-M8
    # from the key while their bars were still drawn.
    arms_present = [
        arm for arm in ARM_COLORS if any(r["arm"] == arm for r in monitors)
    ]
    handles = [
        Patch(facecolor=ARM_COLORS[arm], label=ARM_TITLES[arm]) for arm in arms_present
    ]
    if any(row["kind"] == "member" for row in monitors):
        handles.append(Patch(facecolor=_tint(MUTED, 0.55),
                             label="Lighter: one monitor inside that ensemble"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.6)

    _heading(
        ax, "Every monitor, ranked by pAUC", f"95% bootstrap intervals · {context}"
    )
    return _save(fig, path)


def render_all(
    loaded: dict[str, dict[str, ArmScores]],
    summaries: Sequence[dict[str, Any]],
    out_dir: Path,
    context: str,
    monitors: Sequence[dict[str, Any]] = (),
    comparisons: Sequence[dict[str, Any]] = (),
) -> dict[str, list[tuple[str, str]]]:
    """Draw every figure the report uses.

    Args:
        loaded: ``{readout: {arm: ArmScores}}``.
        summaries: All ``summarize`` rows.
        out_dir: Where to write the ``.png`` files.
        context: Setup and slice description for subtitles.
        monitors: Rows from ``analysis.monitor_rows``, for the ranked bars.
        comparisons: Primary-readout rows from ``analysis.compare_arms``, for
            the ingredient-cost chart.

    Returns:
        ``{"<arm>" | "comparison": [(path relative to the report, caption)]}``.
    """
    from src.analysis import PRIMARY_READOUT

    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        by_arm.setdefault(row["arm"], {})[row["readout"]] = row

    primary = loaded[PRIMARY_READOUT]
    entries: dict[str, list[tuple[str, str]]] = {}

    for arm, result in primary.items():
        roc = plot_arm_roc(result, by_arm[arm][PRIMARY_READOUT],
                           out_dir / f"{arm}_roc.png", context)
        scores = plot_arm_scores(result, loaded.get("text", {}).get(arm),
                                 by_arm[arm], out_dir / f"{arm}_scores.png", context)
        members = (f" Grey lines are the {len(result.members)} individual monitors "
                   "averaged into this ensemble.") if result.members else ""
        entries[arm] = [
            (f"figures/{roc.name}",
             f"{ARM_TITLES[arm]}: ROC curve for false-positive rates up to 10%."
             f"{members}"),
            (f"figures/{scores.name}",
             f"{ARM_TITLES[arm]}: share of backdoored items (above the line) and "
             "clean items (below) at each score."),
        ]

    primary_rows = {arm: by_arm[arm][PRIMARY_READOUT] for arm in primary}
    comparison: list[tuple[str, str]] = []

    facets = None
    if len(primary) >= 2:
        try:
            facets = plot_roc_facets(
                primary, primary_rows, out_dir / "roc_facets.png", context
            )
        except ValueError:
            facets = None
    if facets is not None:
        comparison.append((
            f"figures/{facets.name}",
            "ROC at false-positive rates up to 10%, split by question. M4, the "
            "benchmark both questions are measured against, is the dashed line "
            "in each panel.",
        ))

    if len(primary) >= 2:
        grid = plot_roc_small_multiples(
            primary, primary_rows, out_dir / "all_arms_roc.png", context
        )
        comparison.append((
            f"figures/{grid.name}",
            "One panel per arm: the highlighted arm in colour against its peers "
            "in grey. Six curves cannot share one axes, so each gets its own.",
        ))
        crowded = plot_all_arms_roc(
            primary, primary_rows, out_dir / "all_arms_roc_single.png", context
        )
        comparison.append((
            f"figures/{crowded.name}",
            "The same arms on one axes, kept for reference. Curves overlap "
            "heavily -- this is why the panels above exist.",
        ))

    cost_rows = [c for c in comparisons if c.get("readout") == PRIMARY_READOUT]
    if cost_rows:
        try:
            costs = plot_ingredient_costs(
                cost_rows, out_dir / "ingredient_costs.png", context
            )
        except ValueError:
            costs = None
        if costs is not None:
            comparison.append((
                f"figures/{costs.name}",
                "What each ingredient is worth, as a difference with its 95% "
                "interval. Differences this small cannot be read off "
                "overlapping ROC curves.",
            ))

    if len(monitors) >= 2:
        bars = plot_pauc_bars(monitors, out_dir / "pauc_bars.png", context)
        comparison.append((
            f"figures/{bars.name}",
            "pAUC for every arm and every monitor inside an ensemble, with 95% "
            "intervals. Lighter bars are an ensemble's individual members.",
        ))

    if len(primary) >= 2:
        metrics = plot_metric_intervals(primary_rows, out_dir / "metrics.png", context)
        frontier = plot_frontier(primary_rows, out_dir / "frontier.png", context)
        headline = plot_frontier(
            primary_rows, out_dir / "frontier_headline.png", context,
            arms=HEADLINE_ARMS,
        )
        comparison += [
            (f"figures/{metrics.name}",
             "pAUC and AUROC per arm with 95% bootstrap intervals."),
            (f"figures/{headline.name}",
             "pAUC against inference cost, headline arms only: does a 1× "
             "trained arm reach the 3× ceiling?"),
            (f"figures/{frontier.name}",
             "The same chart with every arm, as the full-detail view."),
        ]

    if comparison:
        entries["comparison"] = comparison
    return entries
