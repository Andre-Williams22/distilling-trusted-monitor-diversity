"""Charts for the results report, drawn with matplotlib.

Two kinds of figure:

- **Per arm** -- a low-FPR ROC chart in the style of the anchor ensemble
  paper (FPR 0-0.10, the arm highlighted, an ensemble's individual members in
  light grey, a dashed random line), and a chart of how its scores split
  between clean and backdoored items.
- **All arms** -- the same ROC chart with every arm together, pAUC with 95%
  intervals per arm, and pAUC against inference cost.

Colours follow the arm, never its rank, so M2 is the same aqua in every chart
and a newly scored arm never repaints the others. The five arm colours pass a
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
    "m3": "M3 · DPO, MACA-style",
    "m4": "M4 · two-term SFT",
}

#: Fixed per arm, in run order of the palette's validated categorical slots.
ARM_COLORS = {
    "m0": "#2a78d6",
    "m1": "#eb6834",
    "m2": "#1baf7a",
    "m3": "#eda100",
    "m4": "#e87ba4",
}

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
    summaries: dict[str, dict[str, Any]], path: Path, context: str
) -> Path:
    """Draw pAUC against inference cost -- the experiment's main figure.

    Each arm sits at its cost (1x or 3x). The M0 line marks the single-monitor
    floor and the M2 line the diversity ceiling; the research question is
    whether a 1x trained arm lands near the ceiling.

    Args:
        summaries: Primary ``summarize`` rows keyed by arm.
        path: Destination file.
        context: Setup and slice description.

    Returns:
        ``path``.
    """
    apply_style()
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
    ax.legend(loc="lower right", borderpad=0.8)
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

    run_order = ("m0", "m1", "m2", "m4", "m3")
    arms_present = [arm for arm in run_order if any(r["arm"] == arm for r in monitors)]
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
) -> dict[str, list[tuple[str, str]]]:
    """Draw every figure the report uses.

    Args:
        loaded: ``{readout: {arm: ArmScores}}``.
        summaries: All ``summarize`` rows.
        out_dir: Where to write the ``.png`` files.
        context: Setup and slice description for subtitles.
        monitors: Rows from ``analysis.monitor_rows``, for the ranked bars.

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

    if len(primary) >= 2:
        roc = plot_all_arms_roc(
            primary, primary_rows, out_dir / "all_arms_roc.png", context
        )
        comparison.append((
            f"figures/{roc.name}",
            "Every arm's ROC curve at false-positive rates up to 10%, with "
            "individual ensemble members in grey.",
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
        comparison += [
            (f"figures/{metrics.name}",
             "pAUC and AUROC per arm with 95% bootstrap intervals."),
            (f"figures/{frontier.name}",
             "pAUC against inference cost. The question is whether a 1× trained "
             "arm reaches the M2 ceiling."),
        ]

    if comparison:
        entries["comparison"] = comparison
    return entries
