"""The analyse stage, on small synthetic score files.

These check the parts where a quiet mistake would put wrong numbers in front of
a reader: ensembles averaging without rounding, members being separated
correctly, arms that have not been scored staying "pending" rather than being
guessed at, and comparison figures appearing only when there is something to
compare.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis import (
    aggregate,
    evaluate_hypotheses,
    load_arm,
    member_name,
    run_analysis,
    slice_name,
)

ITEMS = 40


def write_arm(directory, arm, members, separation=1.5, seed=0):
    """Write a synthetic score file for one arm.

    Args:
        directory: Where generation files live.
        arm: Arm id.
        members: ``[(prompt_name, sample)]``, one per individual monitor.
        separation: How far apart the two classes' scores sit.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for prompt, sample in members:
        for i in range(ITEMS):
            for label in (1, 0):
                centre = separation if label else -separation
                score = float(1 / (1 + np.exp(-rng.normal(centre, 1.0))))
                rows.append({
                    "item_id": f"p{i}__{'backdoored' if label else 'clean'}::{prompt}",
                    "problem_id": f"p{i}", "label": label, "arm": arm,
                    "prompt_name": prompt, "sample": sample,
                    "logprob_score": score, "text_score": round(score * 10),
                    "finish_reason": "stop", "response": "",
                    "backend": "synthetic", "model": "test",
                })
    (directory / f"{arm}__val.json").write_text(json.dumps(rows))


def test_slice_name_matches_score_file_naming():
    """Analyse must find exactly the files `score` wrote."""
    assert slice_name("val", None) == "val"
    assert slice_name("val", 100) == "val__limit100"


def test_member_names():
    """Personas are named for what they look at; repeat draws by number."""
    assert member_name({"prompt_name": "persona_control_flow"}) == "control flow"
    assert member_name({"prompt_name": "baseline", "sample": 2}) == "sample 3"


def test_ensemble_mean_is_not_rounded():
    """Averaging 5, 6 and 8 must give 6.333…, never 6."""
    records = [
        {"item_id": f"a::{p}", "label": 1, "text_score": s}
        for p, s in (("x", 5), ("y", 6), ("z", 8))
    ]
    _, scores, _, _, _ = aggregate(records, "text_score")
    assert scores[0] == pytest.approx(19 / 3)


def test_unreadable_samples_are_counted_not_hidden():
    """A missing sample shrinks the mean's support and is reported."""
    records = [
        {"item_id": "a::x", "label": 1, "logprob_score": 0.8},
        {"item_id": "a::y", "label": 1, "logprob_score": None},
        {"item_id": "b::x", "label": 0, "logprob_score": None},
    ]
    ids, scores, _, partial, unreadable = aggregate(records, "logprob_score")
    assert ids == ["a"] and scores[0] == pytest.approx(0.8)
    assert partial == 1
    assert unreadable == 1


def test_load_arm_separates_ensemble_members(tmp_path):
    """M2's file yields one averaged arm plus its three personas."""
    personas = [("persona_control_flow", 0), ("persona_reference_solution", 0),
                ("persona_attacker", 0)]
    write_arm(tmp_path, "m2", personas)
    result = load_arm("m2", "val", None, "logprob", tmp_path)

    assert len(result.item_ids) == 2 * ITEMS
    assert set(result.members) == {"control flow", "reference solution", "attacker"}
    assert load_arm("m1", "val", None, "logprob", tmp_path) is None


def test_hypotheses_stay_pending_without_their_arms():
    """Nothing is concluded about arms that have not been scored."""
    rows = evaluate_hypotheses(summaries=[], comparisons=[])
    assert {row["status"] for row in rows} == {"pending"}


def test_run_analysis_writes_report_tables_and_figures(tmp_path):
    """One arm gives per-arm charts only; two arms add comparison charts."""
    generations = tmp_path / "gen"
    generations.mkdir()
    results = tmp_path / "results"

    write_arm(generations, "m0", [("baseline", 0)], seed=1)
    run_analysis("val", resamples=50, generations_dir=generations, results_dir=results)
    figures = results / "val" / "figures"
    assert {p.name for p in figures.iterdir()} == {"m0_roc.png", "m0_scores.png"}

    write_arm(generations, "m1", [("baseline", s) for s in range(3)], seed=2)
    report = run_analysis(
        "val", resamples=50, generations_dir=generations, results_dir=results
    )
    names = {p.name for p in figures.iterdir()}
    assert {"all_arms_roc.png", "pauc_bars.png", "metrics.png", "frontier.png"} <= names

    text = report.read_text()
    assert "M1 · 3 identical samples" in text
    assert "figures/all_arms_roc.png" in text
    assert (results / "val" / "summary.csv").exists()

    h1 = next(
        line for line in (results / "val" / "hypotheses.csv").read_text().splitlines()
        if line.startswith("H1")
    )
    assert "pending" not in h1


def test_run_analysis_explains_a_missing_slice(tmp_path):
    """No score files means a clear error naming the command to run."""
    with pytest.raises(FileNotFoundError, match="main.py score"):
        run_analysis("val", generations_dir=tmp_path, results_dir=tmp_path / "r")


def test_detection_at_fpr_on_same_items():
    """A threshold for 10% FPR flags about 10% of the clean items it was set on."""
    from src.analysis import detection_at_fpr

    rng = np.random.default_rng(3)
    labels = np.concatenate([np.ones(500, int), np.zeros(500, int)])
    scores = np.concatenate([rng.normal(1, 1, 500), rng.normal(-1, 1, 500)])
    recall, realised = detection_at_fpr(scores, labels, 0.10)
    assert realised == pytest.approx(0.10, abs=0.01)
    assert recall > realised


def test_detection_at_fpr_uses_val_thresholds_on_test():
    """With calibration, the threshold comes from val, not the test items."""
    from src.analysis import ArmScores, detection_at_fpr

    labels = np.array([1, 1, 0, 0])
    test_scores = np.array([0.9, 0.6, 0.55, 0.1])
    val = ArmScores("m0", "logprob", [], np.array([0.9, 0.8, 0.5, 0.4]),
                    np.array([1, 1, 0, 0]))
    recall, realised = detection_at_fpr(test_scores, labels, 0.01, calibration=val)
    # Val's clean 99th percentile is ~0.499, so test's clean 0.55 is flagged.
    assert realised == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)


def test_ensemble_table_and_diversity_multiplier():
    """M2's gain over M0 divided by M1's gain, as in the anchor paper."""
    from src.analysis import ensemble_rows

    monitors = [
        {"monitor": "M2", "arm": "m2", "kind": "arm", "pauc": 0.82},
        {"monitor": "M1", "arm": "m1", "kind": "arm", "pauc": 0.76},
        {"monitor": "M0", "arm": "m0", "kind": "arm", "pauc": 0.72},
        {"monitor": "M2 · a", "arm": "m2", "kind": "member", "pauc": 0.74},
        {"monitor": "M2 · b", "arm": "m2", "kind": "member", "pauc": 0.70},
        {"monitor": "M2 · c", "arm": "m2", "kind": "member", "pauc": 0.66},
    ]
    rows, multiplier = ensemble_rows(monitors)
    assert multiplier == pytest.approx(0.10 / 0.04)
    roles = [row["role"] for row in rows]
    assert roles[0] == "diverse ensemble"
    median = next(row for row in rows if row["role"] == "median persona")
    assert median["pauc"] == pytest.approx(0.70)


def test_fpr_table_ranks_by_pauc_and_carries_cost(tmp_path):
    """Rows are ranked by pAUC and list inference cost."""
    import csv

    generations = tmp_path / "gen"
    generations.mkdir()
    write_arm(generations, "m0", [("baseline", 0)], separation=0.8, seed=4)
    write_arm(generations, "m1", [("baseline", s) for s in range(3)],
              separation=1.6, seed=5)
    run_analysis("val", resamples=50, generations_dir=generations,
                 results_dir=tmp_path / "results")

    with open(tmp_path / "results" / "val" / "fpr_detection.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["rank"] for row in rows] == ["1", "2"]
    assert float(rows[0]["pauc"]) >= float(rows[1]["pauc"])
    assert {row["arm"]: row["inference_cost"] for row in rows} == {"m0": "1", "m1": "3"}
    assert "recall_at_1pct_fpr" in rows[0]
    assert (tmp_path / "results" / "val" / "figures" / "pauc_bars.png").exists()
