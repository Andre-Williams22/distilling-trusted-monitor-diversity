"""Split-construction invariants, checked on synthetic problems.

No network and no dataset download: these build ``Problem`` objects directly,
so they run in milliseconds on a laptop and fail for exactly one reason.

The invariants here are the ones whose violation is silent. A leaked problem or
a nondeterministic item in test does not raise anything -- it just quietly
inflates every number in the write-up.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.config import SplitConfig
from src.data import Problem, assign_splits, build_items


def make_problems(n_deterministic: int, n_nondeterministic: int) -> list[Problem]:
    """Build synthetic problems with distinct ids and code.

    Args:
        n_deterministic: How many deterministic problems.
        n_nondeterministic: How many nondeterministic ones.

    Returns:
        The combined list.
    """
    problems = []
    for i in range(n_deterministic + n_nondeterministic):
        problems.append(
            Problem(
                problem_id=f"p{i:05d}",
                question=f"question {i}",
                clean_code=f"def solve(): return {i}",
                backdoor_code=f"def solve(): return {i} if x != 7 else 0",
                backdoor_input="7",
                difficulty="interview",
                is_nondeterministic=i >= n_deterministic,
                backdoor_source="apps",
            )
        )
    return problems


@pytest.fixture
def config() -> SplitConfig:
    """Small split sizes, same shape as the real config."""
    return SplitConfig(train_problems=60, val_problems=15, test_problems=50)


@pytest.fixture
def problems() -> list[Problem]:
    """110 deterministic plus 15 nondeterministic."""
    return make_problems(110, 15)


def test_build_items_pairs_a_problem(problems):
    """One problem yields exactly one backdoored and one clean item."""
    backdoored, clean = build_items(problems[0])
    assert backdoored.label == 1
    assert clean.label == 0
    assert backdoored.problem_id == clean.problem_id
    assert backdoored.item_id != clean.item_id


def test_paired_items_share_the_question_and_differ_in_code(problems):
    """The only difference a monitor may detect is the code itself.

    If the questions differed, a monitor could learn "hard problem ->
    suspicious" instead of detecting backdoors.
    """
    backdoored, clean = build_items(problems[0])
    assert backdoored.question == clean.question
    assert backdoored.code != clean.code


def test_no_problem_appears_in_two_splits(problems, config):
    """The leakage invariant: splits are cut by problem, never by item."""
    assignment = assign_splits(problems, config)
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for problem_id, split in assignment.items():
        by_split[split].add(problem_id)

    assert not by_split["train"] & by_split["test"]
    assert not by_split["train"] & by_split["val"]
    assert not by_split["val"] & by_split["test"]
    assert sum(len(v) for v in by_split.values()) == len(problems)


def test_val_and_test_sizes_are_exact(problems, config):
    """Val and test take exactly the configured counts; train takes the rest."""
    counts = Counter(assign_splits(problems, config).values())
    assert counts["val"] == config.val_problems
    assert counts["test"] == config.test_problems
    assert counts["train"] == len(problems) - config.val_problems - config.test_problems


def test_nondeterministic_problems_all_land_in_train(problems, config):
    """Noisy labels are free in train and toxic in val and test.

    No training arm reads a ground-truth label -- M4 regresses onto teacher
    scores and M3's pairs come from monitor consensus -- so label noise cannot
    reach the training signal. It would corrupt val thresholds and every test
    metric.
    """
    assignment = assign_splits(problems, config)
    for problem in problems:
        if problem.is_nondeterministic:
            assert assignment[problem.problem_id] == "train"


def test_assignment_is_independent_of_input_order(problems, config):
    """Ordering must come from a stable hash, not from iteration order.

    Python's built-in ``hash`` is salted per process, so using it would put the
    same problem in train on the laptop and test on the GPU box.
    """
    forward = assign_splits(problems, config)
    backward = assign_splits(list(reversed(problems)), config)
    assert forward == backward


def test_assignment_is_stable_across_runs(problems, config):
    """The split is a preregistered artifact; the same seed must reproduce it."""
    assert assign_splits(problems, config) == assign_splits(problems, config)


def test_different_seeds_give_different_splits(problems, config):
    """Sanity check that the seed is actually wired into the ordering."""
    other = SplitConfig(
        train_problems=config.train_problems,
        val_problems=config.val_problems,
        test_problems=config.test_problems,
        seed=config.seed + 1,
    )
    assert assign_splits(problems, config) != assign_splits(problems, other)


def test_insufficient_deterministic_problems_raises(config):
    """Failing loudly beats silently shrinking test and wrecking the pAUC region."""
    too_few = make_problems(config.val_problems + config.test_problems - 1, 50)
    with pytest.raises(ValueError, match="deterministic problems"):
        assign_splits(too_few, config)
