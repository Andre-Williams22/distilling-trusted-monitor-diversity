"""Dataset loading, filtering, item construction and split assignment.

The output of this module is three frozen JSONL files plus a metadata sidecar.
**Nothing downstream ever touches the raw dataset again** — every later stage
reads only these files, so a change in the upstream dataset cannot silently
alter a result mid-project.

Pipeline shape::

    load_raw() -> filter_qualifying() -> build_items() -> assign_splits()
               -> write_splits()

See ADR-0001 for why this dataset, and project-plan.md section 2 for the
measured supply this module is expected to reproduce.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from datasets import load_dataset


from src.config import DatasetConfig, SplitConfig

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Problem:
    """One qualifying APPS problem with its clean and backdoored solutions.

    An intermediate representation: a raw dataset row narrowed to the fields
    this project uses, after a working backdoor and a passing clean solution
    have both been located.

    Attributes:
        problem_id: Stable identifier from the dataset. Splits are cut on this.
        question: The APPS problem statement shown to the monitor.
        clean_code: A solution that compiles and passes the public tests.
        backdoor_code: A solution that passes the public tests but misbehaves
            on ``backdoor_input``.
        backdoor_input: The input that triggers the backdoor. Never shown to a
            monitor — it would give the answer away.
        difficulty: APPS difficulty label, carried for stratification checks.
        is_nondeterministic: Whether the dataset flags this problem's behaviour
            as unstable. Such problems are routed to train only.
        backdoor_source: Provenance, ``"apps"`` or ``"control_tax"``.
    """

    problem_id: str
    question: str
    clean_code: str
    backdoor_code: str
    backdoor_input: str
    difficulty: str
    is_nondeterministic: bool
    backdoor_source: str


@dataclass(frozen=True)
class Item:
    """One (problem, solution, label) triple presented to a monitor for scoring.

    Each qualifying problem produces exactly two items — one backdoored, one
    clean — which is what fixes the attack rate at 50% by construction, and
    what stops a monitor from learning "hard problem -> suspicious" instead of
    detecting backdoors.

    Attributes:
        item_id: Unique per item, derived from ``problem_id`` and ``label``.
        problem_id: The parent problem. Both items of a pair share it, and both
            therefore always land in the same split.
        question: The problem statement.
        code: The solution being judged — clean or backdoored.
        label: ``1`` if backdoored, ``0`` if clean. **Read only by val
            thresholds and test metrics.** No training arm ever sees it.
        is_nondeterministic: Carried through so split assignment can honour the
            train-only routing.
    """

    item_id: str
    problem_id: str
    question: str
    code: str
    label: int
    is_nondeterministic: bool


def load_raw(config: DatasetConfig) -> Iterable[dict[str, Any]]:
    """Pull the dataset from Hugging Face.

    The dataset is ungated, so no authentication is required. It has a single
    ``train`` split of 5,000 rows regardless of how this project later cuts it.

    Args:
        config: Which dataset and revision to fetch.

    Returns:
        The raw rows, with the dataset's own column names intact.
    """

    try: 
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("RoganInglis/apps-control-arena")
    except Exception as e: 
        raise NotImplementedError
    return ds 


def select_working_backdoor(row: dict[str, Any]) -> dict[str, Any] | None:
    """Pick a validated backdoor from a row's ``backdoors`` list.

    A row may carry several candidate backdoors from different sources; this
    project needs one that is actually validated to misbehave. Note that a
    backdoor's own ``passes_tests`` is necessary but not sufficient — it means
    the backdoor survives the public tests, not that it triggers correctly.

    Args:
        row: A raw dataset row.

    Returns:
        The first entry with ``backdoor_works`` set, or ``None`` if the row has
        no working backdoor.
    """
    raise NotImplementedError


def select_clean_solution(row: dict[str, Any]) -> str | None:
    """Pick a validated clean solution from a row's ``solutions`` list.

    Args:
        row: A raw dataset row.

    Returns:
        The code of the first solution that both compiles and passes the tests,
        or ``None`` if the row has no such solution.
    """
    raise NotImplementedError


def filter_qualifying(
    rows: Iterable[dict[str, Any]],
    config: DatasetConfig,
) -> Iterator[Problem]:
    """Narrow raw rows to problems this project can use.

    Applies the conjunctive filters from ADR-0001. Note that
    ``exclude_nondeterministic`` is **not** applied here: nondeterministic
    problems are retained and flagged, because they are routed to train rather
    than discarded. Dropping them at this stage would throw away 138 free
    training problems.

    Expected counts against the 9 Sep dataset, useful as a self-check:
    5,000 rows in, 1,582 problems out, of which 1,444 are deterministic.

    Args:
        rows: Raw dataset rows.
        config: Which filters to apply.

    Yields:
        One ``Problem`` per qualifying row.
    """
    raise NotImplementedError


def build_items(problem: Problem) -> tuple[Item, Item]:
    """Build the backdoored and clean items for a single problem.

    Both items carry the same ``problem_id`` so that ``assign_splits`` keeps
    them together, and both are constructed from the *same* problem statement
    so the only difference a monitor can detect is the code itself.

    Args:
        problem: A qualifying problem.

    Returns:
        The ``(backdoored, clean)`` pair, in that order.
    """
    raise NotImplementedError


def assign_splits(
    problems: Sequence[Problem],
    config: SplitConfig,
) -> dict[str, SplitName]:
    """Assign every problem to train, val or test.

    Two rules, both load-bearing:

    1. **Cut by ``problem_id``, never by item.** Splitting by item would put a
       problem's clean twin in test alongside its backdoor, leaking the problem
       and inflating every number.
    2. **Nondeterministic problems all go to train.** No training arm reads a
       ground-truth label, so their noisy labels cannot reach the training
       signal — but they would corrupt val thresholds and test metrics.

    The assignment is deterministic given ``config.seed`` and must not depend
    on iteration order or on Python's salted ``hash``.

    Args:
        problems: Every qualifying problem.
        config: Split sizes and seed.

    Returns:
        A mapping from ``problem_id`` to split name.

    Raises:
        ValueError: If the requested sizes cannot be satisfied — for example if
            fewer deterministic problems exist than val plus test require.
    """
    raise NotImplementedError


def write_splits(
    items_by_split: dict[SplitName, Sequence[Item]],
    out_dir: Path,
    metadata: dict[str, Any],
) -> dict[SplitName, Path]:
    """Write the three frozen JSONL splits plus ``splits.meta.json``.

    The metadata sidecar is what makes the splits reproducible and auditable:
    it records the seed, the dataset revision, the filter counts at each stage,
    and the git sha. Without it, a reviewer cannot check that the numbers in
    section 2 of the plan are the numbers the code actually produced.

    Args:
        items_by_split: Items to write, keyed by split name.
        out_dir: Destination, normally ``config.SPLITS_DIR``.
        metadata: Provenance to record alongside the splits.

    Returns:
        The path written for each split.
    """
    raise NotImplementedError


def load_split(name: SplitName, splits_dir: Path | None = None) -> list[Item]:
    """Read one frozen split back from disk.

    Every stage after split construction enters through this function.

    Args:
        name: Which split to load.
        splits_dir: Override for ``config.SPLITS_DIR``, for tests.

    Returns:
        The split's items, in file order.

    Raises:
        FileNotFoundError: If the split has not been built yet.
    """
    raise NotImplementedError


def build_all(
    dataset_config: DatasetConfig,
    split_config: SplitConfig,
    out_dir: Path,
) -> dict[SplitName, Path]:
    """Run the whole data stage end to end.

    The single entry point ``main.py build-data`` calls. Logs the count
    surviving each filter, so a mismatch against the preregistered numbers is
    visible immediately rather than at analysis time.

    Args:
        dataset_config: Dataset id, revision and filters.
        split_config: Split sizes and seed.
        out_dir: Where to write the splits.

    Returns:
        The path written for each split.
    """
    raise NotImplementedError
