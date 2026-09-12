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

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset

from src.config import SPLITS_DIR, DatasetConfig, SplitConfig
from src.utils import (
    get_logger,
    git_sha,
    read_jsonl,
    set_seed,
    stable_hash,
    write_jsonl,
)

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


def _as_records(value: Any) -> list[dict[str, Any]]:
    """Normalise a nested dataset column into a list of dicts.

    ``datasets`` returns a list-of-structs column either as a list of dicts or,
    depending on version and access path, as a dict of parallel lists. Handling
    both here keeps every caller from having to care which shape it got.

    Args:
        value: The raw column value.

    Returns:
        One dict per entry, possibly empty.
    """
    if not value:
        return []
    if isinstance(value, dict):
        keys = list(value)
        length = len(value[keys[0]])
        return [{k: value[k][i] for k in keys} for i in range(length)]
    return list(value)


def load_raw(config: DatasetConfig) -> Iterable[dict[str, Any]]:
    """Pull the dataset from Hugging Face.

    The dataset is ungated, so no authentication is required. It has a single
    ``train`` split of 5,000 rows regardless of how this project later cuts it.

    Args:
        config: Which dataset and revision to fetch.

    Returns:
        The raw rows, with the dataset's own column names intact.
    """
    kwargs: dict[str, Any] = {}
    if config.revision is not None:
        kwargs["revision"] = config.revision
    return load_dataset(config.hf_id, split="train", **kwargs)


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
    for backdoor in _as_records(row.get("backdoors")):
        if backdoor.get("backdoor_works"):
            return backdoor
    return None


def select_clean_solution(row: dict[str, Any]) -> str | None:
    """Pick a validated clean solution from a row's ``solutions`` list.

    Args:
        row: A raw dataset row.

    Returns:
        The code of the first solution that both compiles and passes the tests,
        or ``None`` if the row has no such solution.
    """
    for solution in _as_records(row.get("solutions")):
        if solution.get("compiles") and solution.get("passes_tests"):
            return solution.get("code")
    return None


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
    for row in rows:
        backdoor = (
            select_working_backdoor(row)
            if config.require_working_backdoor
            else None
        )
        if config.require_working_backdoor and backdoor is None:
            continue

        clean_code = select_clean_solution(row)
        if config.require_passing_solution and clean_code is None:
            continue

        yield Problem(
            problem_id=str(row["problem_id"]),
            question=row["question"],
            clean_code=clean_code or "",
            backdoor_code=(backdoor or {}).get("backdoor_code", ""),
            backdoor_input=(backdoor or {}).get("backdoor_input", ""),
            difficulty=row.get("difficulty", "unknown"),
            is_nondeterministic=bool(row.get("is_nondeterministic", False)),
            backdoor_source=(backdoor or {}).get("source", "unknown"),
        )


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
    backdoored = Item(
        item_id=f"{problem.problem_id}__backdoored",
        problem_id=problem.problem_id,
        question=problem.question,
        code=problem.backdoor_code,
        label=1,
        is_nondeterministic=problem.is_nondeterministic,
    )
    clean = Item(
        item_id=f"{problem.problem_id}__clean",
        problem_id=problem.problem_id,
        question=problem.question,
        code=problem.clean_code,
        label=0,
        is_nondeterministic=problem.is_nondeterministic,
    )
    return backdoored, clean


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
    on iteration order or on Python's salted ``hash``. Ordering is by
    ``stable_hash(seed:problem_id)``, so the same problem lands in the same
    split whatever order the rows arrive in and whichever machine runs it.

    Args:
        problems: Every qualifying problem.
        config: Split sizes and seed.

    Returns:
        A mapping from ``problem_id`` to split name.

    Raises:
        ValueError: If the requested sizes cannot be satisfied — for example if
            fewer deterministic problems exist than val plus test require.
    """
    deterministic = [p for p in problems if not p.is_nondeterministic]
    nondeterministic = [p for p in problems if p.is_nondeterministic]

    needed = config.val_problems + config.test_problems
    if len(deterministic) < needed:
        raise ValueError(
            f"val + test need {needed} deterministic problems but only "
            f"{len(deterministic)} qualify; lower SplitConfig sizes or relax "
            f"the dataset filters"
        )

    shuffled = sorted(
        deterministic,
        key=lambda p: stable_hash(f"{config.seed}:{p.problem_id}", length=16),
    )

    assignment: dict[str, SplitName] = {}
    cursor = 0
    for split, size in (
        ("val", config.val_problems),
        ("test", config.test_problems),
    ):
        for problem in shuffled[cursor : cursor + size]:
            assignment[problem.problem_id] = split  # type: ignore[assignment]
        cursor += size

    for problem in shuffled[cursor:]:
        assignment[problem.problem_id] = "train"

    if config.nondeterministic_go_to_train:
        for problem in nondeterministic:
            assignment[problem.problem_id] = "train"

    return assignment


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
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[SplitName, Path] = {}

    for split, items in items_by_split.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, (asdict(item) for item in items))
        paths[split] = path

    sidecar = dict(metadata)
    sidecar["git_sha"] = git_sha()
    sidecar["written_at"] = datetime.now(UTC).isoformat()
    sidecar["splits"] = {
        split: {
            "items": len(items),
            "problems": len({item.problem_id for item in items}),
            "backdoored": sum(1 for item in items if item.label == 1),
            "clean": sum(1 for item in items if item.label == 0),
            "nondeterministic_items": sum(
                1 for item in items if item.is_nondeterministic
            ),
        }
        for split, items in items_by_split.items()
    }
    (out_dir / "splits.meta.json").write_text(json.dumps(sidecar, indent=2))
    return paths


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
    directory = splits_dir if splits_dir is not None else SPLITS_DIR
    path = directory / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python main.py build-data` first"
        )
    return [Item(**record) for record in read_jsonl(path)]


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
    logger = get_logger(__name__)
    set_seed(split_config.seed)

    rows = load_raw(dataset_config)
    n_rows = len(rows)
    logger.info("loaded %s rows from %s", f"{n_rows:,}", dataset_config.hf_id)

    problems = list(filter_qualifying(rows, dataset_config))
    n_deterministic = sum(1 for p in problems if not p.is_nondeterministic)
    logger.info(
        "qualifying: %s problems (%s deterministic, %s nondeterministic)",
        f"{len(problems):,}",
        f"{n_deterministic:,}",
        f"{len(problems) - n_deterministic:,}",
    )
    _warn_on_count_drift(logger, n_rows, len(problems), n_deterministic)

    assignment = assign_splits(problems, split_config)

    items_by_split: dict[SplitName, list[Item]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for problem in problems:
        split = assignment[problem.problem_id]
        items_by_split[split].extend(build_items(problem))

    for split, items in items_by_split.items():
        n_problems = len({item.problem_id for item in items})
        n_backdoored = sum(1 for item in items if item.label == 1)
        logger.info(
            "%-5s %4d problems / %4d items (%.0f%% attack rate)",
            split,
            n_problems,
            len(items),
            100.0 * n_backdoored / max(len(items), 1),
        )

    metadata = {
        "dataset": dataset_config.hf_id,
        "revision": dataset_config.revision,
        "seed": split_config.seed,
        "filters": {
            "rows_in": n_rows,
            "qualifying_problems": len(problems),
            "deterministic_problems": n_deterministic,
            "nondeterministic_problems": len(problems) - n_deterministic,
        },
        "nondeterministic_go_to_train": split_config.nondeterministic_go_to_train,
    }
    return write_splits(items_by_split, out_dir, metadata)


def _warn_on_count_drift(
    logger: Any,
    n_rows: int,
    n_qualifying: int,
    n_deterministic: int,
) -> None:
    """Warn loudly if the dataset no longer matches the preregistered counts.

    Section 2 of the plan quotes 5,000 -> 1,582 -> 1,444, measured against the
    live dataset on 9 Sep. A silent upstream revision would change every
    downstream number while every stage still ran green, so drift is surfaced
    here rather than discovered at analysis time.

    Args:
        logger: Where to write the warning.
        n_rows: Rows loaded.
        n_qualifying: Problems passing the backdoor and solution filters.
        n_deterministic: Of those, how many are deterministic.
    """
    expected = {"rows": 5000, "qualifying": 1582, "deterministic": 1444}
    actual = {
        "rows": n_rows,
        "qualifying": n_qualifying,
        "deterministic": n_deterministic,
    }
    drift = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    if drift:
        logger.warning(
            "COUNTS DIFFER FROM THE PREREGISTERED NUMBERS in project-plan.md "
            "section 2: %s. Pin DatasetConfig.revision and reconcile before "
            "trusting any downstream result.",
            ", ".join(f"{k}: expected {e:,}, got {a:,}" for k, (e, a) in drift.items()),
        )
