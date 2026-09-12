"""Small shared helpers: I/O, determinism, provenance, logging.

Nothing here knows anything about monitors, backdoors or metrics. If a function
would need to import ``src.config`` to make sense, it belongs in the stage
module that uses it, not here.
"""

from __future__ import annotations

import json, os, tempfile 
import logging
import hashlib 
import subprocess 
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured once for the whole project.

    Stages run for hours over SSH, so the format must carry a timestamp and be
    readable in a scrollback buffer. Calling this repeatedly with the same name
    must not attach duplicate handlers.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A logger writing to stderr at INFO level.
    """
    raise NotImplementedError


def set_seed(seed: int) -> None:
    """Seed every random source that could affect a result.

    Covers ``random`` and ``numpy``. Deliberately does **not** touch torch —
    the training modules own their own seeding, and importing torch here would
    make this module unimportable on the laptop.

    Args:
        seed: The seed to apply.
    """
    raise NotImplementedError


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file one record at a time.

    Yields rather than returning a list: generation files reach hundreds of
    thousands of lines and holding them all in memory on a 48 GB box competes
    with the vLLM KV cache.

    Args:
        path: File to read.

    Yields:
        One decoded object per line, in file order.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    raise NotImplementedError


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, replacing anything already there.

    Creates parent directories as needed. Writes to a temporary file and moves
    it into place, so an interrupted write never leaves a half-file that a
    later stage would silently treat as complete.

    Args:
        path: Destination file.
        records: Objects to serialise, one per line.

    Returns:
        How many records were written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
            count += 1
    os.replace(tmp, path)   # atomic on POSIX
    return count


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file, creating it if absent.

    This is the resumability primitive. Generation jobs run for hours on
    interruptible instances; appending each completed result immediately means
    a killed instance costs only the in-flight request rather than the run.

    Args:
        path: Destination file.
        records: Objects to append, one per line.

    Returns:
        How many records were appended.
    """
    raise NotImplementedError


def completed_ids(path: Path, id_field: str = "item_id") -> set[str]:
    """Read back which records a partially-finished job already produced.

    Lets a resumed generation stage skip work it has already paid for. Returns
    an empty set when the file does not exist, so callers need no special case
    for a first run.

    Args:
        path: The JSONL file a previous attempt was appending to.
        id_field: Field holding the identifier to collect.

    Returns:
        Every value of ``id_field`` found in the file.
    """
    raise NotImplementedError


def stable_hash(value: str, length: int = 8) -> str:
    """Hash a string deterministically across processes and machines.

    Python's built-in ``hash`` is salted per process, so it cannot be used to
    assign a problem to a split — the same problem would land in train on the
    laptop and test on the GPU box.

    Args:
        value: String to hash.
        length: How many leading hex characters to return.

    Returns:
        A lowercase hex digest prefix.
    """
    return hashlib.blake2b(value.encode(), digest_size=16).hexdigest()[:length]


def git_sha(short: bool = True) -> str:
    """Return the current commit sha, for stamping run directories and configs.

    Every result must be attributable to the code that produced it. Returns
    ``"unknown"`` rather than raising when git is unavailable or the tree is
    not a repository, so a run never dies over provenance.

    Args:
        short: Return the abbreviated sha rather than the full 40 characters.

    Returns:
        The sha, or ``"unknown"``.
    """
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    if short:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_dir(arm: str, root: Path | None = None) -> Path:
    """Create and return a fresh run directory named ``DATE__arm__githash``.

    M4 will be trained four or five times while debugging, and "which adapter
    produced the numbers in the draft?" always gets asked on the last day. The
    name answers it without a lookup.

    Args:
        arm: Arm id, e.g. ``"m4-sft"``.
        root: Override for ``config.RUNS_DIR``, for tests.

    Returns:
        The created directory.
    """
    raise NotImplementedError
