"""Small shared helpers: I/O, determinism, provenance, logging.

Nothing here knows anything about monitors, backdoors or metrics. If a function
would need to import ``src.config`` to make sense, it belongs in the stage
module that uses it, not here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np 

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


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
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed every random source that could affect a result."""
    random.seed(seed)
    np.random.seed(seed)


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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
    raise NotImplementedError


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, replacing anything already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
                count += 1
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return count


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file, creating it if infra error during training runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
            f.flush()
            count += 1
    return count


def completed_ids(path: Path, id_field: str = "item_id") -> set[str]:
    """Read back which records a partially-finished job already produced."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)[id_field])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from a killed instance
    return done



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
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
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
    from src.config import RUNS_DIR

    base = root if root is not None else RUNS_DIR
    path = base / f"{date.today():%Y-%m-%d}__{arm}__{git_sha()}"
    path.mkdir(parents=True, exist_ok=True)
    return path

