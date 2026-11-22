"""File formats shared by the preparer, selector runner and private grader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import rankdata

VERSION = 1
REVISION = "3dac8221081e190f157d78e93bfa8867f90d0965"
TAIL = 0.05
DEFAULT = "IForest/default"
# These ranking-based endpoints are available even on historical banks.
ENDPOINTS = ("ap", "auroc", "pr_auc_trapezoid", "precision_at_budget")
TARGETS = (*ENDPOINTS, "f1")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value) -> str:
    content = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def new_directory(target: Path) -> Iterator[Path]:
    """Publish a complete nonempty result, or leave the requested path untouched."""
    if target.exists():
        raise FileExistsError(f"Choose a new output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-", dir=target.parent) as staging:
        result = Path(staging) / "result"
        result.mkdir()
        yield result
        result.rename(target)


@dataclass(frozen=True)
class Batch:
    X: np.ndarray
    scores: np.ndarray  # candidate × sample, higher means more anomalous
    families: tuple[str, ...]
    views: dict[str, np.ndarray]


def read_batch(directory, key, checksum, families):
    path = directory / f"{key}.npz"
    if digest(path) != checksum:
        raise ValueError(f"Score bank checksum mismatch for {key}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    for values in arrays.values():
        values.setflags(write=False)
    return Batch(arrays.pop("X"), arrays.pop("scores"), tuple(families), arrays)


def ranking(values, size):
    """Direction is already oriented. Defined infinities can be ranked safely."""
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or np.isnan(values).any():
        raise ValueError(f"Expected {size} numeric candidate values without NaNs")
    return rankdata(-values, method="average")


def top_membership(scores, k):
    """Give equal fractional membership to scores tied across the budget boundary."""
    scores = np.asarray(scores)
    cutoff = np.partition(scores, -k)[-k]
    selected = (scores > cutoff).astype(float)
    equal = scores == cutoff
    selected[equal] = (k - selected.sum()) / equal.sum()
    return selected
