"""Dataset loading and the only label-aware split boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ADBENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/"
    "adbench/datasets/Classical"
)


@dataclass(frozen=True)
class BenchmarkSplit:
    development_normals: FloatArray
    test_features: FloatArray
    test_labels: IntArray
    development_normal_indices: IntArray
    test_indices: IntArray
    split_hash: str


def load_dataset(
    name: str,
    *,
    path: Path | None = None,
    data_dir: Path | None = None,
    verify_checksum: bool = False,
) -> tuple[FloatArray, IntArray]:
    """Load the built-in smoke data or an ADBench-style ``X``/``y`` NPZ."""
    if name == "synthetic":
        return _synthetic_dataset()

    resolved = path or _find_adbench_file(name, data_dir)
    if verify_checksum:
        _verify_registered_file(name, resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        if "X" not in archive or "y" not in archive:
            raise ValueError(f"{resolved} must contain X and y arrays")
        X = np.asarray(archive["X"], dtype=float)
        y = (np.asarray(archive["y"]).reshape(-1) != 0).astype(int)
    _validate_arrays(X, y, name)
    return X, y


def make_benchmark_split(
    X: FloatArray,
    y: IntArray,
    *,
    seed: int,
    test_fraction: float,
    max_normal_samples: int | None = None,
) -> BenchmarkSplit:
    """Stratify first, then expose only development normals upstream."""
    development_indices, test_indices = train_test_split(
        np.arange(len(X)),
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    development_normal_indices = development_indices[y[development_indices] == 0]
    if (
        max_normal_samples is not None
        and len(development_normal_indices) > max_normal_samples
    ):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            len(development_normal_indices), max_normal_samples, replace=False
        )
        development_normal_indices = development_normal_indices[chosen]
    development_normals = X[development_normal_indices]
    split_hash = _array_hash(development_normal_indices, test_indices)
    return BenchmarkSplit(
        development_normals=development_normals,
        test_features=X[test_indices],
        test_labels=y[test_indices],
        development_normal_indices=np.asarray(development_normal_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        split_hash=split_hash,
    )


def dataset_hash(X: FloatArray, y: IntArray) -> str:
    """Hash semantic array content independently of file path or NPZ encoding."""
    return _array_hash(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64))


def fetch_adbench_data(destination: Path, names: set[str] | None = None) -> list[Path]:
    """Download and verify the versioned ADBench files used by this benchmark."""
    registry = dataset_registry()
    selected = [row for row in registry if names is None or row["name"] in names]
    unknown = (names or set()) - {row["name"] for row in registry}
    if unknown:
        raise ValueError(f"unknown benchmark datasets: {', '.join(sorted(unknown))}")
    destination.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for row in selected:
        target = destination / row["file"]
        if target.exists() and _file_hash(target) == row["sha256"]:
            output.append(target)
            continue
        temporary = target.with_suffix(".npz.part")
        with urllib.request.urlopen(
            f"{ADBENCH_RAW_BASE}/{row['file']}", timeout=120
        ) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if _file_hash(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch while downloading {row['name']}")
        temporary.replace(target)
        output.append(target)
    return output


def dataset_registry() -> list[dict[str, str]]:
    """Return the immutable v1 dataset registry."""
    return json.loads(files("label_free_bench").joinpath("datasets.json").read_text())


def _verify_registered_file(name: str, path: Path) -> None:
    registry = dataset_registry()
    expected = next((row for row in registry if row["name"] == name), None)
    if expected is None:
        raise ValueError(f"dataset {name!r} is not in the v1 registry")
    if path.name != expected["file"] or _file_hash(path) != expected["sha256"]:
        raise ValueError(f"dataset {name!r} does not match the v1 checksum registry")


def split_development_normals(
    X: FloatArray, *, seed: int, validation_fraction: float
) -> tuple[FloatArray, FloatArray]:
    fit, validation = train_test_split(
        X, test_size=validation_fraction, random_state=seed
    )
    return fit, validation


def _find_adbench_file(name: str, data_dir: Path | None) -> Path:
    if data_dir is None:
        raise FileNotFoundError(
            f"dataset {name!r} needs either a path or a configured data_dir"
        )
    matches = sorted(data_dir.glob(f"*_{name}.npz")) + sorted(
        data_dir.glob(f"{name}.npz")
    )
    if not matches:
        raise FileNotFoundError(
            f"could not find ADBench dataset {name!r} in {data_dir}"
        )
    return matches[0]


def _validate_arrays(X: FloatArray, y: IntArray, name: str) -> None:
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"dataset {name!r} has incompatible X/y shapes")
    if len(X) < 20 or np.unique(y).size != 2:
        raise ValueError(f"dataset {name!r} needs enough normal and anomaly samples")
    if not np.isfinite(X).all():
        raise ValueError(f"dataset {name!r} contains non-finite features")


def _synthetic_dataset() -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(2026)
    normals = rng.normal(0.0, 1.0, size=(240, 6))
    anomalies = rng.normal(3.0, 1.2, size=(40, 6))
    X = np.vstack((normals, anomalies))
    y = np.concatenate(
        (np.zeros(len(normals), dtype=int), np.ones(len(anomalies), dtype=int))
    )
    return X, y


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
