"""Trusted construction of a public score bank and a private grading key."""

import csv
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from . import protocol
from .protocol import (
    DEFAULT,
    REVISION,
    TAIL,
    VERSION,
    digest,
    fingerprint,
    new_directory,
    top_membership,
    write_json,
)

MANIFEST = Path(__file__).with_name("adbench.csv")
ENDPOINTS = ("ap", "auroc", "pr_auc_trapezoid", "precision_at_budget")


def portfolio(seed):
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    return {
        DEFAULT: IForest(random_state=seed, n_jobs=1),
        "IForest/64": IForest(max_samples=64, random_state=seed, n_jobs=1),
        "KNN/5": KNN(n_neighbors=5),
        "KNN/25": KNN(n_neighbors=25),
        "LOF/5": LOF(n_neighbors=5),
        "LOF/25": LOF(n_neighbors=25),
        "HBOS/5": HBOS(n_bins=5),
        "HBOS/10": HBOS(n_bins=10),
    }


def datasets(split, requested):
    with MANIFEST.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    if requested:
        missing = set(requested) - {row["dataset"] for row in rows}
        if missing:
            raise ValueError(f"Unknown datasets for {split}: {sorted(missing)}")
        rows = [row for row in rows if row["dataset"] in requested]
    return rows


def verified_file(raw, row, download):
    path = raw / (row["dataset"] + ".npz")

    def verify(source):
        data = source.read_bytes()
        header = f"blob {len(data)}\0".encode()
        if hashlib.sha1(header + data).hexdigest() != row["git_blob"]:
            raise ValueError(f"Pinned ADBench file does not match: {path.name}")

    if path.is_file():
        verify(path)
        return path
    if not download:
        raise FileNotFoundError(
            f"{path} is missing. Supply the pinned file or use --download."
        )
    raw.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/Minqi824/ADBench/{REVISION}/adbench/datasets/Classical/{path.name}"
    with TemporaryDirectory(dir=raw) as tmp:
        downloaded = Path(tmp) / path.name
        with urlopen(url, timeout=90) as response, downloaded.open("wb") as stream:
            copyfileobj(response, stream)
        verify(downloaded)
        downloaded.replace(path)
    return path


def scale(train, *others):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(train)
    active = np.ptp(train, axis=0) > 0
    if not np.any(active):
        raise ValueError("The batch has no variable features")
    scaler = StandardScaler().fit(train[:, active])
    return [
        scaler.transform(train[:, active]),
        *[scaler.transform(imputer.transform(x)[:, active]) for x in others],
    ]


def extra_views(raw, seed):
    """Shared unlabeled holdout, Gaussian probes, uniform probes and five refits."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw))
    cut = int(len(raw) * 0.7)
    train, hold = scale(raw[order[:cut]], raw[order[cut:]])
    gaussian = rng.normal(train.mean(axis=0), train.std(axis=0), size=hold.shape)
    models = [m.fit(train) for m in portfolio(seed).values()]
    view = {
        "validation": np.stack([m.decision_function(hold) for m in models]),
        "gaussian": np.stack([m.decision_function(gaussian) for m in models]),
    }
    refits = []
    for _ in range(5):
        subset = rng.choice(len(train), int(0.8 * len(train)), replace=False)
        models = portfolio(int(rng.integers(2**31))).values()
        refits.append([m.fit(train[subset]).decision_function(hold) for m in models])
    view["refits"] = np.stack(refits, axis=1)

    # Scale each projected support to a unit box so volumes are comparable.
    # Box construction uses the unlabeled batch. Fits still use only train.
    real, uniform = [], []
    for _ in range(3 if train.shape[1] > 5 else 1):
        cols = rng.choice(train.shape[1], min(5, train.shape[1]), replace=False)
        both = np.vstack([train[:, cols], hold[:, cols]])
        lower, width = both.min(axis=0), np.ptp(both, axis=0)
        subtrain, subhold = (
            (train[:, cols] - lower) / width,
            (hold[:, cols] - lower) / width,
        )
        probes = rng.uniform(size=(2048, len(cols)))
        models = [m.fit(subtrain) for m in portfolio(seed).values()]
        real.append([m.decision_function(subhold) for m in models])
        uniform.append([m.decision_function(probes) for m in models])
    view["projected_real"], view["uniform"] = np.asarray(real), np.asarray(uniform)
    return view


def oracle(y, scores):
    if np.unique(y).size != 2:
        return None
    columns = {name: [] for name in ENDPOINTS}
    k = max(1, int(np.ceil(TAIL * len(y))))
    for s in scores:
        precision, recall, _ = precision_recall_curve(y, s)
        for name, value in zip(
            ENDPOINTS,
            (
                average_precision_score(y, s),
                roc_auc_score(y, s),
                auc(recall, precision),
                top_membership(s, k) @ y / k,
            ),
            strict=True,
        ):
            columns[name].append(float(value))
    return columns


def prepare(args):
    if (
        args.max_samples < 80
        or len(set(args.seeds)) != len(args.seeds)
        or any(s < 0 for s in args.seeds)
    ):
        raise ValueError("Use at least 80 samples and distinct non-negative seeds")
    with new_directory(args.out) as result:
        public, private = result / "public", result / "private"
        public.mkdir()
        private.mkdir(mode=0o700)
        names = list(portfolio(args.seeds[0]))
        bank = {
            "version": VERSION,
            "track": "transductive",
            "split": args.split,
            "revision": REVISION,
            "max_samples": args.max_samples,
            "seeds": args.seeds,
            "extra_views": args.extra_views,
            "candidates": names,
            "families": [name.split("/")[0] for name in names],
            "tail": TAIL,
            "versions": {
                pkg: version(pkg)
                for pkg in ("pyod", "scikit-learn", "scipy", "numpy", "numba")
            },
            "python": platform.python_version(),
            "prepare_sha256": digest(Path(__file__)),
            "protocol_sha256": digest(Path(protocol.__file__)),
            "manifest_sha256": digest(MANIFEST),
            "tasks": {},
        }
        truth = {}
        for row in datasets(args.split, args.datasets):
            path = verified_file(args.raw_dir, row, args.download)
            with np.load(path, allow_pickle=False) as data:
                raw, labels = data["X"], data["y"].reshape(-1)
            if (
                raw.ndim != 2
                or len(raw) != len(labels)
                or not np.isin(labels, [0, 1]).all()
            ):
                raise ValueError(
                    f"Invalid feature matrix or binary labels in {path.name}"
                )
            for seed in args.seeds:
                indices = np.random.default_rng(seed).permutation(len(raw))[
                    : args.max_samples
                ]
                batch, y = np.asarray(raw[indices], dtype=float), labels[indices]
                (X,) = scale(batch)
                scores = np.stack(
                    [m.fit(X).decision_scores_ for m in portfolio(seed).values()]
                )
                views = extra_views(batch, seed) if args.extra_views else {}
                arrays = {"X": X, "scores": scores, **views}
                if any(not np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError(f"Non-finite scores in {path.name}, seed {seed}")
                key = f"task-{len(truth):04d}"
                output = public / f"{key}.npz"
                np.savez_compressed(output, **arrays)
                bank["tasks"][key] = digest(output)
                truth[key] = {
                    "dataset": row["dataset"],
                    "source": row["source"],
                    "seed": seed,
                    "n": len(y),
                    "features": X.shape[1],
                    "positives": int(y.sum()),
                    "outcomes": oracle(y, scores),
                }
                print(f"Prepared {row['dataset']} seed {seed}", flush=True)
        bank_id = fingerprint(bank)
        write_json(public / "bank.json", {**bank, "bank_id": bank_id})
        write_json(
            private / "truth.json",
            {
                "bank_id": bank_id,
                "candidates": names,
                "split": args.split,
                "tasks": truth,
            },
        )
    print(f"Prepared {len(truth)} tasks for {args.split}. Bank {bank_id[:12]}")
