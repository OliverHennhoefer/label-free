import csv
from argparse import Namespace

import numpy as np
import pytest

from benchmarks import prepare
from benchmarks.grade import grade, group_means
from benchmarks.protocol import (
    DEFAULT,
    VERSION,
    digest,
    fingerprint,
    read_json,
    top_membership,
    write_json,
)
from benchmarks.run import run


def test_manifest_keeps_variants_and_public_examples_together():
    rows = [
        r
        for split in ("dev", "validation", "test")
        for r in prepare.datasets(split, [])
    ]
    assert len(rows) == len({r["dataset"] for r in rows}) == 47
    for source in {r["source"] for r in rows}:
        assert len({r["split"] for r in rows if r["source"] == source}) == 1
    examples = {
        "2_annthyroid",
        "4_breastw",
        "6_cardio",
        "29_Pima",
        "38_thyroid",
        "42_WBC",
    }
    assert examples <= {r["dataset"] for r in prepare.datasets("dev", [])}
    with pytest.raises(ValueError, match="Unknown datasets"):
        prepare.datasets("test", ["6_cardio"])


def test_private_targets_and_threshold_ties():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([[1, 1, 1, 1], [0, 0, 1, 1]])
    values = prepare.oracle(labels, scores)
    assert values["ap"] == [0.25, 0.5]
    assert values["auroc"] == pytest.approx([0.5, 5 / 6])
    assert values["precision_at_budget"] == [0.25, 0.5]
    assert values["pr_auc_trapezoid"] != values["ap"]
    np.testing.assert_allclose(top_membership(scores[1], 1), [0, 0, 0.5, 0.5])
    assert prepare.oracle(np.zeros(4), scores) is None


def test_hierarchical_average():
    rows = [{"source": "a", "dataset": "x", "value": v} for v in [0.0, 0.0, 0.0]] + [
        {"source": "a", "dataset": "y", "value": 1.0},
        {"source": "b", "dataset": "z", "value": 1.0},
    ]
    np.testing.assert_array_equal(group_means(rows, "value"), [0.5, 1.0])


@pytest.fixture
def bank(tmp_path):
    public, private = tmp_path / "bank", tmp_path / "key"
    public.mkdir()
    private.mkdir()
    scores = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    file = public / "task.npz"
    np.savez(file, X=np.arange(8).reshape(4, 2), scores=scores)
    manifest = {
        "version": VERSION,
        "families": ["a", "b"],
        "tasks": {"task": digest(file)},
    }
    bank_id = fingerprint(manifest)
    write_json(public / "bank.json", {**manifest, "bank_id": bank_id})
    task = {
        "dataset": "x",
        "source": "x",
        "seed": 0,
        "n": 4,
        "features": 2,
        "outcomes": prepare.oracle(np.array([0, 0, 0, 1]), scores),
    }
    write_json(
        private / "truth.json",
        {
            "split": "dev",
            "bank_id": bank_id,
            "candidates": [DEFAULT, "other"],
            "tasks": {"task": task},
        },
    )
    return public, private


def test_freeze_grade_and_failures(tmp_path, bank):
    public, private = bank
    trials = []
    for name, expression in [
        ("flat", "[2,2]"),
        ("bad", "[float('nan'), 1]"),
        ("abstain", "None"),
    ]:
        module = tmp_path / f"{name}.py"
        module.write_text(
            f'"""Example hypothesis."""\ndef select(batch):\n'
            f'    assert not hasattr(batch, "y") and not batch.scores.flags.writeable\n'
            f"    return {expression}\n"
        )
        out = tmp_path / name
        args = Namespace(
            bank=public,
            out=out,
            selector=module,
            hypothesis=None,
            ireos_rows=None,
            timeout=30,
        )
        run(args)
        assert (out / "selector.py").read_bytes() == module.read_bytes()
        trials.append(out)
        with pytest.raises(FileExistsError):
            run(args)
    args = Namespace(
        truth=private,
        submissions=trials,
        out=tmp_path / "report",
        reference="iforest_default",
    )
    grade(args)
    report = read_json(args.out / "report.json")
    assert report["track"] == "transductive"  # Older private keys predate track names.
    flat, bad, abstain = (
        next(k for k in report["summary"] if k.startswith(n + "@"))
        for n in ("flat", "bad", "abstain")
    )
    assert report["summary"][flat]["ap"] == report["summary"]["uniform_model"]["ap"]
    assert report["summary"][bad]["status_counts"]["failed"] == 1
    assert report["summary"][abstain]["status_counts"]["abstained"] == 1
    assert report["summary"][bad]["ap"] == report["summary"]["iforest_default"]["ap"]
    assert report["summary"][flat]["delta_ap_ci95"] is None
    assert report["by_task"][flat][0]["selected"] == [DEFAULT, "other"]

    key = read_json(private / "truth.json")
    key["split"] = "test"
    write_json(private / "truth.json", key)
    args.out = tmp_path / "holdout-report"
    grade(args)
    assert "by_task" not in read_json(args.out / "report.json")
    key["bank_id"] = "wrong"
    write_json(private / "truth.json", key)
    with pytest.raises(ValueError, match="different benchmark bank"):
        grade(args)


def test_corrupt_bank_and_incomplete_submission(tmp_path, bank):
    public, private = bank
    plugin = tmp_path / "metric.py"
    plugin.write_text("def select(batch): return [1,2]")
    args = Namespace(
        bank=public,
        out=tmp_path / "out",
        selector=plugin,
        hypothesis="test",
        ireos_rows=None,
        timeout=30,
    )
    run(args)
    file = args.out / "submission.json"
    trial = read_json(file)
    trial["tasks"] = {}
    write_json(file, trial)
    with pytest.raises(ValueError, match="Incomplete"):
        grade(
            Namespace(
                truth=private,
                submissions=[args.out],
                out=tmp_path / "scores",
                reference="iforest_default",
            )
        )
    (public / "task.npz").write_bytes(b"broken")
    args.out = tmp_path / "second"
    with pytest.raises(ValueError, match="checksum"):
        run(args)
    assert not args.out.exists()


def test_hypothesis_has_time_budget(tmp_path, bank):
    public, _ = bank
    plugin = tmp_path / "slow.py"
    plugin.write_text("from time import sleep\ndef select(batch): sleep(60)")
    out = tmp_path / "timed"
    run(
        Namespace(
            bank=public,
            out=out,
            selector=plugin,
            hypothesis=None,
            ireos_rows=None,
            timeout=0.2,
        )
    )
    (result,) = read_json(out / "submission.json")["tasks"]["task"].values()
    assert result["status"] == "failed"
    assert "per-task budget" in result["reason"]


def test_pinned_download_rejects_wrong_blob(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    fake = tmp_path / "response"
    fake.write_bytes(b"wrong content")
    monkeypatch.setattr(prepare, "urlopen", lambda *a, **kw: fake.open("rb"))
    row = {"dataset": "fake", "git_blob": "0" * 40}
    with pytest.raises(ValueError, match="does not match"):
        prepare.verified_file(raw, row, True)
    assert list(raw.iterdir()) == []


def test_prepare_cli_defaults_to_normal_only(monkeypatch):
    import sys

    from benchmarks.__main__ import main

    received = []
    monkeypatch.setattr(prepare, "prepare", lambda args: received.append(args.track))
    monkeypatch.setattr(
        sys, "argv", ["benchmarks", "prepare", "--raw-dir", "raw", "--out", "bank"]
    )
    main()
    assert received == ["normal-only"]


def test_changing_private_measurements_changes_only_the_grades(monkeypatch):
    class FirstFeature:
        def fit(self, X):
            self.mean = X.mean(axis=0)
            return self

        def decision_function(self, X):
            return (X - self.mean)[:, 0]

    monkeypatch.setattr(prepare, "portfolio", lambda seed: {"probe": FirstFeature()})
    X = np.random.default_rng(6).normal(size=(192, 4))
    y = (np.arange(len(X)) % 4 == 0).astype(int)
    fit, selection, test = prepare.normal_partitions(y)
    assert not y[fit].any() and not y[selection].any()
    assert len(np.unique(np.r_[fit, selection, test])) == len(fit) + len(
        selection
    ) + len(test)
    assert set(y[test]) == {0, 1}
    X[y == 1, 0] = 100
    X[:, -1] = 0
    changed = X.copy()
    changed[test] += (
        10  # Even private normal observations must be invisible at selection.
    )
    changed[y == 1] = -100
    public, labels, hidden = prepare.normal_only_task(X, y, 6, True)
    public_again, labels_again, hidden_again = prepare.normal_only_task(
        changed, y, 6, True
    )
    assert public["X"].shape == (len(selection), 3)
    for key in public:
        np.testing.assert_array_equal(public[key], public_again[key])
    assert prepare.oracle(labels, hidden)["auroc"] == [1.0]
    assert prepare.oracle(labels_again, hidden_again)["auroc"] == [0.0]


@pytest.mark.parametrize("track", ["normal-only", "transductive"])
def test_prepare_and_baselines_on_pyod(tmp_path, monkeypatch, track):
    pytest.importorskip("pyod")
    import hashlib

    from benchmarks.protocol import read_batch

    path = tmp_path / "fixture.npz"
    rng = np.random.default_rng(12)
    X = rng.normal(size=(280, 6))
    X[:3, 2] = np.nan
    X[:, 5] = 1
    y = (np.arange(len(X)) % 7 == 0).astype(int)
    np.savez(path, X=X, y=y)
    data = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    manifest = tmp_path / "datasets.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(
            [
                ["dataset", "source", "split", "git_blob"],
                ["fixture", "group", "dev", blob],
            ]
        )
    monkeypatch.setattr(prepare, "MANIFEST", manifest)
    args = Namespace(
        raw_dir=tmp_path,
        datasets=["fixture"],
        split="dev",
        track=track,
        seeds=[5],
        max_samples=256 if track == "normal-only" else 128,
        extra_views=True,
        download=False,
        out=tmp_path / "prepared",
    )
    prepare.prepare(args)
    public = args.out / "public"
    bank = read_json(public / "bank.json")
    key, sha = next(iter(bank["tasks"].items()))
    batch = read_batch(public, key, sha, bank["families"])
    truth = read_json(args.out / "private" / "truth.json")
    task = truth["tasks"][key]
    assert bank["track"] == truth["track"] == track
    expected_views = {
        "validation",
        "gaussian",
        "refits",
        "projected_real",
        "uniform",
    }
    if track == "normal-only":
        expected_views.add("train")
        sampled = np.random.default_rng(5).permutation(len(y))[: args.max_samples]
        fit, select, test = prepare.normal_partitions(y[sampled])
        tr, sel, hidden = prepare.scale(
            X[sampled[fit]], X[sampled[select]], X[sampled[test]]
        )
        np.testing.assert_allclose(batch.views["train"], tr)
        np.testing.assert_allclose(batch.X, sel)
        assert (task["n_fit"], task["n_select"], task["n"]) == (
            len(tr),
            len(sel),
            len(hidden),
        )
        models = [m.fit(tr) for m in prepare.portfolio(5).values()]
        np.testing.assert_allclose(
            batch.scores, [m.decision_function(sel) for m in models]
        )
        expected = prepare.oracle(
            y[sampled[test]], [m.decision_function(hidden) for m in models]
        )
        for name in expected:
            np.testing.assert_allclose(task["outcomes"][name], expected[name])
    else:
        assert batch.X.shape == (128, 5)
    assert set(batch.views) == expected_views
    assert set(np.load(public / f"{key}.npz").files) == {"X", "scores", *batch.views}
    assert "fixture" not in (public / "bank.json").read_text()
    trial = tmp_path / "baselines"
    run(
        Namespace(
            bank=public,
            out=trial,
            selector=None,
            hypothesis=None,
            ireos_rows=None,
            timeout=30,
        )
    )
    submission = read_json(trial / "submission.json")
    assert submission["track"] == track
    outcomes = submission["tasks"][key]
    assert all(r["status"] == "ok" for r in outcomes.values()), outcomes
    assert len(outcomes) >= 28
    assert set(task["outcomes"]) == set(prepare.ENDPOINTS)
    out = tmp_path / "grade"
    grade(
        Namespace(
            truth=args.out / "private",
            submissions=[trial],
            out=out,
            reference="iforest_default",
        )
    )
    report = read_json(out / "report.json")
    assert report["track"] == track
    assert report["summary"]["iforest_default"]["ap"] == task["outcomes"]["ap"][0]
    if track == "normal-only":
        assert report["by_task"]["iforest_default"][0]["n_fit"] == task["n_fit"]
