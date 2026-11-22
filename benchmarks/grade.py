"""Grade frozen rankings. This process only reads data files."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .protocol import (
    DEFAULT,
    TARGETS,
    VERSION,
    digest,
    new_directory,
    read_json,
    write_json,
)

DEFAULT_POLICY = "iforest_default"
RANDOM_POLICY = "uniform_model"


def load_trials(paths, truth):
    tasks = {key: {} for key in truth["tasks"]}
    provenance = []
    seen = {DEFAULT_POLICY, RANDOM_POLICY}
    seen.update(f"fixed/{name}" for name in truth["candidates"])
    for directory in paths:
        path = directory / "submission.json"
        trial = read_json(path)
        if trial["version"] != VERSION or trial["bank_id"] != truth["bank_id"]:
            raise ValueError(
                f"Submission belongs to a different benchmark bank: {path}"
            )
        if trial["tasks"].keys() != tasks.keys():
            raise ValueError(f"Incomplete or unexpected set of tasks in {path}")
        names = set(next(iter(trial["tasks"].values())))
        if not names or seen & names:
            raise ValueError(
                f"Empty or duplicate selector names in {path}: {seen & names}"
            )
        for key, results in trial["tasks"].items():
            if set(results) != names:
                raise ValueError(f"Missing selector results on {key}")
            for item in results.values():
                if item["status"] not in {"ok", "abstained", "failed", "unavailable"}:
                    raise ValueError("Unknown submission status")
                if item["status"] == "ok":
                    ranks = np.asarray(item["ranks"], dtype=float)
                    if ranks.shape != (len(truth["candidates"]),) or not np.array_equal(
                        ranks, rankdata(ranks)
                    ):
                        raise ValueError("Invalid candidate ranks")
            tasks[key].update(results)
        seen.update(names)
        provenance.append(
            {
                "sha256": digest(path),
                "methods": sorted(names),
                "hypothesis": trial["hypothesis"],
                **trial["provenance"],
                "selector_sha256": trial.get("selector_sha256"),
            }
        )
    return tasks, provenance


def evaluate_choice(item, outcomes, candidates, default):
    status = item["status"]
    ranks = np.asarray(item["ranks"]) if status == "ok" else None
    chosen = np.flatnonzero(ranks == ranks.min()) if ranks is not None else [default]
    quality = {
        name: float(np.mean(np.asarray(values)[chosen]))
        for name, values in outcomes.items()
    }
    diagnostics = {}
    for name, values in outcomes.items():
        best = max(values)
        diagnostics[f"{name}_regret"] = best - quality[name]
        diagnostics[f"{name}_best_pick_rate"] = float(
            np.mean(np.asarray(values)[chosen] == best)
        )
        correlation = None
        if ranks is not None and np.ptp(ranks) and np.ptp(values):
            correlation = float(spearmanr(-ranks, values).statistic)
        diagnostics[f"spearman_{name}"] = correlation
    return {
        **quality,
        "status": status,
        "selected": [candidates[i] for i in chosen],
        "reason": item.get("reason"),
        "seconds": item.get("seconds"),
        **diagnostics,
    }


def group_means(records, column):
    # First collapse repeat seeds, then treat variants of one source as a block.
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row[column] is not None:
            groups[row["source"]][row["dataset"]].append(row[column])
    return np.array(
        [
            np.mean([np.mean(v) for v in datasets.values()])
            for _, datasets in sorted(groups.items())
        ]
    )


def summarize(records, target="ap"):
    endpoints = [e for e in TARGETS if e in records[0]]
    columns = (
        *endpoints,
        *(f"{e}_regret" for e in endpoints),
        *(f"{e}_best_pick_rate" for e in endpoints),
        *(f"spearman_{e}" for e in endpoints),
        *(f"delta_{e}" for e in endpoints),
    )
    means = {c: group_means(records, c) for c in columns}
    delta = means[f"delta_{target}"]
    intervals = {f"delta_{e}_ci95": None for e in endpoints}
    if len(delta) >= 2:
        draws = np.random.default_rng(0).integers(len(delta), size=(5000, len(delta)))
        for e in endpoints:
            replicates = means[f"delta_{e}"][draws].mean(axis=1)
            intervals[f"delta_{e}_ci95"] = np.quantile(
                replicates, [0.025, 0.975]
            ).tolist()
    statuses = {
        s: sum(row["status"] == s for row in records)
        for s in ("ok", "abstained", "failed", "unavailable")
    }
    return {
        **{c: float(v.mean()) if len(v) else None for c, v in means.items()},
        **intervals,
        "sources": len(delta),
        "source_wins": int(np.sum(delta > 0)),
        "tasks": len(records),
        "status_counts": statuses,
    }


def grade(args):
    truth = read_json(args.truth / "truth.json")
    target = getattr(args, "target", "ap")
    submissions, provenance = load_trials(args.submissions, truth)
    candidates = truth["candidates"]
    default = candidates.index(DEFAULT)
    controls = {RANDOM_POLICY: {"status": "ok", "ranks": [1] * len(candidates)}}
    for index, name in enumerate(candidates):
        policy = DEFAULT_POLICY if index == default else f"fixed/{name}"
        controls[policy] = {
            "status": "ok",
            "ranks": rankdata(np.arange(len(candidates)) != index).tolist(),
        }
    records, skipped, supervised_winners = defaultdict(list), [], {}
    for key, task in truth["tasks"].items():
        outcomes = task["outcomes"]
        if outcomes is None:
            skipped.append(key)
            continue
        if target not in outcomes:
            raise ValueError(
                f"Target {target!r} is unavailable in this grading key. "
                "For F1, prepare a normal-only bank with calibrated alert thresholds."
            )
        best = {e: max(scores) for e, scores in outcomes.items()}
        if truth["split"] == "dev":
            supervised_winners[key] = [
                name
                for name, score in zip(candidates, outcomes[target], strict=True)
                if score == best[target]
            ]
        choices = {**submissions[key], **controls}
        assessed = {
            name: evaluate_choice(item, outcomes, candidates, default)
            for name, item in choices.items()
        }
        if args.reference not in assessed:
            raise ValueError(
                f"Reference selector missing from submissions: {args.reference}"
            )
        reference = assessed[args.reference]
        for name, values in assessed.items():
            records[name].append(
                {
                    **values,
                    "task": key,
                    "dataset": task["dataset"],
                    "source": task["source"],
                    "seed": task["seed"],
                    "n": task["n"],
                    **{k: task[k] for k in ("n_fit", "n_select") if k in task},
                    "features": task["features"],
                    **{f"oracle_{e}": value for e, value in best.items()},
                    **{f"delta_{e}": values[e] - reference[e] for e in outcomes},
                }
            )
    if not records:
        raise ValueError("No batch has both classes. Increase the label-blind row cap.")
    report = {
        "version": VERSION,
        "bank_id": truth["bank_id"],
        "track": truth.get("track", "transductive"),
        "split": truth["split"],
        "target": target,
        "reference": args.reference,
        "submissions": provenance,
        "grader_sha256": digest(Path(__file__)),
        "single_class_tasks": len(skipped),
        "summary": {
            name: summarize(rows, target) for name, rows in sorted(records.items())
        },
    }
    if truth["split"] == "dev":
        report["by_task"] = dict(records)
        report["supervised_winners"] = supervised_winners
        report["single_class_task_ids"] = skipped
    with new_directory(args.out) as output:
        write_json(output / "report.json", report)
    print(f"Results for the {report['track']} track, supervised target {target}")
    print(
        f"{'selector':36} {'score':>7} {'gap':>8} {'vs ref':>9} {'best pick':>10} {'scored':>9}"
    )
    for name, row in report["summary"].items():
        print(
            f"{name:36} {row[target]:7.3f} {row[f'{target}_regret']:8.3f} "
            f"{row[f'delta_{target}']:+9.3f} {row[f'{target}_best_pick_rate']:10.1%} "
            f"{row['status_counts']['ok']:5}/{row['tasks']}"
        )
    if skipped:
        print(f"{len(skipped)} single-class batches omitted for all selectors.")
