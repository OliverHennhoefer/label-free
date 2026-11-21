"""Compare frozen methods source by source in a development report."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.grade import group_means


def compare(report, selector, reference, endpoint):
    methods = report["by_task"]
    baseline = {r["task"]: r[endpoint] for r in methods[reference]}
    rows = [
        {**r, "difference": r[endpoint] - baseline[r["task"]]}
        for r in methods[selector]
    ]
    # group_means first averages repeats within each dataset, then datasets
    # within each source. The returned array follows sorted source names.
    sources = sorted({r["source"] for r in rows})
    differences = group_means(rows, "difference")
    result = {
        "source_differences": dict(zip(sources, differences.tolist(), strict=True)),
        "mean_difference": float(differences.mean()),
        "wins_ties_losses": [
            int(np.sum(differences > 1e-12)),
            int(np.sum(np.abs(differences) <= 1e-12)),
            int(np.sum(differences < -1e-12)),
        ],
    }
    if len(sources) > 1:
        replicates = np.random.default_rng(0).choice(
            differences, (5000, len(sources)), replace=True
        )
        result["source_bootstrap_ci95"] = np.quantile(
            replicates.mean(axis=1), [0.025, 0.975]
        ).tolist()
        omissions = (differences.sum() - differences) / (len(sources) - 1)
        result["leave_one_source_out_range"] = [
            float(omissions.min()),
            float(omissions.max()),
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--selectors", nargs="+", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument(
        "--endpoint",
        default="ap",
        choices=["ap", "auroc", "precision_at_budget", "pr_auc_trapezoid"],
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    results = {
        s: {r: compare(report, s, r, args.endpoint) for r in args.references}
        for s in args.selectors
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
