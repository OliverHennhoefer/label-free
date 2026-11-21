"""Run the registered paired-world experiment and freeze both benchmark selectors."""

import argparse
import platform
import shutil
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from benchmarks import protocol
from benchmarks.hypotheses import conditional_repair
from benchmarks.protocol import Batch, digest, new_directory, write_json


def experiment(seed):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=540)
    signs = np.repeat([-1.0, 1.0], 15)
    discordant = 2 * signs
    extremes = 4 * signs
    x = np.concatenate([normal, discordant, extremes])
    y = np.concatenate([normal, -discordant, extremes]) + rng.normal(0, 0.15, len(x))
    order = rng.permutation(len(x))
    X = np.column_stack([x, y])[order]
    scores = np.stack([np.abs(y - x), np.abs((x + y) / 2)])[:, order]
    batch = Batch(X, scores, ("discordance", "leverage"), {})
    repair = conditional_repair.select(batch)
    residual = conditional_repair.direct_residual(batch)

    # Both selectors have already returned. Regrade the identical rankings twice.
    groups = np.repeat([0, 1, 2], [540, 30, 30])[order]
    return {
        "seed": seed,
        "repair": repair.tolist(),
        "direct_residual": residual.tolist(),
        "discordance_ap": [
            float(average_precision_score(groups == 1, s)) for s in scores
        ],
        "extremes_ap": [float(average_precision_score(groups == 2, s)) for s in scores],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with new_directory(args.out) as out:
        design = out / "design.md"
        shutil.copyfile(Path(__file__).with_name("design.md"), design)
        candidate = out / "conditional_repair.py"
        shutil.copyfile(conditional_repair.__file__, candidate)
        # The runner fingerprints one standalone file per method. Freeze the rival
        # with its full predictor implementation so the two submissions are portable.
        rival = out / "direct_residual.py"
        rival.write_text(
            '"""High conditional prediction loss in the alerts identifies a good detector."""\n\n'
            + candidate.read_text()
            + "\nselect = direct_residual\n"
        )
        rows = [experiment(seed) for seed in range(20)]
        counts = {
            "repair_prefers_discordance": sum(
                r["repair"][0] > r["repair"][1] for r in rows
            ),
            "discordance_repair_positive": sum(r["repair"][0] > 0 for r in rows),
            "rival_prefers_discordance": sum(
                r["direct_residual"][0] > r["direct_residual"][1] for r in rows
            ),
            "ap_oracle_reversals": sum(
                r["discordance_ap"][0] > r["discordance_ap"][1]
                and r["extremes_ap"][1] > r["extremes_ap"][0]
                for r in rows
            ),
        }
        write_json(
            out / "probe.json",
            {
                "design_sha256": digest(design),
                "probe_sha256": digest(Path(__file__)),
                "protocol_sha256": digest(Path(protocol.__file__)),
                "selector_sha256": digest(candidate),
                "rival_sha256": digest(rival),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "counts": counts,
                "by_seed": rows,
            },
        )
    print(f"Results across {len(rows)} paired worlds: {counts}")


if __name__ == "__main__":
    main()
