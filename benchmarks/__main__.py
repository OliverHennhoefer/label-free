import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate label-free selectors for batch anomaly detection."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare",
        help="Build separate public and private artifacts from pinned ADBench files",
    )
    prep.add_argument("--raw-dir", type=Path, required=True)
    prep.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing files from the pinned ADBench revision",
    )
    prep.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    prep.add_argument(
        "--datasets", nargs="+", help="Optional ADBench stems within the selected split"
    )
    prep.add_argument("--seeds", type=int, nargs="+", default=[0])
    prep.add_argument("--max-samples", type=int, default=2000)
    prep.add_argument(
        "--extra-views",
        action="store_true",
        help="Cache refits and held-out reference scores for NPD, EM/MV and stability",
    )
    prep.add_argument("--out", type=Path, required=True)
    run = commands.add_parser(
        "run", help="Run the library baselines or one selector against a public bank"
    )
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Wall-clock seconds per custom selector per batch, including startup",
    )
    run.add_argument(
        "--selector", type=Path, help="Standalone Python module exporting select(batch)"
    )
    run.add_argument(
        "--hypothesis", help="Prediction or rationale to record with this submission"
    )
    run.add_argument(
        "--ireos-rows",
        type=int,
        help="Also evaluate shared-range IREOS on up to this many rows",
    )
    run.add_argument("--out", type=Path, required=True)
    grade = commands.add_parser(
        "grade", help="Read frozen submissions and the private grading key"
    )
    grade.add_argument("--truth", type=Path, required=True)
    grade.add_argument("--submissions", type=Path, nargs="+", required=True)
    grade.add_argument(
        "--reference",
        default="iforest_default",
        help="Selector for paired AP differences",
    )
    grade.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare as action
    elif args.command == "run":
        from .run import run as action
    else:
        from .grade import grade as action
    action(args)


if __name__ == "__main__":
    main()
