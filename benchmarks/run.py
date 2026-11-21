"""Run selectors against public features and cached detector scores."""

import ast
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from time import perf_counter

from . import protocol, selectors, worker
from .protocol import (
    VERSION,
    digest,
    fingerprint,
    new_directory,
    read_batch,
    read_json,
    write_json,
)


def run_hypothesis(directory, key, snapshot, timeout):
    start = perf_counter()
    with TemporaryDirectory() as work, TemporaryFile() as log:
        result = Path(work) / "result.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.worker",
            str(directory.resolve()),
            key,
            str(snapshot.resolve()),
            str(result),
        ]
        try:
            process = subprocess.run(
                command,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=log,
                check=False,
            )
        except subprocess.TimeoutExpired:
            reason = f"Exceeded {timeout:g}s per-task budget"
        else:
            if process.returncode == 0 and result.is_file():
                return read_json(result)
            log.seek(max(0, log.tell() - 1600))
            reason = f"Worker exited {process.returncode}. {log.read().decode(errors='replace')}"
    return {"status": "failed", "reason": reason, "seconds": perf_counter() - start}


def run(args):
    bank = read_json(args.bank / "bank.json")
    bank_id = bank.pop("bank_id")
    if bank["version"] != VERSION or fingerprint(bank) != bank_id:
        raise ValueError("Invalid public bank manifest")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("The per-task timeout must be positive and finite")
    if args.ireos_rows is not None and (args.ireos_rows < 3 or args.selector):
        raise ValueError(
            "--ireos-rows needs at least 3 rows and applies to the built-in baselines"
        )
    with new_directory(args.out) as output:
        import labelfree

        library = Path(labelfree.__file__).parent
        provenance = {
            "library_version": labelfree.__version__,
            "library_sha256": fingerprint(
                {str(p.relative_to(library)): digest(p) for p in library.rglob("*.py")}
            ),
            "runner_sha256": fingerprint(
                [digest(Path(m.__file__)) for m in (protocol, selectors, worker)]
                + [digest(Path(__file__))]
            ),
        }
        trial = {
            "bank_id": bank_id,
            "version": VERSION,
            "provenance": provenance,
            "hypothesis": args.hypothesis,
            "tasks": {},
        }
        if args.selector:
            snapshot = output / "selector.py"
            snapshot.write_bytes(args.selector.read_bytes())
            sha = digest(snapshot)
            try:
                description = ast.get_docstring(ast.parse(snapshot.read_bytes()))
            except SyntaxError:
                description = (
                    None  # The worker records the import error as a failed trial.
                )
            name = f"{args.selector.stem}@{sha[:12]}"
            trial.update(
                selector_sha256=sha,
                hypothesis=args.hypothesis or description,
                timeout=args.timeout,
            )
        for key, checksum in bank["tasks"].items():
            if args.selector:
                if digest(args.bank / f"{key}.npz") != checksum:
                    raise ValueError(f"Score bank checksum mismatch for {key}")
                results = {name: run_hypothesis(args.bank, key, snapshot, args.timeout)}
            else:
                batch = read_batch(args.bank, key, checksum, bank["families"])
                results = selectors.baselines(batch, args.ireos_rows)
            trial["tasks"][key] = results
        write_json(output / "submission.json", trial)
    print(f"Saved {len(trial['tasks'])} ranked batches to {args.out}")
