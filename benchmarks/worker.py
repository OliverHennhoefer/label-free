"""Execute one hypothesis on one public task in a fresh interpreter."""

import runpy
import sys
from pathlib import Path

from .protocol import read_batch, read_json, write_json
from .selectors import assess


def main():
    directory, key, selector, result = sys.argv[1:]
    bank = read_json(Path(directory) / "bank.json")
    batch = read_batch(Path(directory), key, bank["tasks"][key], bank["families"])

    def execute():
        return runpy.run_path(selector)["select"](batch)

    write_json(Path(result), assess(execute, len(batch.scores)))


if __name__ == "__main__":
    main()
