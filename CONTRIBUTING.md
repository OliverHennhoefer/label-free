# Contributing

Open an issue before large changes. For a focused fix:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Keep pull requests small, add one regression test for changed behavior, and cite
the paper or author implementation when changing a published metric.
