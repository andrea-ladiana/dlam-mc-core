# Contributing Guide

Thank you for your interest in improving this project.

## Development setup

1. Create a Python environment (Python 3.10+).
2. Install the package in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

3. Run the test suite:

```bash
pytest
```

## Contribution principles

- Keep scientific definitions aligned with the model in the paper.
- Preserve reproducibility guarantees based on `SimParams.seed`.
- Add or update tests for every functional change.
- Keep public APIs documented through docstrings and README examples.

## Pull request checklist

- [ ] Code compiles and tests pass locally.
- [ ] Added tests for new behavior.
- [ ] Updated docs if behavior or API changed.
- [ ] Kept changes focused and minimal.
