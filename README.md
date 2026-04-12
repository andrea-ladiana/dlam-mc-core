# DLAM Monte Carlo Core

A clean, publication-grade Python implementation of the **Monte Carlo simulation core** for the Dreaming L-directional Associative Memory (DLAM) model.

This repository intentionally focuses on the **core simulator** and excludes heavy numerical experiment pipelines, plotting batches, and manuscript-specific post-processing.

## Scientific scope

The simulator implements a 3-layer DLAM with:

- noisy supervised data generation per layer,
- dreaming kernel
  - `A_l(t_l) = (1+t_l) (I + t_l C_l)^{-1}`,
- hetero-associative coupling across layers,
- synchronous Glauber dynamics,
- trajectory-level magnetization tracking.

## Key features

- reproducible runs via explicit random seed control,
- per-run reproducibility metadata (runtime, dtype, parameter digest, optional git SHA),
- typed dataclass API for parameters and outputs,
- clear module split between generation, kernels, dynamics, and orchestration,
- portable pure-NumPy implementation,
- CLI entrypoint for quick scripted execution,
- test suite and CI workflow with lint/format/coverage gates included.

## Installation

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e .[dev]
```

## Quick start

```python
from dlam_mc_core import SimParams, run_simulation

params = SimParams(
    N1=300,
    N2=300,
    N3=300,
    K=30,
    steps=200,
    beta=1.5,
    seed=7,
    init_mode="disentangle",
)

result = run_simulation(params)
print(result.m1[-1], result.m2[-1], result.m3[-1])
```

Run the same setup across multiple seeds:

```python
from dlam_mc_core import SimParams, run_multi_seed

base = SimParams(steps=100, init_mode="disentangle")
results = run_multi_seed(base, seeds=[0, 1, 2, 3], dtype="float32")
print([r.m1[-1] for r in results])
```

## Command line usage

```bash
python -m dlam_mc_core --steps 200 --seed 7 --beta 1.5 --dtype float32
```

Optionally persist one run to NPZ:

```bash
python -m dlam_mc_core --steps 200 --seed 7 --output outputs/run_seed7.npz
```

## Reproducibility contract

Each `SimulationResult` carries a `metadata` object including:

- UTC run timestamp,
- Python and NumPy versions,
- platform descriptor,
- simulator package version,
- compute dtype,
- random seed,
- SHA-256 digest of all simulation parameters,
- optional `GITHUB_SHA` if present in environment.

This metadata is written by `save_result_npz(...)` alongside trajectories and final states.

## Project layout

```text
repository/
  src/dlam_mc_core/
    __init__.py
    __main__.py
    types.py
    random.py
    generation.py
    kernels.py
    dynamics.py
    simulation.py
  tests/
  docs/
  examples/
```

## Verification

Run tests with:

```bash
pytest
```

Local quality checks:

```bash
ruff check .
ruff format --check .
pytest
```

## License

MIT License, Copyright (c) 2026 Andrea Ladiana.

## Citation

See `CITATION.cff` for citation metadata.
