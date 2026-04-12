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
- typed dataclass API for parameters and outputs,
- clear module split between generation, kernels, dynamics, and orchestration,
- portable pure-NumPy implementation,
- test suite and CI workflow included.

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

## License

MIT License, Copyright (c) 2026 Andrea Ladiana.

## Citation

See `CITATION.cff` for citation metadata.
