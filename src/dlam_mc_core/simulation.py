from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import logging
import os
import platform as _platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import numpy.typing as npt

from .dynamics import compute_local_fields, glauber_step
from .generation import generate_layer_pack
from .kernels import compute_coupling_scalars, dreaming_core
from .random import init_disentangle, rademacher
from .types import LayerPack, ReproducibilityMetadata, SimParams, SimulationResult


def setup_logger(name: str = "dlam_mc_core", level: int = logging.INFO) -> logging.Logger:
    """Create a stdout logger suitable for CLI and script usage."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _resolve_simulator_version() -> str:
    try:
        return version("dlam-mc-core")
    except PackageNotFoundError:
        return "0+unknown"


def _compute_params_digest(params: SimParams) -> str:
    payload = json.dumps(
        dc.asdict(params), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_reproducibility_metadata(
    params: SimParams, dtype: npt.DTypeLike
) -> ReproducibilityMetadata:
    """Build runtime metadata attached to every simulation output."""
    git_commit = os.environ.get("GITHUB_SHA")
    return ReproducibilityMetadata(
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        platform=_platform.platform(),
        simulator_version=_resolve_simulator_version(),
        dtype=np.dtype(dtype).name,
        seed=params.seed,
        params_digest=_compute_params_digest(params),
        git_commit=git_commit,
    )


def initialize_states(
    params: SimParams,
    layers: LayerPack,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize spin states according to the selected cue strategy."""
    mu0 = params.mu0

    if params.init_mode == "disentangle":
        if not params.common_index_space:
            raise ValueError(
                "init_mode='disentangle' requires N1 == N2 == N3 "
                "because a common cue is shared across layers"
            )
        return init_disentangle(
            layers.L1.Xi[mu0],
            layers.L2.Xi[mu0],
            layers.L3.Xi[mu0],
            epsilon=params.init_noise,
            rng=rng,
        )

    s1 = rademacher(params.N1, rng, dtype=np.int8)
    s2 = rademacher(params.N2, rng, dtype=np.int8)
    s3 = rademacher(params.N3, rng, dtype=np.int8)

    if params.cue_layer == 1:
        s1 = layers.L1.Xi[mu0].copy()
    elif params.cue_layer == 2:
        s2 = layers.L2.Xi[mu0].copy()
    else:
        s3 = layers.L3.Xi[mu0].copy()

    return s1, s2, s3


def run_simulation(params: SimParams, dtype: npt.DTypeLike = np.float64) -> SimulationResult:
    """Run one Monte Carlo trajectory for the 3-layer DLAM model."""
    resolved_dtype = np.dtype(dtype)
    rng = np.random.default_rng(params.seed)

    layers = generate_layer_pack(params, rng, dtype=resolved_dtype)
    couplings = compute_coupling_scalars(layers, params)

    k1 = dreaming_core(layers.L1.C.astype(resolved_dtype, copy=False), params.t1).astype(
        resolved_dtype, copy=False
    )
    k2 = dreaming_core(layers.L2.C.astype(resolved_dtype, copy=False), params.t2).astype(
        resolved_dtype, copy=False
    )
    k3 = dreaming_core(layers.L3.C.astype(resolved_dtype, copy=False), params.t3).astype(
        resolved_dtype, copy=False
    )

    s1, s2, s3 = initialize_states(params, layers, rng)

    m1 = np.empty(params.steps + 1, dtype=np.float64)
    m2 = np.empty(params.steps + 1, dtype=np.float64)
    m3 = np.empty(params.steps + 1, dtype=np.float64)

    target1 = layers.L1.Xi[params.mu0]
    target2 = layers.L2.Xi[params.mu0]
    target3 = layers.L3.Xi[params.mu0]

    m1[0] = np.mean((s1 * target1).astype(np.float64))
    m2[0] = np.mean((s2 * target2).astype(np.float64))
    m3[0] = np.mean((s3 * target3).astype(np.float64))

    for t in range(1, params.steps + 1):
        h1, h2, h3 = compute_local_fields(
            s1,
            s2,
            s3,
            layers=layers,
            k1=k1,
            k2=k2,
            k3=k3,
            couplings=couplings,
        )

        s1 = glauber_step(s1, h1, params.beta, rng)
        s2 = glauber_step(s2, h2, params.beta, rng)
        s3 = glauber_step(s3, h3, params.beta, rng)

        m1[t] = np.mean((s1 * target1).astype(np.float64))
        m2[t] = np.mean((s2 * target2).astype(np.float64))
        m3[t] = np.mean((s3 * target3).astype(np.float64))

    metadata = build_reproducibility_metadata(params=params, dtype=resolved_dtype)
    return SimulationResult(
        m1=m1,
        m2=m2,
        m3=m3,
        s1=s1,
        s2=s2,
        s3=s3,
        layers=layers,
        params=params,
        metadata=metadata,
    )


def run_multi_seed(
    base_params: SimParams,
    seeds: list[int],
    dtype: npt.DTypeLike = np.float64,
) -> list[SimulationResult]:
    """Run the same setup over multiple random seeds."""
    results: list[SimulationResult] = []
    for seed in seeds:
        params = dc.replace(base_params, seed=int(seed))
        results.append(run_simulation(params=params, dtype=dtype))
    return results
