from __future__ import annotations

import numpy as np

from .types import CouplingScalars, LayerPack


def compute_local_fields(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    *,
    layers: LayerPack,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    couplings: CouplingScalars,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the effective local fields for all three layers."""
    h1s1 = layers.L1.hat_xi @ s1.astype(np.float64, copy=False)
    h2s2 = layers.L2.hat_xi @ s2.astype(np.float64, copy=False)
    h3s3 = layers.L3.hat_xi @ s3.astype(np.float64, copy=False)

    v1 = couplings.f1 * (k1 @ h1s1) + couplings.c12 * h2s2 + couplings.c13 * h3s3
    v2 = couplings.f2 * (k2 @ h2s2) + couplings.c12 * h1s1 + couplings.c23 * h3s3
    v3 = couplings.f3 * (k3 @ h3s3) + couplings.c13 * h1s1 + couplings.c23 * h2s2

    h1 = layers.L1.hat_xi.T @ v1
    h2 = layers.L2.hat_xi.T @ v2
    h3 = layers.L3.hat_xi.T @ v3
    return h1, h2, h3


def glauber_step(
    state: np.ndarray,
    field: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perform one synchronous Glauber update for a binary layer."""
    probs = 0.5 * (1.0 + np.tanh(beta * field))
    draws = rng.random(state.shape[0])
    updated = np.where(draws < probs, 1, -1)
    return updated.astype(np.int8, copy=False)
