from __future__ import annotations

from typing import Any

import numpy as np


def rademacher(shape: Any, rng: np.random.Generator, dtype=np.int8) -> np.ndarray:
    """Sample a Rademacher array in {-1, +1}."""
    draws = rng.integers(0, 2, size=shape, dtype=np.int8)
    out = np.where(draws == 0, -1, 1)
    return out.astype(dtype, copy=False)


def init_disentangle(
    xi1: np.ndarray,
    xi2: np.ndarray,
    xi3: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a common mixed cue and copy it to all layers.

    The cue is sign(xi1 + xi2 + xi3), with random tie-breaking on zeros and
    optional bit-flip noise controlled by epsilon.
    """
    if xi1.ndim != 1 or xi2.ndim != 1 or xi3.ndim != 1:
        raise ValueError("All xi arrays must be one-dimensional")
    if not (xi1.size == xi2.size == xi3.size):
        raise ValueError("Disentanglement initialization requires N1 == N2 == N3")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    mixed = xi1.astype(np.int16) + xi2.astype(np.int16) + xi3.astype(np.int16)
    cue = np.sign(mixed).astype(np.int8)

    zero_mask = cue == 0
    if np.any(zero_mask):
        cue[zero_mask] = rademacher(int(np.count_nonzero(zero_mask)), rng, dtype=np.int8)

    if epsilon > 0.0:
        flips = rng.random(cue.size) < epsilon
        cue[flips] *= -1

    return cue.copy(), cue.copy(), cue.copy()
