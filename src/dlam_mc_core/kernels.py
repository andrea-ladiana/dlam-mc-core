from __future__ import annotations

import numpy as np

from .types import CouplingScalars, LayerPack, SimParams


def dreaming_core(C: np.ndarray, t: float) -> np.ndarray:
    """Compute the dreaming kernel (1+t) * (I + tC)^(-1)."""
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    if t < 0.0:
        raise ValueError("t must be >= 0")

    kdim = C.shape[0]
    eye = np.eye(kdim, dtype=C.dtype)
    return np.linalg.solve(eye + t * C, (1.0 + t) * eye)


def compute_coupling_scalars(layers: LayerPack, params: SimParams) -> CouplingScalars:
    """Precompute constants used in local field equations."""
    l1, l2, l3 = layers.L1, layers.L2, layers.L3

    f1 = params.g11 * (1.0 + l1.rho) / l1.N
    f2 = params.g22 * (1.0 + l2.rho) / l2.N
    f3 = params.g33 * (1.0 + l3.rho) / l3.N

    c12 = params.g12 * np.sqrt((1.0 + l1.rho) * (1.0 + l2.rho)) / np.sqrt(l1.N * l2.N)
    c13 = params.g13 * np.sqrt((1.0 + l1.rho) * (1.0 + l3.rho)) / np.sqrt(l1.N * l3.N)
    c23 = params.g23 * np.sqrt((1.0 + l2.rho) * (1.0 + l3.rho)) / np.sqrt(l2.N * l3.N)

    return CouplingScalars(f1=f1, f2=f2, f3=f3, c12=c12, c13=c13, c23=c23)
