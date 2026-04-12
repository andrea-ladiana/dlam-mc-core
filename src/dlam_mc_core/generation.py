from __future__ import annotations

import numpy as np

from .random import rademacher
from .types import LayerData, LayerPack, SimParams


def generate_layer(
    *,
    K: int,
    N: int,
    M: int,
    r: float,
    rng: np.random.Generator,
    dtype=np.float64,
) -> LayerData:
    """Generate one data layer according to the DLAM definitions.

    Definitions
    -----------
    xi ~ Rademacher.
    eta = xi * chi, where chi in {-1, +1} and P(chi = +1) = (1 + r) / 2.
    U = (1 / M) sum_a eta_a.
    rho = (1 - r^2) / (M * r^2).
    hat_xi = U / (r * (1 + rho)).
    C = (hat_xi @ hat_xi.T) / N.
    """
    if K <= 0 or N <= 0 or M <= 0:
        raise ValueError("K, N and M must be strictly positive")
    if not (0.0 < r <= 1.0):
        raise ValueError("r must be in (0, 1]")

    xi = rademacher((K, N), rng, dtype=np.int8)

    p_plus = (1.0 + r) / 2.0
    chi = np.where(rng.random((K, M, N)) < p_plus, 1, -1).astype(np.int8)
    eta = (xi[:, None, :] * chi).astype(np.int8, copy=False)

    u = eta.mean(axis=1, dtype=np.float64)
    rho = (1.0 - r * r) / (M * r * r)
    hat_xi = (u / (r * (1.0 + rho))).astype(dtype, copy=False)
    c = (hat_xi @ hat_xi.T) / float(N)

    return LayerData(Xi=xi, Eta=eta, U=u, hat_xi=hat_xi, C=c, N=N, M=M, r=r, rho=rho)


def generate_layer_pack(params: SimParams, rng: np.random.Generator, dtype=np.float64) -> LayerPack:
    """Generate all three simulation layers from one parameter object."""
    l1 = generate_layer(K=params.K, N=params.N1, M=params.M1, r=params.r1, rng=rng, dtype=dtype)
    l2 = generate_layer(K=params.K, N=params.N2, M=params.M2, r=params.r2, rng=rng, dtype=dtype)
    l3 = generate_layer(K=params.K, N=params.N3, M=params.M3, r=params.r3, rng=rng, dtype=dtype)
    return LayerPack(L1=l1, L2=l2, L3=l3)
