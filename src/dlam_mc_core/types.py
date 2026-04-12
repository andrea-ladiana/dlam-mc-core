from __future__ import annotations

import dataclasses as dc
from typing import Literal

import numpy as np

InitMode = Literal["reconstruction", "disentangle"]


@dc.dataclass(frozen=True)
class SimParams:
    """Simulation parameters for a 3-layer DLAM Monte Carlo experiment."""

    # Layer sizes and storage load
    N1: int = 400
    N2: int = 400
    N3: int = 400
    K: int = 40

    # Dataset sizes and qualities per layer
    M1: int = 20
    M2: int = 20
    M3: int = 20
    r1: float = 0.6
    r2: float = 0.6
    r3: float = 0.6

    # Dreaming times
    t1: float = 5.0
    t2: float = 5.0
    t3: float = 5.0

    # Couplings
    g11: float = 1.0
    g22: float = 1.0
    g33: float = 1.0
    g12: float = 1.0
    g13: float = 1.0
    g23: float = 1.0

    # Dynamics
    beta: float = 1.0
    steps: int = 100
    seed: int = 0

    # Initialization
    cue_layer: int = 1
    mu0: int = 0
    init_mode: InitMode = "disentangle"
    init_noise: float = 0.0

    def __post_init__(self) -> None:
        _ensure_positive_int("N1", self.N1)
        _ensure_positive_int("N2", self.N2)
        _ensure_positive_int("N3", self.N3)
        _ensure_positive_int("K", self.K)
        _ensure_positive_int("M1", self.M1)
        _ensure_positive_int("M2", self.M2)
        _ensure_positive_int("M3", self.M3)

        _ensure_probability_open_closed("r1", self.r1)
        _ensure_probability_open_closed("r2", self.r2)
        _ensure_probability_open_closed("r3", self.r3)

        _ensure_non_negative("t1", self.t1)
        _ensure_non_negative("t2", self.t2)
        _ensure_non_negative("t3", self.t3)

        _ensure_positive("beta", self.beta)

        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0 <= self.mu0 < self.K:
            raise ValueError("mu0 must satisfy 0 <= mu0 < K")
        if self.cue_layer not in (1, 2, 3):
            raise ValueError("cue_layer must be one of {1, 2, 3}")
        if self.init_mode not in ("reconstruction", "disentangle"):
            raise ValueError("init_mode must be 'reconstruction' or 'disentangle'")
        if not 0.0 <= self.init_noise <= 1.0:
            raise ValueError("init_noise must be in [0, 1]")

    @property
    def common_index_space(self) -> bool:
        """Return True if all layers share the same neuron index space."""
        return self.N1 == self.N2 == self.N3


@dc.dataclass(frozen=True)
class LayerData:
    """Data and statistics for one layer.

    Attributes
    ----------
    Xi:
        Array of shape (K, N) with latent archetypes in {-1, +1}.
    Eta:
        Array of shape (K, M, N) with noisy examples in {-1, +1}.
    U:
        Array of shape (K, N), empirical means over examples.
    hat_xi:
        Array of shape (K, N), normalized empirical archetypes.
    C:
        Array of shape (K, K), correlation matrix C = (hat_xi @ hat_xi.T) / N.
    """

    Xi: np.ndarray
    Eta: np.ndarray
    U: np.ndarray
    hat_xi: np.ndarray
    C: np.ndarray
    N: int
    M: int
    r: float
    rho: float


@dc.dataclass(frozen=True)
class LayerPack:
    """Container for the three layers used by the simulator."""

    L1: LayerData
    L2: LayerData
    L3: LayerData


@dc.dataclass(frozen=True)
class CouplingScalars:
    """Precomputed scalar factors entering local field equations."""

    f1: float
    f2: float
    f3: float
    c12: float
    c13: float
    c23: float


@dc.dataclass(frozen=True)
class SimulationResult:
    """Outputs of a simulation trajectory."""

    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    s3: np.ndarray
    layers: LayerPack
    params: SimParams


def _ensure_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a strictly positive integer")


def _ensure_probability_open_closed(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1]")


def _ensure_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
