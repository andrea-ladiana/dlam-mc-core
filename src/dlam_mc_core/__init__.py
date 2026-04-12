"""DLAM Monte Carlo core package."""

from .generation import generate_layer, generate_layer_pack
from .io import result_to_dict, save_result_npz
from .kernels import compute_coupling_scalars, dreaming_core
from .random import init_disentangle, rademacher
from .simulation import initialize_states, run_multi_seed, run_simulation, setup_logger
from .types import (
    CouplingScalars,
    LayerData,
    LayerPack,
    ReproducibilityMetadata,
    SimParams,
    SimulationResult,
)

__all__ = [
    "CouplingScalars",
    "LayerData",
    "LayerPack",
    "ReproducibilityMetadata",
    "SimParams",
    "SimulationResult",
    "compute_coupling_scalars",
    "dreaming_core",
    "generate_layer",
    "generate_layer_pack",
    "init_disentangle",
    "initialize_states",
    "rademacher",
    "result_to_dict",
    "run_multi_seed",
    "run_simulation",
    "save_result_npz",
    "setup_logger",
]
