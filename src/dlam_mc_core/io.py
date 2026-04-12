from __future__ import annotations

import dataclasses as dc
import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import SimulationResult


def result_to_dict(result: SimulationResult) -> dict[str, Any]:
    """Convert a simulation result to a JSON-serializable dictionary."""
    return {
        "params": dc.asdict(result.params),
        "metadata": dc.asdict(result.metadata),
        "trajectories": {
            "m1": result.m1.tolist(),
            "m2": result.m2.tolist(),
            "m3": result.m3.tolist(),
        },
        "final_state": {
            "s1": result.s1.tolist(),
            "s2": result.s2.tolist(),
            "s3": result.s3.tolist(),
        },
    }


def save_result_npz(result: SimulationResult, output_path: str | Path) -> Path:
    """Persist trajectories, final states, params, and metadata in a compressed NPZ file."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        target,
        m1=result.m1,
        m2=result.m2,
        m3=result.m3,
        s1=result.s1,
        s2=result.s2,
        s3=result.s3,
        params_json=np.array(json.dumps(dc.asdict(result.params), sort_keys=True), dtype=np.str_),
        metadata_json=np.array(
            json.dumps(dc.asdict(result.metadata), sort_keys=True), dtype=np.str_
        ),
    )
    return target
