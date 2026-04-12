import json

import numpy as np

from dlam_mc_core import SimParams, result_to_dict, run_simulation, save_result_npz


def _params() -> SimParams:
    return SimParams(
        N1=20,
        N2=20,
        N3=20,
        K=3,
        M1=5,
        M2=5,
        M3=5,
        r1=0.7,
        r2=0.7,
        r3=0.7,
        steps=4,
        seed=11,
        init_mode="disentangle",
    )


def test_result_to_dict_has_expected_sections() -> None:
    result = run_simulation(_params())
    payload = result_to_dict(result)

    assert set(payload.keys()) == {"params", "metadata", "trajectories", "final_state"}
    assert len(payload["trajectories"]["m1"]) == 5
    assert len(payload["final_state"]["s1"]) == 20


def test_save_result_npz_roundtrip_metadata(tmp_path) -> None:
    result = run_simulation(_params(), dtype=np.float32)
    out_path = save_result_npz(result, tmp_path / "run.npz")

    data = np.load(out_path)
    metadata = json.loads(str(data["metadata_json"]))

    assert data["m1"].shape == (5,)
    assert metadata["dtype"] == "float32"
    assert metadata["seed"] == 11
