import dataclasses as dc

import numpy as np
import pytest

from dlam_mc_core import SimParams, run_multi_seed, run_simulation


def _small_params(**kwargs) -> SimParams:
    base = SimParams(
        N1=32,
        N2=32,
        N3=32,
        K=4,
        M1=6,
        M2=6,
        M3=6,
        r1=0.7,
        r2=0.7,
        r3=0.7,
        t1=2.0,
        t2=2.0,
        t3=2.0,
        beta=1.2,
        steps=8,
        seed=5,
        init_mode="disentangle",
        mu0=1,
    )
    return dc.replace(base, **kwargs)


def test_run_simulation_shapes_and_bounds() -> None:
    result = run_simulation(_small_params())

    assert result.m1.shape == (9,)
    assert result.m2.shape == (9,)
    assert result.m3.shape == (9,)
    assert result.s1.shape == (32,)
    assert result.s2.shape == (32,)
    assert result.s3.shape == (32,)

    for m in (result.m1, result.m2, result.m3):
        assert np.all(m <= 1.0 + 1e-12)
        assert np.all(m >= -1.0 - 1e-12)


def test_run_simulation_is_reproducible_with_same_seed() -> None:
    params = _small_params(seed=123)
    a = run_simulation(params)
    b = run_simulation(params)

    assert np.array_equal(a.m1, b.m1)
    assert np.array_equal(a.m2, b.m2)
    assert np.array_equal(a.m3, b.m3)
    assert np.array_equal(a.s1, b.s1)
    assert np.array_equal(a.s2, b.s2)
    assert np.array_equal(a.s3, b.s3)


def test_disentangle_requires_common_index_space() -> None:
    params = _small_params(N2=40, init_mode="disentangle")
    with pytest.raises(ValueError, match="requires N1 == N2 == N3"):
        run_simulation(params)


def test_reconstruction_supports_heterogeneous_layer_sizes() -> None:
    params = _small_params(N1=20, N2=28, N3=36, init_mode="reconstruction", cue_layer=2)
    result = run_simulation(params)

    assert result.s1.shape == (20,)
    assert result.s2.shape == (28,)
    assert result.s3.shape == (36,)


def test_result_exposes_reproducibility_metadata() -> None:
    result = run_simulation(_small_params(seed=77), dtype=np.float32)

    assert result.metadata.seed == 77
    assert result.metadata.dtype == "float32"
    assert result.metadata.params_digest
    assert len(result.metadata.params_digest) == 64
    assert "T" in result.metadata.created_at_utc


def test_run_multi_seed_matches_individual_runs() -> None:
    base = _small_params(steps=5)
    seeds = [13, 17, 23]

    batched = run_multi_seed(base, seeds, dtype=np.float32)
    singles = [run_simulation(dc.replace(base, seed=s), dtype=np.float32) for s in seeds]

    assert [res.metadata.seed for res in batched] == seeds
    for b_res, s_res in zip(batched, singles):
        assert np.array_equal(b_res.m1, s_res.m1)
        assert np.array_equal(b_res.m2, s_res.m2)
        assert np.array_equal(b_res.m3, s_res.m3)
