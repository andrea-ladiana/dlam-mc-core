from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import dlam_mc_core.__main__ as module_main
import dlam_mc_core.simulation as simulation_module
from dlam_mc_core.types import SimParams


class _DummyLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        text = message % args if args else message
        self.messages.append(text)


def test_main_entrypoint_runs_with_mocked_dependencies(monkeypatch) -> None:
    logger = _DummyLogger()
    dummy_result = SimpleNamespace(
        m1=np.array([0.0, 0.1], dtype=np.float64),
        m2=np.array([0.0, 0.2], dtype=np.float64),
        m3=np.array([0.0, 0.3], dtype=np.float64),
    )

    monkeypatch.setattr(module_main, "setup_logger", lambda level=20: logger)
    monkeypatch.setattr(module_main, "run_simulation", lambda params: dummy_result)

    module_main.main()

    assert any("Running DLAM Monte Carlo core demo" in msg for msg in logger.messages)
    assert any("Final magnetizations" in msg for msg in logger.messages)


def test_private_metadata_helpers_cover_fallback_and_digest(monkeypatch) -> None:
    params = SimParams(seed=123, steps=2)

    def _raise_not_found(_name: str) -> str:
        raise simulation_module.PackageNotFoundError

    monkeypatch.setattr(simulation_module, "version", _raise_not_found)
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    v = simulation_module._resolve_simulator_version()
    digest_a = simulation_module._compute_params_digest(params)
    digest_b = simulation_module._compute_params_digest(params)
    metadata = simulation_module.build_reproducibility_metadata(params=params, dtype=np.float64)

    assert v == "0+unknown"
    assert digest_a == digest_b
    assert len(digest_a) == 64
    assert metadata.git_commit == "abc123"
    assert metadata.seed == 123
