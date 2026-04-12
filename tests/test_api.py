import dlam_mc_core as core


def test_public_api_has_main_entrypoints() -> None:
    assert hasattr(core, "SimParams")
    assert hasattr(core, "run_simulation")
    assert hasattr(core, "generate_layer")
    assert hasattr(core, "dreaming_core")
