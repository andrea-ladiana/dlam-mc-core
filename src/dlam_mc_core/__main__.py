from __future__ import annotations

from .simulation import run_simulation, setup_logger
from .types import SimParams


def main() -> None:
    logger = setup_logger(level=20)
    params = SimParams(steps=50, seed=1)
    logger.info("Running DLAM Monte Carlo core demo...")
    result = run_simulation(params)
    logger.info(
        "Final magnetizations: m1=%.3f, m2=%.3f, m3=%.3f",
        result.m1[-1],
        result.m2[-1],
        result.m3[-1],
    )


if __name__ == "__main__":
    main()
