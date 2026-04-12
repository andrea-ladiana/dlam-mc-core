from __future__ import annotations

import argparse

import numpy as np

from .io import save_result_npz
from .simulation import run_simulation, setup_logger
from .types import SimParams


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DLAM Monte Carlo simulation.")
    parser.add_argument("--steps", type=int, default=50, help="Number of Monte Carlo steps.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature.")
    parser.add_argument(
        "--init-mode",
        choices=("disentangle", "reconstruction"),
        default="disentangle",
        help="Initialization strategy.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating point precision for core matrix operations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output NPZ path. If omitted, results are not written to disk.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger = setup_logger(level=20)
    params = SimParams(steps=args.steps, seed=args.seed, beta=args.beta, init_mode=args.init_mode)
    dtype = np.float32 if args.dtype == "float32" else np.float64

    logger.info("Running DLAM Monte Carlo core demo...")
    result = run_simulation(params, dtype=dtype)
    logger.info(
        "Final magnetizations: m1=%.3f, m2=%.3f, m3=%.3f",
        result.m1[-1],
        result.m2[-1],
        result.m3[-1],
    )

    if args.output:
        output_path = save_result_npz(result, args.output)
        logger.info("Saved simulation output to %s", output_path)


if __name__ == "__main__":
    main()
