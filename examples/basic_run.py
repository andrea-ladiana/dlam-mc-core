from dlam_mc_core import SimParams, run_simulation


def main() -> None:
    params = SimParams(
        N1=200,
        N2=200,
        N3=200,
        K=20,
        M1=20,
        M2=20,
        M3=20,
        r1=0.65,
        r2=0.60,
        r3=0.55,
        t1=4.0,
        t2=4.0,
        t3=4.0,
        beta=1.8,
        steps=100,
        seed=42,
        init_mode="disentangle",
    )

    result = run_simulation(params)
    print("Final magnetizations:")
    print(f"m1={result.m1[-1]:.4f}, m2={result.m2[-1]:.4f}, m3={result.m3[-1]:.4f}")


if __name__ == "__main__":
    main()
