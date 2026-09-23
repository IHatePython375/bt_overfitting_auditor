"""
Benchmark the C++ kernel against the numpy path.

    py scripts/bench_kernel.py
    py scripts/bench_kernel.py --strategies 200 --max-python 16

Exists so the numbers quoted in the README can be re-measured rather than
trusted. The Python path is skipped above --max-python because it costs about
78 us per partition on the development machine, which is 3.5 minutes at
C(24,12) and 13 hours at C(32,16); above the cutoff its column is an
extrapolation from the measured per-partition cost and is labelled as one.

Timings are single-shot, not a best-of-N. The point is the order of magnitude
between the paths, and a benchmark that quietly reports its luckiest run is a
worse guide to what an audit will actually cost.

READ THE LAST ROWS WITH SUSPICION ON A LAPTOP. The table runs from small S to
large, so the biggest cases execute last, after the machine has been at full
load for a while. On the development machine the S=24 single-thread row came
out at 5.8 s in sequence and 2.4 s run on its own -- the same work, 2.4x apart,
entirely thermal. For a number worth quoting, time one size in a fresh
process:  py scripts/bench_kernel.py --blocks 24
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from auditor.pbo import kernel_available, kernel_info, n_partitions, pbo


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the CSCV kernel.")
    ap.add_argument("--obs", type=int, default=2000, help="T, observations")
    ap.add_argument("--strategies", type=int, default=50, help="N, variants")
    ap.add_argument(
        "--blocks", type=int, nargs="+", default=[12, 16, 20, 22, 24],
        help="S values to time",
    )
    ap.add_argument(
        "--max-python", type=int, default=16,
        help="largest S to actually run the numpy path at",
    )
    args = ap.parse_args()

    if not kernel_available():
        raise SystemExit("kernel not built; run scripts/build_kernel.py")

    print(f"kernel : {kernel_info()}")
    print(f"data   : T={args.obs}, N={args.strategies}")
    print()

    rng = np.random.default_rng(0)
    m = rng.standard_normal((args.obs, args.strategies)) * 0.01 + 0.0002

    header = f"{'S':>3} {'partitions':>13} {'numpy':>11} {'cpp 1t':>9} {'cpp all':>9} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    us_per_partition = None
    for s in args.blocks:
        parts = n_partitions(s)

        if s <= args.max_python:
            t = time.perf_counter()
            ref = pbo(m, s, method="sufstats")
            py = time.perf_counter() - t
            us_per_partition = py / parts * 1e6
            py_label = f"{py:>10.2f}s"
        elif us_per_partition is not None:
            py = parts * us_per_partition / 1e6
            ref = None
            py_label = f"~{py:>9.0f}s"
        else:
            py, ref, py_label = float("nan"), None, "-".rjust(11)

        t = time.perf_counter()
        one = pbo(m, s, method="cpp", n_threads=1, store_lambdas=False)
        c1 = time.perf_counter() - t

        t = time.perf_counter()
        allc = pbo(m, s, method="cpp", n_threads=0, store_lambdas=False)
        cn = time.perf_counter() - t

        # A benchmark that does not check its own output is measuring how fast
        # it can be wrong.
        assert one["pbo"] == allc["pbo"]
        if ref is not None:
            assert ref["pbo"] == one["pbo"], (ref["pbo"], one["pbo"])

        print(
            f"{s:>3} {parts:>13,} {py_label} {c1:>8.2f}s {cn:>8.2f}s "
            f"{py / cn:>8.0f}x"
        )

    print()
    print("numpy times above --max-python are extrapolated at "
          f"{us_per_partition:.0f} us/partition, not measured.")
    print("later rows run under sustained load and throttle on a laptop; "
          "time one S in a fresh process for a quotable number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
