"""Run a grid of benchmark configurations, each in its own fresh process.

Each (variant, budget) pair is one ``python benchmark.py <overrides>`` process.
Separate processes are deliberate: running an embedder policy after another
policy in the same process has hung on CUDA initialization before (see the
review notes), and isolated processes also let independent runs proceed in
parallel. Selection-only runs (``experiment.generate=false``) are CPU-bound and
parallelize well; generation runs share one GPU, so keep ``--jobs 1`` for them.

Usage:
    python scripts/sweep.py --jobs 4 --budgets 256 512 \\
        --common model=qwen2.5-3b experiment.generate=false \\
        --variant policy=sliding_window \\
        --variant "policy=retrieval ++policy.packing=skip"

Each ``--variant`` is one space-separated override string. A per-run log is
written next to the results as ``<output_dir>/_logs/<variant>_b<budget>.log``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _slug(text: str) -> str:
    """Readable, filesystem-safe name for a variant, unique via a short hash
    (variants often differ only past the truncation point)."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return re.sub(r"[^0-9A-Za-z._=-]+", "_", text).strip("_")[:100] + "_" + digest


def run_one(overrides: list[str], log_path: Path, threads: int) -> tuple[int, float]:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONIOENCODING="utf-8")
    if "experiment.generate=false" in overrides:
        # Selection-only runs never need the GPU; hiding it keeps parallel runs
        # from each holding a CUDA context next to a generation run.
        env["CUDA_VISIBLE_DEVICES"] = ""
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, "benchmark.py", *overrides],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return code, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", action="append", required=True,
                        help="Space-separated Hydra overrides for one variant (repeatable).")
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--common", nargs="*", default=[],
                        help="Overrides applied to every run.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads per run (OMP/MKL).")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip runs whose log shows a completed run (resume a sweep).")
    args = parser.parse_args()

    output_dir = next((o.split("=", 1)[1] for o in args.common
                       if o.startswith("experiment.output_dir=")), "experiments/results")
    log_dir = ROOT / output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = [(variant, budget) for variant in args.variant for budget in args.budgets]
    if args.skip_done:
        def done(variant: str, budget: int) -> bool:
            log = log_dir / f"{_slug(variant)}_b{budget}.log"
            return log.is_file() and "artifacts   :" in log.read_text(encoding="utf-8", errors="replace")
        skipped = [r for r in runs if done(*r)]
        runs = [r for r in runs if r not in skipped]
        print(f"skipping {len(skipped)} completed run(s)", flush=True)
    print(f"{len(runs)} runs, {args.jobs} at a time; logs in {log_dir}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for variant, budget in runs:
            overrides = [*args.common, *variant.split(), f"budget={budget}"]
            log_path = log_dir / f"{_slug(variant)}_b{budget}.log"
            futures[pool.submit(run_one, overrides, log_path, args.threads)] = (variant, budget)
        for future in as_completed(futures):
            variant, budget = futures[future]
            code, seconds = future.result()
            failures += code != 0
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"[{status}] {variant} budget={budget} in {seconds / 60:.1f} min", flush=True)
    if failures:
        raise SystemExit(f"{failures} run(s) failed; see logs in {log_dir}")


if __name__ == "__main__":
    main()
