"""CLI and isolated workers. Public output is Markdown; IPC stays in memory."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import timeit
import tracemalloc


ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("numpy", "strided", "jagged", "missing", "deep", "heavy-tail", "extreme")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def measure(operation, samples, min_time, warmups):
    for _ in range(warmups):
        operation()
    timer = timeit.Timer(operation)
    number = 1
    while True:
        duration = timer.timeit(number)
        if duration >= min_time or number >= 1_000_000:
            break
        number = min(1_000_000, number * max(2, min(10, math.ceil(min_time / max(duration, 1e-9)))))
    return {"seconds": [t / number for t in timer.repeat(samples, number)], "loops": number}


def allocations(operation):
    # Code paths are warmed before tracing; timing never runs under tracemalloc.
    operation()
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()  # Keep the result alive through both readings.
        retained, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
    return {"retained_bytes": retained, "peak_bytes": peak}


def provenance(source):
    import lvec
    import numpy as np
    digest = hashlib.sha256()
    for path in sorted((source / "lvec").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(["git", "-C", str(source), *args],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    versions = {"numpy": np.__version__}
    for package in ("awkward", "vector"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "versions": versions,
            "lvec_version": lvec.__version__, "source": str(source),
            "loaded_module": lvec.__file__, "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
            "source_sha256": digest.hexdigest(),
            "threads": {key: os.environ.get(key) for key in THREAD_VARS}}


def worker(config):
    source = Path(config["source"]).resolve()
    sys.path.insert(0, str(source))
    # Load framework cases by filename so --source changes only LVec, even if
    # that checkout has a different (or no) benchmarks package.
    spec = importlib.util.spec_from_file_location("lvec_benchmark_cases", Path(__file__).with_name("cases.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    info = provenance(source)
    if Path(info["loaded_module"]).resolve().parent != (source / "lvec").resolve():
        raise RuntimeError("worker imported LVec outside the requested source")
    cases, metadata = module.build_cases(config["layout"], config["size"],
                                          config["seed"], config["compare_vector"])
    if config["cases"]:
        unknown = set(config["cases"]) - {case.name for case in cases}
        if unknown:
            raise ValueError(f"unknown cases for {config['layout']}: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.name in config["cases"]]
    # Identical data across processes; different deterministic measurement order.
    random.Random(config["seed"] + config["process"]).shuffle(cases)
    rows = {}
    for spec in cases:
        case = spec.prepare()
        case.check(case.operation())
        if config["mode"] == "check":
            rows[case.name] = {"validated": True}
        elif config["mode"] == "memory":
            rows[case.name] = allocations(case.operation)
        else:
            rows[case.name] = measure(case.operation, config["samples"],
                                      config["min_time"], config["warmups"])
        case.check(case.operation())
        del case
    return {"environment": info, "input": metadata, "rows": rows}


def run_worker(config, timeout):
    env = dict(os.environ, PYTHONHASHSEED="0", **{key: "1" for key in THREAD_VARS})
    result = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(Path(__file__).resolve()),
                             "--worker", json.dumps(config)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{config['layout']}/{config['size']} ({config['mode']}):\n{result.stderr}")
    return json.loads(result.stdout)


def format_report(args, cells, environment):
    lines = ["# LVec benchmark run", "",
             f"UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             f"- Python: {environment['python']} ({environment['executable']})",
             f"- Platform: {environment['platform']}; processor: {environment['processor']}; logical CPUs: {environment['cpu_count']}",
             f"- Packages: {', '.join(f'{k} {v}' for k, v in environment['versions'].items())}; LVec {environment['lvec_version']}",
             f"- Source: `{environment['source']}`; revision: `{environment['revision']}`; dirty: {environment['dirty']}",
             f"- LVec source SHA256: `{environment['source_sha256']}`",
             f"- Framework SHA256: `{framework_hash()}`",
             f"- Seed: {args.seed}; processes/cell: {args.processes}; samples/process: {args.samples}; warmups/case: {args.warmups}; target sample: {args.min_time:g} s",
             f"- Threads: {', '.join(f'{k}={v}' for k, v in environment['threads'].items())}; PYTHONHASHSEED=0",
             f"- Invocation: `python -m benchmarks {' '.join(sys.argv[1:])}`", ""]
    if args.check:
        lines += ["Validation only; no performance measurements.", "",
                  "| Layout | Size | Particles generated / visible | Cases checked |",
                  "| --- | ---: | ---: | ---: |"]
        for layout, size, runs, _ in cells:
            info = runs[0]["input"]
            lines.append(f"| {layout} | {size} | {info['particles']} / {info['visible_particles']} | {len(runs[0]['rows'])} |")
        return "\n".join(lines) + "\n"
    lines += ["Times are microseconds/call: median and median absolute deviation (MAD) of pooled samples.",
              "Memory is a separate worker's incremental traced allocation, including the live result; it is not RSS.", "",
              "| Layout | Size | Particles generated / visible | Case | Median µs | MAD µs | Loops/sample range | Retained KiB | Peak KiB |",
              "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for layout, size, runs, memory in cells:
        info = runs[0]["input"]
        for name in sorted(runs[0]["rows"]):
            values = [t for run in runs for t in run["rows"][name]["seconds"]]
            median = statistics.median(values)
            mad = statistics.median(abs(t - median) for t in values)
            loops = [run["rows"][name]["loops"] for run in runs]
            allocated = memory["rows"][name] if memory else None
            retained = f"{allocated['retained_bytes'] / 1024:.2f}" if allocated else "—"
            peak = f"{allocated['peak_bytes'] / 1024:.2f}" if allocated else "—"
            lines.append(f"| {layout} | {size} | {info['particles']} / {info['visible_particles']} | {name} | {median * 1e6:.3f} | {mad * 1e6:.3f} | {min(loops)}–{max(loops)} | {retained} | {peak} |")
    lines += ["", "Raw samples (µs/call), grouped by fresh process, for local analysis:", ""]
    for layout, size, runs, _ in cells:
        for name in sorted(runs[0]["rows"]):
            values = " / ".join(", ".join(f"{t * 1e6:.6f}" for t in run["rows"][name]["seconds"])
                                for run in runs)
            lines.append(f"- {layout}/{size}/{name}: {values}")
    return "\n".join(lines) + "\n"


def framework_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    # Workers exchange JSON through pipes only; no result files are created.
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description="Validate and benchmark LVec; print a reproducible Markdown report.")
    parser.add_argument("--source", type=Path, default=ROOT, help="checkout to import LVec from")
    parser.add_argument("--layouts", choices=LAYOUTS, nargs="+", default=["numpy"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 1_000_000],
                        help="vectors for flat/extreme layouts; events for Awkward layouts")
    parser.add_argument("--cases", nargs="+", help="exact case names; see benchmarks/README.md")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--min-time", type=float, default=.02)
    parser.add_argument("--timeout", type=float, default=180, help="seconds per worker")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--compare-vector", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate workloads without timing")
    parser.add_argument("--output", type=Path, help="optional local Markdown report (.md)")
    args = parser.parse_args()
    if (min(args.sizes + [args.processes, args.samples, args.warmups]) < 1
            or not math.isfinite(args.min_time) or args.min_time <= 0
            or not math.isfinite(args.timeout) or args.timeout <= 0 or args.seed < 0):
        parser.error("sizes/counts/durations must be positive and finite; seed must be non-negative")
    args.source = args.source.resolve()
    if "extreme" in args.layouts and max(args.sizes) > 10_000:
        parser.error("extreme layout is capped at 10000 vectors; run it separately")
    if not (args.source / "lvec" / "__init__.py").is_file():
        parser.error("--source must contain lvec/__init__.py")
    if args.output and args.output.suffix != ".md":
        parser.error("--output must be a Markdown (.md) file")
    required = []
    if any(layout in {"jagged", "missing", "deep", "heavy-tail"} for layout in args.layouts):
        required.append("awkward")
    if args.compare_vector:
        required.append("vector")
        if not set(args.layouts) & {"numpy", "jagged"}:
            parser.error("--compare-vector requires a numpy or jagged layout")
    for package in required:
        if importlib.util.find_spec(package) is None:
            parser.error(f"{package} is required for this selection; install '.[benchmark]'")
    cells, environment = [], None
    try:
        for layout in dict.fromkeys(args.layouts):
            for size in dict.fromkeys(args.sizes):
                config = {"source": str(args.source), "layout": layout, "size": size,
                          "seed": args.seed, "compare_vector": args.compare_vector,
                          "cases": args.cases, "samples": args.samples,
                          "min_time": args.min_time, "warmups": args.warmups}
                runs = []
                for process in range(1 if args.check else args.processes):
                    print(f"Checking {layout}/{size}" if args.check else
                          f"Timing {layout}/{size}, process {process + 1}/{args.processes}", file=sys.stderr)
                    result = run_worker(dict(config, mode="check" if args.check else "timing",
                                             process=process), args.timeout)
                    if environment is not None and result["environment"] != environment:
                        raise RuntimeError("source or environment changed during the run")
                    environment = result["environment"]
                    runs.append(result)
                memory = None
                if not args.check and not args.no_memory:
                    print(f"Memory {layout}/{size}", file=sys.stderr)
                    memory = run_worker(dict(config, mode="memory", process=0), args.timeout)
                    if memory["environment"] != environment:
                        raise RuntimeError("source or environment changed during the run")
                cells.append((layout, size, runs, memory))
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")
    report = format_report(args, cells, environment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
