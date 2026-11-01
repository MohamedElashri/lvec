#!/usr/bin/env python3
"""Seeded, dependency-light benchmarks; run with PYTHONPATH=. from a checkout.

Run this same script against another checkout with PYTHONPATH=/path/to/checkout
for before/after comparisons. Timings and memory are measured independently.
"""

import argparse
import gc
import inspect
import json
import platform
import statistics
import timeit
import tracemalloc

import numpy as np

from lvec import LVec


def timing(operation, repeats):
    trial = timeit.timeit(operation, number=1)
    number = max(1, min(1000, int(0.02 / max(trial, 1e-9))))
    samples = timeit.repeat(operation, number=number, repeat=repeats)
    return {"median_seconds": statistics.median(samples) / number,
            "repeats": repeats, "operations_per_repeat": number}


def cache_memory(data, options):
    vector = LVec(*data, **options)
    gc.collect()
    tracemalloc.start()
    result = vector.mass
    del result
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Count actual retained cache buffers without needing new cache APIs on the
    # baseline checkout. This excludes components and Python bookkeeping.
    cache_bytes = sum(getattr(value, "nbytes", 0)
                      for value in vector._cache._values.values())
    return {"cache_array_bytes": cache_bytes,
            "traced_retained_bytes": retained, "traced_peak_bytes": peak,
            "cache_entries": len(vector._cache)}


def benchmark_numpy(size, rng, repeats, options):
    px, py, pz = rng.normal(size=(3, size))
    energy = np.sqrt(px * px + py * py + pz * pz + 1.)
    data = (px, py, pz, energy)
    vector = LVec(*data, **options)
    np.testing.assert_allclose(vector.mass, 1., atol=1e-12)

    def cold_mass():
        vector.clear_cache()
        return vector.mass

    def direct_mass():
        return np.sqrt(energy * energy - (px * px + py * py + pz * pz))

    def workflow():
        # New vectors and a fresh mass calculation on every call.
        return (vector + vector).boost(0.1, 0.05, 0.).mass

    np.testing.assert_allclose(workflow(), 2., atol=1e-12)
    cold = timing(cold_mass, repeats)
    _ = vector.mass
    return {"kind": "numpy", "size": size,
            "construction": timing(lambda: LVec(*data, **options), repeats),
            "mass_cold": cold,
            "mass_hot": timing(lambda: vector.mass, repeats),
            "mass_direct_numpy": timing(direct_mass, repeats),
            "add_boost_mass": timing(workflow, repeats),
            "mass_memory": cache_memory(data, options)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 1000000])
    parser.add_argument('--jagged-events', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--cache-intermediates', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes) or args.jagged_events < 0:
        parser.error('repeats/sizes must be positive and jagged-events non-negative')
    options = {}
    if args.cache_intermediates:
        if 'cache_intermediates' not in inspect.signature(LVec).parameters:
            parser.error('this checkout does not support cache_intermediates')
        options['cache_intermediates'] = True
    rng = np.random.default_rng(args.seed)
    environment = {"kind": "environment", "python": platform.python_version(),
                   "platform": platform.platform(), "numpy": np.__version__,
                   "lvec_source": inspect.getfile(LVec), "seed": args.seed,
                   "options": options}
    try:
        import awkward as ak
        environment['awkward'] = ak.__version__
    except ImportError:
        ak = None
    print(json.dumps(environment), flush=True)
    for size in args.sizes:
        print(json.dumps(benchmark_numpy(size, rng, args.repeats, options)), flush=True)
    if ak is not None and args.jagged_events:
        n = args.jagged_events
        counts = rng.integers(0, 5, size=n)
        px, py, pz = rng.normal(size=(3, counts.sum()))
        energy = np.sqrt(px * px + py * py + pz * pz + 1.)
        data = tuple(ak.unflatten(values, counts) for values in (px, py, pz, energy))
        vector = LVec(*data, **options)
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(vector.mass)), 1., atol=1e-12)
        def cold_mass():
            vector.clear_cache()
            return vector.mass

        cold = timing(cold_mass, args.repeats)
        _ = vector.mass
        print(json.dumps({"kind": "awkward", "events": n, "particles": int(counts.sum()),
                          "construction": timing(lambda: LVec(*data, **options), args.repeats),
                          "mass_cold": cold,
                          "mass_hot": timing(lambda: vector.mass, args.repeats),
                          "mass_memory": cache_memory(data, options)}), flush=True)


if __name__ == '__main__':
    main()
