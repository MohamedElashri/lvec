#!/usr/bin/env python3
"""Process-isolated scaling, memory and reduction probes; emits JSON Lines.

Run with PYTHONPATH=/path/to/checkout to compare implementations. Sizes count
vectors for NumPy and events for Awkward. Missing/deep/heavy-tail cases are capped at
100k events to keep the default sweep bounded; no wall-clock CI assertions.
"""

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import timeit
import tracemalloc

import numpy as np

from lvec import LVec


def timed(function, repeats):
    function()
    return statistics.median(timeit.repeat(function, number=1, repeat=repeats))


def worker(kind, size, repeats):
    rng = np.random.default_rng(8102026)
    jagged = kind in {'jagged', 'missing', 'deep', 'heavy_tail'}
    if jagged:
        import awkward as ak
        counts = (np.minimum(rng.pareto(1.2, size).astype(np.int64), 1000)
                  if kind == 'heavy_tail' else rng.integers(0, 5, size=size))
        n = int(counts.sum())
    else:
        n = size
    p = rng.normal(size=(3, n * (2 if kind == 'strided' else 1)))
    if kind == 'strided':
        p = p[:, ::2]
    energy = np.sqrt(np.sum(p*p, axis=0) + 1.)
    if kind == 'photons':
        p[0] = rng.uniform(1, 100, n)
        p[1:] = 0.
        energy = p[0].copy()
    if kind == 'extreme':
        p[0], p[1], p[2] = 1e200, 1., 0.
        energy = np.full(n, 1e200)
    data = (*p, energy)
    if jagged:
        data = tuple(ak.unflatten(c, counts) for c in data)
        if kind == 'missing':
            particle_mask = rng.random(n) > .1
            event_mask = rng.random(size) > .1
            data = tuple(ak.mask(ak.unflatten(ak.mask(ak.flatten(c), particle_mask), counts),
                                 event_mask) for c in data)
        if kind == 'deep':
            for _ in range(3):
                data = tuple(ak.unflatten(c, 1) for c in data)

    v = LVec(*data)
    def cold_mass():
        v.clear_cache()
        return v.mass
    def workflow():
        return (v + v).boost(.1, .05, 0.).mass
    def direct_sum():
        reducer = ak.sum if jagged else np.sum
        return tuple(reducer(c, axis=-1) for c in data)
    cold = timed(cold_mass, repeats)
    v.mass
    hot = timeit.timeit(lambda: v.mass, number=10000) / 10000
    cache_bytes = sum(getattr(x, 'nbytes', 0) for x in v._cache._values.values())
    def cold_property(name):
        def calculate():
            v.clear_cache()
            return getattr(v, name)
        return timed(calculate, repeats)
    result = {
        'kind': kind, 'size': size, 'particles': n,
        'construction_s': timed(lambda: LVec(*data), repeats),
        'mass_cold_s': cold, 'mass_hot_s': hot,
        'pt_cold_s': cold_property('pt'), 'p_cold_s': cold_property('p'),
        'eta_cold_s': cold_property('eta'),
        'workflow_s': timed(workflow, repeats),
        'sum_s': timed(lambda: v.sum(axis=-1), repeats),
        'direct_sum_s': timed(direct_sum, repeats),
        'mass_cache_bytes': cache_bytes,
    }
    if kind == 'extreme':
        result['expected_mass2'] = -1.
        result['observed_mass2'] = float(v.mass2[0])
    v.clear_cache()
    gc.collect()
    tracemalloc.start()
    value = v.mass
    del value
    result['mass_traced_retained_bytes'], result['mass_traced_peak_bytes'] = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # ru_maxrss uses bytes on macOS and KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['process_peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss*1024)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 100000])
    parser.add_argument('--kinds', nargs='+', default=['numpy', 'strided', 'photons', 'jagged',
                                                   'missing', 'deep', 'heavy_tail'],
                        choices=['numpy', 'strided', 'photons', 'jagged', 'missing', 'deep', 'heavy_tail', 'extreme'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.sizes):
        parser.error('sizes and repeats must be positive')
    if args.worker:
        worker(args.kinds[0], args.sizes[0], args.repeats)
        return
    try:
        awkward_version = importlib.metadata.version('awkward')
    except importlib.metadata.PackageNotFoundError:
        awkward_version = None
    print(json.dumps({'kind': 'environment', 'python': platform.python_version(),
                      'numpy': np.__version__, 'awkward': awkward_version,
                      'lvec_source': inspect.getfile(LVec),
                      'platform': platform.platform(), 'seed': 8102026}), flush=True)
    failures = 0
    for kind in args.kinds:
        for size in args.sizes:
            if kind in {'missing', 'deep', 'heavy_tail'} and size > 100000:
                continue
            if kind == 'extreme' and size > 10000:
                continue
            command = [sys.executable, os.path.abspath(__file__), '--worker', '--sizes', str(size),
                       '--kinds', kind, '--repeats', str(args.repeats)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if completed.returncode:
                failures += 1
                print(json.dumps({'kind': kind, 'size': size, 'error': completed.stderr}), flush=True)
            else:
                print(completed.stdout.strip(), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
