#!/usr/bin/env python3
"""Measure adaptive transform/reduction overhead against another checkout.

Example: python benchmarks/adaptive_precision.py --before /tmp/baseline --after .
Workers are isolated, with identical seeded input and warm-up; no CI timing gates.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import timeit


def worker(kind, size, repeats):
    import numpy as np
    from lvec import LVec
    rng = np.random.default_rng(391)
    p = rng.normal(size=(3, size))
    energy = np.sqrt(np.sum(p*p, axis=0)+4)
    values = (*p, energy)
    if kind == 'jagged':
        import awkward as ak
        counts = rng.integers(1, 8, size=size//4)
        counts = counts[counts.cumsum() <= size]
        used = int(counts.sum())
        values = tuple(ak.unflatten(x[:used], counts) for x in values)
        summed = values
    else:
        summed = tuple(x[:size//4*4].reshape(-1, 4) for x in values)
    vector = LVec(*values)
    collection = LVec(*summed)
    operations = {'boost': lambda: vector.boost(.1, .05, .02),
                  'rotation': lambda: vector.rotz(.47),
                  'sum': lambda: collection.sum(-1)}
    result = {'kind': kind, 'size': size}
    for name, operation in operations.items():
        operation()
        result[name+'_s'] = statistics.median(timeit.repeat(operation, number=3, repeat=repeats))/3
    return result


def probe_fallbacks(repeats):
    """Count slow entries separately from uninstrumented mixed-batch timings."""
    from unittest.mock import patch
    import numpy as np
    from lvec import LVec, _adaptive
    rows = []
    for kind in ('numpy', 'jagged'):
        for exceptional in (0, 1, 1000):
            rng = np.random.default_rng(391)
            p = rng.normal(size=(3, 100000))
            energy = np.sqrt(np.sum(p*p, axis=0)+4)
            p[0, :exceptional] = energy[:exceptional] = 1e308
            values = (*p, energy)
            if kind == 'jagged':
                import awkward as ak
                values = tuple(ak.unflatten(x, 4) for x in values)
            vector = LVec(*values)
            operation = lambda: vector.boost(-.9, 0., 0.)
            with patch.object(_adaptive, '_boost_decimal', wraps=_adaptive._boost_decimal) as slow:
                result = operation()
            assert slow.call_count == exceptional
            observed = (float(ak.flatten(result.E)[0]) if kind == 'jagged'
                        else float(result.E[0]))
            seconds = statistics.median(timeit.repeat(operation, number=1, repeat=repeats))
            rows.append({'kind': kind, 'size': 100000, 'exceptional': exceptional,
                         'recovered_entries': slow.call_count, 'boost_s': seconds,
                         'first_energy': observed})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before')
    parser.add_argument('--after', default='.')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 100000, 1000000])
    parser.add_argument('--kinds', nargs='+', default=['numpy', 'jagged'])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('KIND', 'SIZE'))
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe_fallbacks(args.repeats), indent=2))
        return
    if args.worker:
        print(json.dumps(worker(args.worker[0], int(args.worker[1]), args.repeats)))
        return
    if not args.before:
        parser.error('--before is required')
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'repeats': args.repeats, 'calls_per_repeat': 3, 'rows': []}
    for kind in args.kinds:
        for size in args.sizes:
            row = {'kind': kind, 'size': size}
            for label, source in [('before', args.before), ('after', args.after)]:
                env = dict(os.environ, PYTHONPATH=str(Path(source).resolve()))
                output = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                         '--worker', kind, str(size), '--repeats', str(args.repeats)],
                                        env=env, check=True, capture_output=True, text=True)
                row[label] = json.loads(output.stdout)
            row['after_over_before'] = {name: row['after'][name+'_s']/row['before'][name+'_s']
                                        for name in ('boost', 'rotation', 'sum')}
            result['rows'].append(row)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
