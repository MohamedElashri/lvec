#!/usr/bin/env python3
"""Inspect LRU eviction, TTL expiration, and counters through the public API."""
import time

from lvec import LVec


def read(vector, names):
    for name in names:
        getattr(vector, name)


def main():
    print("LRU eviction")
    limited = LVec(1., 2., 3., 4., max_cache_size=5)
    properties = ['pt', 'eta', 'phi', 'mass', 'p', 'rapidity']
    read(limited, properties)
    print(f"Entries after six derived properties: {limited.cache_size} (limit 5)")
    before = limited.cache_stats['properties']['pt']['misses']
    _ = limited.pt
    after = limited.cache_stats['properties']['pt']['misses']
    print(f"Re-reading evicted pt required {after - before} new calculation")

    print("\nTTL expiration")
    expiring = LVec(1., 2., 3., 4., default_ttl=2.)
    read(expiring, ['pt', 'eta', 'phi'])
    print(f"Initial entries: {expiring.cache_size}")
    time.sleep(3)
    print(f"Expired entries removed: {expiring.clear_expired()}")
    print(f"Remaining entries: {expiring.cache_size}")

    print("\nPer-property lifetimes")
    custom = LVec(1., 2., 3., 4., default_ttl=2.)
    read(custom, ['pt', 'eta', 'phi', 'mass', 'p'])
    custom.set_ttl('pt', 1.)
    custom.set_ttl('eta', 5.)
    custom.set_ttl('phi', None)
    time.sleep(2)
    print(f"Expired entries removed: {custom.clear_expired()}")
    before = custom.cache_stats['properties']
    read(custom, ['pt', 'eta', 'phi'])
    after = custom.cache_stats['properties']
    for name in ['pt', 'eta', 'phi']:
        print(f"{name}: {after[name]['misses'] - before[name]['misses']} new calculations")

    print("\nCache counters")
    counted = LVec(1., 2., 3., 4.)
    for _ in range(3):
        read(counted, ['pt', 'eta', 'mass'])
    stats = counted.cache_stats
    print(f"Overall hit ratio: {stats['overall']['hit_ratio']:.2%}")
    for name in ['pt', 'eta', 'mass']:
        entry = stats['properties'][name]
        print(f"{name}: {entry['hits']} hits, {entry['misses']} misses")


if __name__ == '__main__':
    main()
