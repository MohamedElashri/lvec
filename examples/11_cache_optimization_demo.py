#!/usr/bin/env python
"""
Demonstration of LVEC memory optimization features.

This example shows how to use:
1. Cache size limits with LRU eviction
2. Time-to-live (TTL) expiration for cached values
"""

import time
import numpy as np
from lvec import LVec

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# Create vectors with different caching configurations
print_section("Creating vectors with different caching configurations")

# Regular vector with unlimited cache
regular_vec = LVec(1.0, 2.0, 3.0, 4.0)
print("Regular Vector (unlimited cache)")

# Vector with maximum cache size limit of 5 entries (LRU eviction)
size_limited_vec = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=5)
print("Size-limited Vector (max 5 entries, LRU eviction)")

# Vector with TTL of 2 seconds for cached values
ttl_vec = LVec(1.0, 2.0, 3.0, 4.0, default_ttl=2)
print("TTL Vector (2-second expiration for cached values)")

# Vector with both size limit and TTL
combined_vec = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=5, default_ttl=2)
print("Combined Vector (max 5 entries + 2-second TTL)")


# Demonstrate LRU cache eviction
print_section("Demonstrating LRU cache eviction")

# Fill cache with more than 5 properties - using valid properties from the LVec class
print("Accessing 6 different properties on the size-limited vector...")
properties = ['pt', 'eta', 'phi', 'mass', 'p', 'px']

for prop in properties:
    # Access each property
    getattr(size_limited_vec, prop)
    
# Check cache size and stats
print(f"Cache size: {size_limited_vec.cache_size}")
print("Cache stats:")
stats = size_limited_vec.cache_stats
print(f"- Total items in cache: {stats['overall']['cache_size']}")
print(f"- Max cache size: {stats['overall']['max_size']}")
print("- Properties in cache:")
for prop in properties:
    if prop in stats['properties']:
        print(f"  - {prop}: {'In cache' if prop in size_limited_vec._cache._values else 'Evicted'}")


# Demonstrate TTL expiration
print_section("Demonstrating TTL expiration")

# Access properties on TTL vector
print("Accessing properties on TTL vector (2-second expiration)...")
for prop in ['pt', 'eta', 'phi']:
    getattr(ttl_vec, prop)
    
print(f"Initial cache size: {ttl_vec.cache_size}")
print("Waiting for 3 seconds to let properties expire...")
time.sleep(3)

# Check expired items
expired_count = ttl_vec.clear_expired()
print(f"Expired items removed: {expired_count}")
print(f"Cache size after clearing expired items: {ttl_vec.cache_size}")


# Demonstrate setting custom TTL for specific properties
print_section("Setting custom TTL for specific properties")

# Set different TTLs for different properties
for vector, desc in [(combined_vec, "Combined vector")]:
    # Access all properties
    for prop in ['pt', 'eta', 'phi', 'mass', 'p']:
        getattr(vector, prop)
    
    # Set custom TTLs
    vector.set_ttl('pt', 1)  # 1 second
    vector.set_ttl('eta', 5)  # 5 seconds
    vector.set_ttl('phi', None)  # No expiration
    
    print(f"Initial {desc} cache size: {vector.cache_size}")
    
    # Wait for 2 seconds
    print("Waiting for 2 seconds...")
    time.sleep(2)
    
    # Check which properties expired
    expired_count = vector.clear_expired()
    print(f"Expired items removed: {expired_count}")
    print(f"Cache size after clearing: {vector.cache_size}")
    
    print("Properties status:")
    for prop in ['pt', 'eta', 'phi', 'mass', 'p']:
        status = "In cache" if prop in vector._cache._values else "Expired/Evicted"
        print(f"  - {prop}: {status}")
    
# Demonstrate cache hit/miss statistics
print_section("Cache hit/miss statistics")

# Create a new vector for clean stats
stats_vec = LVec(1.0, 2.0, 3.0, 4.0)

# Access properties multiple times
for _ in range(3):
    # First access will be a miss, subsequent ones will be hits
    stats_vec.pt
    stats_vec.eta
    stats_vec.mass

# Print hit ratio stats
print("Cache hit ratios:")
stats = stats_vec.cache_stats
print(f"Overall: {stats['overall']['hit_ratio']:.2f}")
for prop in ['pt', 'eta', 'mass']:
    ratio = stats['properties'][prop]['hit_ratio']
    hits = stats['properties'][prop]['hits']
    misses = stats['properties'][prop]['misses']
    print(f"{prop}: {ratio:.2f} ({hits} hits, {misses} misses)")

print("\nDemo complete!")
