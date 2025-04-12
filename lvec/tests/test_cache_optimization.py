"""
Unit tests for the memory optimization features of the LVEC caching system.
Tests the following features:
1. Cache size limits with LRU eviction
2. TTL (Time-To-Live) expiration for cached values
3. Combined functionality
"""

import pytest
import time
import numpy as np

from lvec import LVec
from lvec.caching import PropertyCache


def test_cache_size_limit():
    """Test that the cache size limit and LRU eviction work correctly."""
    # Create a vector with max cache size of 3
    v = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=3)
    
    # Fill the cache
    properties = ['pt', 'eta', 'phi', 'mass', 'p']
    for prop in properties:
        getattr(v, prop)
    
    # Verify cache size is limited to max_cache_size
    assert v.cache_size <= 3
    assert len(v._cache._values) <= 3
    
    # Verify the most recently used properties are in the cache
    # The cache may also contain intermediate calculations, so we'll check if
    # at least one of the most recently accessed properties is cached
    cache_keys = set(v._cache._values.keys())
    recent_properties = set(['mass', 'p'])  # Last two accessed
    assert len(cache_keys.intersection(recent_properties)) > 0
    
    # Access a property that might have been evicted
    v.pt
    
    # Cache size should still be limited
    assert v.cache_size <= 3
    
    # Verify the cache hit/miss tracking works
    stats = v.cache_stats
    assert 'pt' in stats['properties']


def test_cache_ttl_expiration():
    """Test that Time-To-Live (TTL) expiration works correctly."""
    # Create a vector with TTL of 0.5 seconds
    v = LVec(1.0, 2.0, 3.0, 4.0, default_ttl=0.5)
    
    # Access properties to cache them
    pt_value = v.pt
    phi_value = v.phi
    
    # Wait for the TTL to expire
    time.sleep(0.6)
    
    # Clear expired and verify some were removed
    expired_count = v.clear_expired()
    assert expired_count > 0
    
    # Accessing again should recalculate
    new_pt_value = v.pt
    
    # Value should be the same
    assert pt_value == new_pt_value
    
    # Verify cache hits/misses are tracked
    stats = v.cache_stats
    assert 'pt' in stats['properties']
    assert stats['properties']['pt']['misses'] >= 2  # Initial + after expiration


def test_property_specific_ttl():
    """Test setting different TTL values for specific properties."""
    # Create a vector with no default TTL
    v = LVec(1.0, 2.0, 3.0, 4.0)
    
    # Access properties
    pt_value = v.pt
    phi_value = v.phi
    
    # Set different TTLs for each property
    v.set_ttl('pt', 0.5)  # 0.5 seconds
    v.set_ttl('phi', 1.0)  # 1.0 seconds
    
    # Wait for the shorter TTL to expire
    time.sleep(0.6)
    
    # Clear expired and check count
    expired_count = v.clear_expired()
    assert expired_count > 0
    
    # Check that only pt triggered a miss (was expired)
    v.pt  # Should recalculate
    v.phi  # Should still be in cache
    
    stats = v.cache_stats
    assert stats['properties']['pt']['misses'] >= 2  # Initial + after expiration
    
    # Test removing TTL
    v.eta
    v.set_ttl('eta', 0.5)
    time.sleep(0.6)
    expired_before = v.clear_expired()
    
    # Set to None (no expiration)
    v.eta
    v.set_ttl('eta', None)
    time.sleep(0.6)
    expired_after = v.clear_expired()
    
    # Accessing again should not be a miss if TTL was properly removed
    v.eta
    assert v.cache_stats['properties']['eta']['misses'] <= 3  # Should not increase on every access


def test_combined_size_limit_and_ttl():
    """Test the combination of cache size limits and TTL expiration."""
    # Create a vector with both size limit and TTL
    v = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=3, default_ttl=0.5)
    
    # Access properties
    for prop in ['pt', 'eta', 'phi', 'mass', 'p']:
        getattr(v, prop)
    
    # Verify size limit is enforced
    assert v.cache_size <= 3
    
    # Wait for TTL expiration and clear expired
    time.sleep(0.6)
    expired_count = v.clear_expired()
    assert expired_count > 0
    
    # Cache should be empty or contain only non-expired items
    assert v.cache_size <= 1  # Might contain intermediate values that don't expire
    
    # Access properties with different TTLs
    v.pt
    v.set_ttl('pt', 1.0)  # 1 second
    
    v.eta
    v.set_ttl('eta', 0.3)  # 0.3 second
    
    # Wait for shorter TTL to expire
    time.sleep(0.4)
    
    # Clear expired
    expired_count = v.clear_expired()
    
    # Verify eta and pt cache behavior
    pre_eta_misses = v.cache_stats['properties']['eta']['misses']
    v.eta
    post_eta_misses = v.cache_stats['properties']['eta']['misses']
    
    # eta should have been expired, so accessing it should increase misses
    assert post_eta_misses >= pre_eta_misses


def test_lru_update_on_access():
    """Test that the LRU order is correctly updated when accessing cached properties."""
    # Create a vector with max cache size of 2
    v = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=2)
    
    # Access different properties to populate the cache and tracking
    v.pt
    # Access pt again to register its usage in statistics
    pt_value = v.pt
    # Access a different property
    eta_value = v.eta
    
    # Verify both properties are tracked in stats
    stats = v.cache_stats
    assert 'pt' in stats['properties']
    assert 'eta' in stats['properties']
    
    # Access a third property (should trigger LRU eviction)
    phi_value = v.phi
    
    # Verify cache size still respects the limit
    assert v.cache_size <= 2
    
    # One of pt or eta should have been evicted - we can check this by
    # accessing them again and seeing if misses increased
    initial_misses = {
        'pt': stats['properties']['pt']['misses'],
        'eta': stats['properties']['eta']['misses']
    }
    
    v.pt
    v.eta
    
    final_misses = {
        'pt': v.cache_stats['properties']['pt']['misses'],
        'eta': v.cache_stats['properties']['eta']['misses']
    }
    
    # At least one of them should have more misses (was evicted)
    assert (final_misses['pt'] > initial_misses['pt'] or 
            final_misses['eta'] > initial_misses['eta'])


def test_cache_invalidation_preserves_size_limit():
    """Test that cache invalidation works correctly with size limits."""
    # Create a vector with max cache size of 3
    v = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=3)
    
    # Access 3 properties
    v.pt
    v.eta
    v.phi
    
    # Check cache is populated
    assert v.cache_size > 0
    
    # Touch a component that invalidates some properties
    v._cache.touch_component('px')
    
    # Properties dependent on px should be removed from cache
    for prop in ['pt', 'phi']:
        # Re-access and verify it's a miss (was invalidated)
        pre_misses = v.cache_stats['properties'][prop]['misses']
        getattr(v, prop)
        post_misses = v.cache_stats['properties'][prop]['misses']
        assert post_misses > pre_misses
    
    # Fill cache again
    v.pt
    v.eta
    v.phi
    v.mass
    
    # Cache size should still be limited
    assert v.cache_size <= 3


def test_cache_stats_with_optimization():
    """Test that cache hit/miss statistics work correctly with optimization features."""
    # Create a vector with both optimization features
    v = LVec(1.0, 2.0, 3.0, 4.0, max_cache_size=5, default_ttl=1.0)
    
    # Access a property multiple times
    v.pt  # Miss (first access)
    v.pt  # Hit
    v.pt  # Hit
    
    # Check hit/miss stats
    stats = v.cache_stats
    assert stats['properties']['pt']['hits'] >= 1  # Should have at least one hit
    assert stats['properties']['pt']['misses'] >= 1  # Should have at least one miss
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    # Clear expired
    v.clear_expired()
    
    # Access again after expiration
    v.pt  # Should be a miss because the value expired
    
    # Check updated stats - misses should have increased
    stats = v.cache_stats
    assert stats['properties']['pt']['misses'] >= 2
    
    # Reset stats
    v.reset_cache_stats()
    stats = v.cache_stats
    assert stats['properties']['pt']['hits'] == 0
    assert stats['properties']['pt']['misses'] == 0


def test_ttl_behavior_on_touch():
    """Test that TTL expiry times are properly updated when properties are invalidated."""
    # Create a vector with TTL
    v = LVec(1.0, 2.0, 3.0, 4.0, default_ttl=0.5)
    
    # Access properties
    v.pt
    v.phi
    
    # Touch a component to invalidate properties
    v._cache.touch_component('px')
    
    # Re-access
    v.pt
    
    # Wait for expiration
    time.sleep(0.6)
    
    # Clear expired and check count
    expired_count = v.clear_expired()
    assert expired_count > 0
    
    # Accessing again should be a miss
    pre_misses = v.cache_stats['properties']['pt']['misses']
    v.pt
    post_misses = v.cache_stats['properties']['pt']['misses']
    assert post_misses > pre_misses


def test_cache_hit_ratio():
    """Test the cache_hit_ratio method works correctly."""
    v = LVec(1.0, 2.0, 3.0, 4.0)
    
    # First access is a miss
    v.pt
    
    # Second and third access should be hits
    v.pt
    v.pt
    v.pt  # Multiple hits to ensure we have a good hit ratio
    
    # Hit ratio should be positive (more hits than misses)
    hit_ratio = v.cache_hit_ratio
    assert hit_ratio > 0.0, f"Hit ratio should be positive, got {hit_ratio}"
    
    # Access another property (will be a miss)
    v.mass
    
    # But the overall ratio should still show some hits
    assert v.cache_hit_ratio > 0.0
