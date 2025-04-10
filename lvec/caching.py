"""
Caching system for LVec library with fine-grained invalidation and dependency tracking.

This module provides an optimized caching system that:
1. Tracks dependencies between properties to enable fine-grained cache invalidation
2. Caches intermediate calculations for reuse across multiple properties
3. Uses a dependency graph for efficient lazy evaluation
4. Provides instrumentation to track cache hit ratios and performance metrics
5. Implements memory optimization with LRU eviction and TTL support
"""

import time
from collections import OrderedDict

class PropertyCache:
    """
    Property caching system with dependency tracking and fine-grained invalidation.
    
    This class manages the caching of computed properties while tracking their dependencies
    to minimize recalculations when vector components change.
    
    Attributes:
        _values: Dictionary storing the cached values of properties and intermediate results
        _dependencies: Dictionary mapping properties to their dependencies
        _dependents: Dictionary mapping components to the properties that depend on them
        _comp_version: Dictionary tracking the version of each component
        _value_version: Dictionary tracking the version of each cached value
        _hits: Dictionary tracking number of cache hits per property
        _misses: Dictionary tracking number of cache misses per property
        _max_size: Maximum number of entries in the cache
        _lru_order: OrderedDict tracking usage order for LRU eviction
        _expiry_times: Dictionary tracking expiration time for cached values
    """
    
    def __init__(self, max_size=None, default_ttl=None):
        """
        Initialize the caching system.
        
        Args:
            max_size: Maximum number of entries in the cache (None for unlimited)
            default_ttl: Default time-to-live for cached values in seconds (None for no expiration)
        """
        # Cache of computed values (both properties and intermediate results)
        self._values = {}
        
        # Dependency graph: maps properties to their dependencies
        self._dependencies = {}
        
        # Reverse dependency graph: maps components to properties that depend on them
        self._dependents = {}
        
        # Version counters for each component
        self._comp_version = {'px': 0, 'py': 0, 'pz': 0, 'E': 0, 'x': 0, 'y': 0, 'z': 0}
        
        # Version counters for each cached value
        self._value_version = {}
        
        # Instrumentation counters for performance analysis
        self._hits = {}
        self._misses = {}
        self._enabled = True
        
        # Memory optimization attributes
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lru_order = OrderedDict()
        self._expiry_times = {}

    def register_dependency(self, prop, dependencies):
        """
        Register dependencies for a property.
        
        Args:
            prop: The property name (e.g., 'pt', 'mass')
            dependencies: List of components this property depends on (e.g., ['px', 'py'])
        """
        self._dependencies[prop] = set(dependencies)
        
        # Update the reverse dependency mapping
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(prop)
        
        # Initialize instrumentation counters for this property
        if prop not in self._hits:
            self._hits[prop] = 0
        if prop not in self._misses:
            self._misses[prop] = 0
    
    def touch_component(self, component):
        """
        Mark a component as modified, incrementing its version and invalidating
        dependent properties.
        
        Args:
            component: The name of the modified component (e.g., 'px', 'py')
        """
        # Only increment version if this is a tracked component
        if component in self._comp_version:
            self._comp_version[component] += 1
            
            # Find all properties that depend on this component and remove them from cache
            affected_props = self.get_affected_properties([component])
            for prop in affected_props:
                if prop in self._values:
                    del self._values[prop]
                    if prop in self._lru_order:
                        del self._lru_order[prop]
                    if prop in self._expiry_times:
                        del self._expiry_times[prop]
                    
            # Also remove intermediate results that depend on this component
            for key in list(self._values.keys()):
                if key.endswith('_squared') and key.startswith(component):
                    del self._values[key]
                    if key in self._lru_order:
                        del self._lru_order[key]
                    if key in self._expiry_times:
                        del self._expiry_times[key]
    
    def _update_lru(self, key):
        """
        Update the LRU tracking for a key.
        
        Args:
            key: The key that was accessed
        """
        if key in self._lru_order:
            del self._lru_order[key]
        self._lru_order[key] = None  # Value doesn't matter, just using OrderedDict for ordering
        
        # Enforce max cache size if needed
        if self._max_size is not None and len(self._values) > self._max_size:
            while len(self._values) > self._max_size:
                # Get the least recently used key (first item in OrderedDict)
                lru_key, _ = next(iter(self._lru_order.items()))
                
                # Remove from cache
                if lru_key in self._values:
                    del self._values[lru_key]
                if lru_key in self._value_version:
                    del self._value_version[lru_key]
                if lru_key in self._expiry_times:
                    del self._expiry_times[lru_key]
                
                # Remove from LRU tracking
                del self._lru_order[lru_key]
    
    def _is_expired(self, key):
        """
        Check if a cached value has expired.
        
        Args:
            key: The key to check for expiration
            
        Returns:
            bool: True if the value has expired, False otherwise
        """
        if key in self._expiry_times:
            return time.time() > self._expiry_times[key]
        return False
    
    def set_ttl(self, key, ttl_seconds):
        """
        Set or update the expiration time for a specific cached value.
        
        Args:
            key: The key to set expiration for
            ttl_seconds: Time-to-live in seconds (None for no expiration)
        """
        if key in self._values:
            if ttl_seconds is None:
                # Remove expiration if it exists
                if key in self._expiry_times:
                    del self._expiry_times[key]
            else:
                # Set expiration time
                self._expiry_times[key] = time.time() + ttl_seconds
    
    def get_cached(self, key, compute_func, dependencies=None, ttl=None):
        """
        Get a cached value if valid, otherwise compute and cache it.
        
        Args:
            key: The key for the cached value
            compute_func: Function to compute the value if not cached
            dependencies: List of components this value depends on (for new dependencies)
            ttl: Time-to-live in seconds for this specific value (overrides default)
            
        Returns:
            The cached or freshly computed value
        """
        # Register new dependencies if provided
        if dependencies and key not in self._dependencies:
            self.register_dependency(key, dependencies)
        
        # Check if we need to recompute the value
        needs_recompute = False
        
        # If the value hasn't been computed yet
        if key not in self._values:
            needs_recompute = True
        # If the value has expired
        elif self._is_expired(key):
            needs_recompute = True
        # If any dependency has changed since the last computation
        elif key in self._dependencies:
            deps = self._dependencies[key]
            value_version = self._value_version.get(key, -1)
            for dep in deps:
                if dep in self._comp_version and self._comp_version[dep] > value_version:
                    needs_recompute = True
                    break
        
        # Recompute if needed
        if needs_recompute:
            # Record cache miss for instrumentation
            if self._enabled and key in self._misses:
                self._misses[key] += 1
                
            self._values[key] = compute_func()
            
            # Update the value's version to the latest component version
            max_version = 0
            if key in self._dependencies:
                for dep in self._dependencies[key]:
                    if dep in self._comp_version:
                        max_version = max(max_version, self._comp_version[dep])
            self._value_version[key] = max_version
            
            # Set expiration time if TTL is provided
            ttl_to_use = ttl if ttl is not None else self._default_ttl
            if ttl_to_use is not None:
                self._expiry_times[key] = time.time() + ttl_to_use
                
            # Update LRU tracking
            self._update_lru(key)
        else:
            # Record cache hit for instrumentation
            if self._enabled and key in self._hits:
                self._hits[key] += 1
            
            # Update LRU tracking (accessed an existing item)
            self._update_lru(key)
        
        return self._values[key]
    
    def get_intermediate(self, key, compute_func, ttl=None):
        """
        Get or compute an intermediate result that can be reused by multiple properties.
        
        Args:
            key: The key for the intermediate result (e.g., 'px_squared')
            compute_func: Function to compute the intermediate result
            ttl: Time-to-live in seconds for this specific value (overrides default)
            
        Returns:
            The intermediate result
        """
        # Initialize instrumentation counters for intermediate results too
        if self._enabled and key not in self._hits:
            self._hits[key] = 0
            self._misses[key] = 0
            
        # Check if the intermediate result is already cached
        is_cached = key in self._values and not self._is_expired(key)
        
        # Intermediate results are handled similarly to properties but aren't 
        # part of the dependency graph
        if not is_cached:
            if self._enabled:
                self._misses[key] += 1
            
            # Compute and cache the value
            self._values[key] = compute_func()
            
            # Set expiration time if TTL is provided
            ttl_to_use = ttl if ttl is not None else self._default_ttl
            if ttl_to_use is not None:
                self._expiry_times[key] = time.time() + ttl_to_use
                
            # Update LRU tracking
            self._update_lru(key)
        else:
            if self._enabled:
                self._hits[key] += 1
                
            # Update LRU tracking (accessed an existing item)
            self._update_lru(key)
                
        return self._values[key]
    
    def clear_cache(self):
        """Clear all cached values but retain the dependency information."""
        self._values.clear()
        self._value_version.clear()
        self._lru_order.clear()
        self._expiry_times.clear()
    
    def clear_property(self, prop):
        """
        Clear a specific property from the cache.
        
        Args:
            prop: The property to clear
        """
        if prop in self._values:
            del self._values[prop]
        if prop in self._value_version:
            del self._value_version[prop]
        if prop in self._lru_order:
            del self._lru_order[prop]
        if prop in self._expiry_times:
            del self._expiry_times[prop]
            
    def clear_expired(self):
        """
        Remove all expired values from the cache.
        
        Returns:
            int: Number of expired items removed
        """
        expired_keys = [key for key in self._expiry_times if time.time() > self._expiry_times[key]]
        
        for key in expired_keys:
            if key in self._values:
                del self._values[key]
            if key in self._value_version:
                del self._value_version[key]
            if key in self._lru_order:
                del self._lru_order[key]
            del self._expiry_times[key]
            
        return len(expired_keys)
            
    def get_affected_properties(self, components):
        """
        Get all properties affected by changes to the specified components.
        
        Args:
            components: List of component names that have changed
            
        Returns:
            Set of property names that need to be recalculated
        """
        affected = set()
        for comp in components:
            if comp in self._dependents:
                affected.update(self._dependents[comp])
        return affected
    
    # ----- Instrumentation methods -----
    
    def enable_instrumentation(self, enabled=True):
        """
        Enable or disable cache hit/miss tracking.
        
        Args:
            enabled: Boolean flag to enable or disable instrumentation
        """
        self._enabled = enabled
    
    def reset_counters(self):
        """Reset all hit and miss counters to zero."""
        for key in self._hits:
            self._hits[key] = 0
        for key in self._misses:
            self._misses[key] = 0
    
    def get_hit_ratio(self, key=None):
        """
        Get the hit ratio for a specific property or overall.
        
        Args:
            key: Property name to get hit ratio for, or None for overall ratio
            
        Returns:
            float: Hit ratio as a value between 0.0 and 1.0, or None if no accesses
        """
        if key is not None:
            # Return hit ratio for a specific property
            if key in self._hits:
                hits = self._hits[key]
                total = hits + self._misses.get(key, 0)
                return hits / total if total > 0 else None
            return None
        else:
            # Return overall hit ratio across all properties
            total_hits = sum(self._hits.values())
            total_misses = sum(self._misses.values())
            total = total_hits + total_misses
            return total_hits / total if total > 0 else None
    
    def get_stats(self):
        """
        Get comprehensive statistics about the cache performance.
        
        Returns:
            dict: Dictionary containing hit/miss statistics for all properties
        """
        stats = {
            "overall": {
                "hits": sum(self._hits.values()),
                "misses": sum(self._misses.values()),
                "total": sum(self._hits.values()) + sum(self._misses.values()),
                "hit_ratio": self.get_hit_ratio(),
                "cache_size": len(self._values),
                "max_size": self._max_size,
                "expired_keys": len([k for k in self._expiry_times if self._is_expired(k)])
            },
            "properties": {}
        }
        
        # Add stats for each property
        for key in set(self._hits.keys()) | set(self._misses.keys()):
            hits = self._hits.get(key, 0)
            misses = self._misses.get(key, 0)
            total = hits + misses
            hit_ratio = hits / total if total > 0 else None
            
            stats["properties"][key] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_ratio": hit_ratio,
                "ttl": (self._expiry_times.get(key, None) - time.time()) if key in self._expiry_times else None
            }
            
        return stats
    
    def __len__(self):
        """Return the number of items in the cache."""
        return len(self._values)
