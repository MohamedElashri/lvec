"""
Caching system for LVec library with fine-grained invalidation and dependency tracking.

This module provides an optimized caching system that:
1. Tracks dependencies between properties to enable fine-grained cache invalidation
2. Caches intermediate calculations for reuse across multiple properties
3. Uses a dependency graph for efficient lazy evaluation
4. Provides instrumentation to track cache hit ratios and performance metrics
"""

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
    """
    
    def __init__(self):
        """Initialize the caching system."""
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
                    
            # Also remove intermediate results that depend on this component
            for key in list(self._values.keys()):
                if key.endswith('_squared') and key.startswith(component):
                    del self._values[key]
    
    def get_cached(self, key, compute_func, dependencies=None):
        """
        Get a cached value if valid, otherwise compute and cache it.
        
        Args:
            key: The key for the cached value
            compute_func: Function to compute the value if not cached
            dependencies: List of components this value depends on (for new dependencies)
            
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
        else:
            # Record cache hit for instrumentation
            if self._enabled and key in self._hits:
                self._hits[key] += 1
        
        return self._values[key]
    
    def get_intermediate(self, key, compute_func):
        """
        Get or compute an intermediate result that can be reused by multiple properties.
        
        Args:
            key: The key for the intermediate result (e.g., 'px_squared')
            compute_func: Function to compute the intermediate result
            
        Returns:
            The intermediate result
        """
        # Initialize instrumentation counters for intermediate results too
        if self._enabled and key not in self._hits:
            self._hits[key] = 0
            self._misses[key] = 0
            
        # Check if the intermediate result is already cached
        is_cached = key in self._values
        
        # Intermediate results are handled similarly to properties but aren't 
        # part of the dependency graph
        if not is_cached:
            if self._enabled:
                self._misses[key] += 1
            self._values[key] = compute_func()
        else:
            if self._enabled:
                self._hits[key] += 1
                
        return self._values[key]
    
    def clear_cache(self):
        """Clear all cached values but retain the dependency information."""
        self._values.clear()
        self._value_version.clear()
    
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
                "hit_ratio": self.get_hit_ratio()
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
                "hit_ratio": hit_ratio
            }
            
        return stats
