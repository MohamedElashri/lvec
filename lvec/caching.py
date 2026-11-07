"""Property caching with explicit invalidation, optional LRU, and persistent TTLs."""

from collections import OrderedDict
import math
from numbers import Integral, Real
import time

from .backends import is_ak, is_np


_MISSING = object()


def _freeze_value(value):
    """Protect computed numeric arrays from mutation through property results."""
    if is_np(value):
        value.flags.writeable = False
    elif is_ak(value):
        import awkward as ak
        _, _, buffers = ak.to_buffers(value)
        for buffer in buffers.values():
            if is_np(buffer):
                buffer.flags.writeable = False
    return value


class PropertyCache:
    """Cache final properties, retaining large intermediates only when requested.

    ``max_size=0`` bypasses storage. LRU bookkeeping is only allocated for a
    bounded cache. Component changes must go through ``touch_component``.
    TTLs use a monotonic clock and property-specific settings survive eviction,
    invalidation, expiration, and ``clear_cache``.
    """

    def __init__(self, max_size=None, default_ttl=None, *,
                 cache_intermediates=False, instrumentation=True):
        if max_size is not None and (
            isinstance(max_size, bool) or not isinstance(max_size, Integral)
            or max_size < 0
        ):
            raise ValueError("max_size must be a non-negative integer or None")
        self._validate_ttl(default_ttl)
        self._values = {}
        self._dependencies = {}
        self._dependents = {}
        self._hits = {}
        self._misses = {}
        self._enabled = bool(instrumentation)
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache_intermediates = bool(cache_intermediates)
        self._lru_order = OrderedDict() if max_size is not None else None
        self._expiry_times = {}
        self._ttl_overrides = {}

    @staticmethod
    def _validate_ttl(ttl):
        if ttl is not None and (
            isinstance(ttl, bool) or not isinstance(ttl, Real)
            or not math.isfinite(ttl) or ttl < 0
        ):
            raise ValueError("TTL must be a finite non-negative number or None")

    def register_dependency(self, prop, dependencies):
        """Register the components or properties that determine a value."""
        for dep in self._dependencies.get(prop, ()):
            self._dependents[dep].discard(prop)
        self._dependencies[prop] = frozenset(dependencies)
        for dep in self._dependencies[prop]:
            self._dependents.setdefault(dep, set()).add(prop)

    def touch_component(self, component):
        """Invalidate every value depending directly or indirectly on a component."""
        for prop in self.get_affected_properties([component]):
            self.clear_property(prop)

    def _update_lru(self, key):
        if self._lru_order is None:
            return
        self._lru_order[key] = None
        self._lru_order.move_to_end(key)
        while len(self._values) > self._max_size:
            oldest, _ = self._lru_order.popitem(last=False)
            self._values.pop(oldest, None)
            self._expiry_times.pop(oldest, None)

    def _is_expired(self, key):
        expiry = self._expiry_times.get(key)
        return expiry is not None and time.monotonic() >= expiry

    def _variants(self, prop):
        # Values for a non-default invariant policy are stored under 'name#policy'.
        return [key for key in self._values if key == prop or key.startswith(prop + '#')]

    def set_ttl(self, key, ttl_seconds):
        """Set a persistent TTL, including before first access; None disables it."""
        self._validate_ttl(ttl_seconds)
        self._ttl_overrides[key] = ttl_seconds
        for variant in self._variants(key) or [key]:
            if variant in self._values and ttl_seconds is not None:
                self._expiry_times[variant] = time.monotonic() + ttl_seconds
            else:
                self._expiry_times.pop(variant, None)

    def get_cached(self, key, compute_func, dependencies=None, ttl=None):
        """Return a valid cached value, or compute and optionally store it."""
        if dependencies and key not in self._dependencies:
            self.register_dependency(key, dependencies)
        value = self._values.get(key, _MISSING)
        if value is not _MISSING and (not self._expiry_times or not self._is_expired(key)):
            if self._enabled:
                self._hits[key] = self._hits.get(key, 0) + 1
            if self._lru_order is not None:
                self._update_lru(key)
            return value

        if self._enabled:
            self._misses[key] = self._misses.get(key, 0) + 1
        # Keep a local reference: nested computations can evict each other from
        # a small cache, and max_size=0 intentionally never stores a value.
        value = _freeze_value(compute_func())
        if self._max_size == 0:
            return value
        duration = self._ttl_overrides.get(
            key.split('#', 1)[0], ttl if ttl is not None else self._default_ttl
        )
        self._validate_ttl(duration)
        self._values[key] = value
        if duration is None:
            self._expiry_times.pop(key, None)
        else:
            self._expiry_times[key] = time.monotonic() + duration
        if self._lru_order is not None:
            self._update_lru(key)
        return value

    def get_intermediate(self, key, compute_func, ttl=None):
        """Compute an intermediate without retaining its array by default."""
        if not self._cache_intermediates:
            return compute_func()
        return self.get_cached(key, compute_func, ttl=ttl)

    def clear_cache(self):
        """Drop values, preserving dependency information, TTL policy, and stats."""
        self._values.clear()
        if self._lru_order is not None:
            self._lru_order.clear()
        self._expiry_times.clear()

    def clear_property(self, prop):
        """Drop one cached value without changing its TTL policy or statistics."""
        for key in self._variants(prop) or [prop]:
            self._values.pop(key, None)
            if self._lru_order is not None:
                self._lru_order.pop(key, None)
            self._expiry_times.pop(key, None)

    def clear_expired(self):
        """Remove expired values and return their count."""
        now = time.monotonic()
        expired = [key for key, expiry in self._expiry_times.items() if now >= expiry]
        for key in expired:
            self.clear_property(key)
        return len(expired)

    def get_affected_properties(self, components):
        """Find all direct and transitive dependents, safely handling cycles."""
        pending = list(components)
        visited = set(pending)
        affected = set()
        while pending:
            for prop in self._dependents.get(pending.pop(), ()):
                affected.add(prop)
                if prop not in visited:
                    visited.add(prop)
                    pending.append(prop)
        return affected

    def enable_instrumentation(self, enabled=True):
        """Enable or disable counters without changing cached values."""
        self._enabled = bool(enabled)

    def reset_counters(self):
        """Reset counters while preserving known property names."""
        self._hits = dict.fromkeys(self._hits, 0)
        self._misses = dict.fromkeys(self._misses, 0)

    def get_hit_ratio(self, key=None):
        """Return the hit fraction, or None if no accesses were recorded."""
        hits = self._hits.get(key, 0) if key is not None else sum(self._hits.values())
        misses = self._misses.get(key, 0) if key is not None else sum(self._misses.values())
        return hits / (hits + misses) if hits + misses else None

    def get_stats(self):
        """Return a snapshot; retained bytes count cache values, not input arrays."""
        now = time.monotonic()
        hits, misses = sum(self._hits.values()), sum(self._misses.values())
        return {
            "overall": {
                "hits": hits, "misses": misses, "total": hits + misses,
                "hit_ratio": self.get_hit_ratio(), "cache_size": len(self._values),
                "max_size": self._max_size,
                "expired_keys": sum(now >= expiry for expiry in self._expiry_times.values()),
                "retained_bytes": sum(getattr(value, 'nbytes', 0) for value in self._values.values()),
            },
            "properties": {
                key: {
                    "hits": self._hits.get(key, 0), "misses": self._misses.get(key, 0),
                    "total": self._hits.get(key, 0) + self._misses.get(key, 0),
                    "hit_ratio": self.get_hit_ratio(key),
                    "ttl": self._expiry_times[key] - now if key in self._expiry_times else None,
                }
                for key in self._hits.keys() | self._misses.keys()
            },
        }

    def __len__(self):
        return len(self._values)
