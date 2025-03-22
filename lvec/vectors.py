# vectors.py
from lvec.backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                        backend_sin, backend_cos, backend_atan2)
from .utils import (ensure_array, check_shapes, compute_pt, compute_p)
from .exceptions import ShapeError, InputError, BackendError
from .caching import PropertyCache

class Vec2D:
    """2D Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        x, y: Vector components
        _lib: Backend library ('np' or 'ak')
        _cache: Property caching system for optimized property calculations
    """
    
    def __init__(self, x, y):
        try:
            self._x, self._y, self._lib = ensure_array(x, y)
            check_shapes(self._x, self._y, self._lib)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        # Initialize the enhanced caching system
        self._cache = PropertyCache()
        
        # Register property dependencies
        self._cache.register_dependency('r', ['x', 'y'])
        self._cache.register_dependency('phi', ['x', 'y'])
        
        # Register intermediate calculation dependencies
        self._cache.register_dependency('x_squared', ['x'])
        self._cache.register_dependency('y_squared', ['y'])
        
    def clear_cache(self):
        """Clear the computed property cache."""
        self._cache.clear_cache()

    def touch(self):
        """Invalidate cache by marking all components as modified."""
        self._cache.touch_component('x')
        self._cache.touch_component('y')
            
    def _get_cached(self, key, func, dependencies=None):
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Property or intermediate result name
            func: Function to compute the value if not cached
            dependencies: List of components this value depends on (if not already registered)
            
        Returns:
            The cached or computed value
        """
        return self._cache.get_cached(key, func, dependencies)
    
    def _get_intermediate(self, key, func):
        """
        Get or compute an intermediate result for reuse across properties.
        
        Args:
            key: Intermediate result identifier
            func: Function to compute the result
            
        Returns:
            The intermediate result
        """
        return self._cache.get_intermediate(key, func)
    
    # Cache instrumentation properties
    @property
    def cache_stats(self):
        """Get comprehensive statistics about the cache performance."""
        return self._cache.get_stats()
    
    @property
    def cache_hit_ratio(self):
        """Get the overall cache hit ratio as a float between 0 and 1."""
        return self._cache.get_hit_ratio()
    
    def reset_cache_stats(self):
        """Reset all cache hit and miss counters to zero."""
        self._cache.reset_counters()
        
    @property
    def x(self): return self._x
    
    @property
    def y(self): return self._y
    
    # Cached intermediate calculations for reuse
    def _x_squared(self):
        return self._get_intermediate('x_squared', lambda: self._x**2)
    
    def _y_squared(self):
        return self._get_intermediate('y_squared', lambda: self._y**2)
    
    @property
    def r(self):
        """Magnitude of the vector."""
        def calculate_r():
            # Use cached intermediate calculations
            x_squared = self._x_squared()
            y_squared = self._y_squared()
            return backend_sqrt(x_squared + y_squared, self._lib)
            
        return self._get_cached('r', calculate_r, ['x', 'y'])
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: backend_atan2(self._y, self._x, self._lib),
                              ['x', 'y'])
    
    def __add__(self, other):
        """Add two vectors."""
        return Vec2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtract two vectors."""
        return Vec2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Multiply by scalar."""
        return Vec2D(scalar * self.x, scalar * self.y)
    
    __rmul__ = __mul__
    
    def __getitem__(self, idx):
        """Index into vector components."""
        return Vec2D(self.x[idx], self.y[idx])
    
    def dot(self, other):
        """Compute dot product with another vector."""
        return self.x * other.x + self.y * other.y
        
    def rotate(self, angle):
        """
        Rotate the 2D vector by the specified angle.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            Vec2D: New rotated 2D vector
        """
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return Vec2D(c*self.x - s*self.y, s*self.x + c*self.y)

class Vec3D:
    """3D Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        x, y, z: Vector components
        _lib: Backend library ('np' or 'ak')
        _cache: Property caching system for optimized property calculations
    """
    
    def __init__(self, x, y, z):
        try:
            self._x, self._y, self._z, self._lib = ensure_array(x, y, z)
            check_shapes(self._x, self._y, self._z, self._lib)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        # Initialize the enhanced caching system
        self._cache = PropertyCache()
        
        # Register property dependencies
        self._cache.register_dependency('r', ['x', 'y', 'z'])
        self._cache.register_dependency('rho', ['x', 'y'])
        self._cache.register_dependency('phi', ['x', 'y'])
        self._cache.register_dependency('theta', ['x', 'y', 'z'])
        
        # Register intermediate calculation dependencies
        self._cache.register_dependency('x_squared', ['x'])
        self._cache.register_dependency('y_squared', ['y'])
        self._cache.register_dependency('z_squared', ['z'])
        self._cache.register_dependency('xy_squared', ['x', 'y'])  # x² + y²
            
    def clear_cache(self):
        """Clear the computed property cache."""
        self._cache.clear_cache()

    def touch(self):
        """Invalidate cache by marking all components as modified."""
        self._cache.touch_component('x')
        self._cache.touch_component('y')
        self._cache.touch_component('z')
            
    def _get_cached(self, key, func, dependencies=None):
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Property or intermediate result name
            func: Function to compute the value if not cached
            dependencies: List of components this value depends on (if not already registered)
            
        Returns:
            The cached or computed value
        """
        return self._cache.get_cached(key, func, dependencies)
    
    def _get_intermediate(self, key, func):
        """
        Get or compute an intermediate result for reuse across properties.
        
        Args:
            key: Intermediate result identifier
            func: Function to compute the result
            
        Returns:
            The intermediate result
        """
        return self._cache.get_intermediate(key, func)
    
    # Cache instrumentation properties
    @property
    def cache_stats(self):
        """Get comprehensive statistics about the cache performance."""
        return self._cache.get_stats()
    
    @property
    def cache_hit_ratio(self):
        """Get the overall cache hit ratio as a float between 0 and 1."""
        return self._cache.get_hit_ratio()
    
    def reset_cache_stats(self):
        """Reset all cache hit and miss counters to zero."""
        self._cache.reset_counters()
    
    @property
    def x(self): return self._x
    
    @property
    def y(self): return self._y
    
    @property
    def z(self): return self._z
    
    # Cached intermediate calculations for reuse
    def _x_squared(self):
        return self._get_intermediate('x_squared', lambda: self._x**2)
    
    def _y_squared(self):
        return self._get_intermediate('y_squared', lambda: self._y**2)
    
    def _z_squared(self):
        return self._get_intermediate('z_squared', lambda: self._z**2)
    
    def _xy_squared(self):
        """Compute x² + y² for reuse in multiple properties."""
        return self._get_intermediate('xy_squared', 
                                    lambda: self._x_squared() + self._y_squared())
    
    @property
    def r(self):
        """Magnitude of the vector."""
        def calculate_r():
            # Use cached intermediate calculations for better efficiency
            return backend_sqrt(self._xy_squared() + self._z_squared(), self._lib)
            
        return self._get_cached('r', calculate_r, ['x', 'y', 'z'])
    
    @property
    def rho(self):
        """Cylindrical radius."""
        def calculate_rho():
            # Use the shared intermediate calculation
            return backend_sqrt(self._xy_squared(), self._lib)
            
        return self._get_cached('rho', calculate_rho, ['x', 'y'])
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: backend_atan2(self._y, self._x, self._lib),
                              ['x', 'y'])
    
    @property
    def theta(self):
        """Polar angle."""
        def calculate_theta():
            # Use the cached rho property
            return backend_atan2(self.rho, self._z, self._lib)
            
        return self._get_cached('theta', calculate_theta, ['x', 'y', 'z'])
    
    def __add__(self, other):
        """Add two vectors."""
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """Subtract two vectors."""
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        """Multiply by scalar."""
        return Vec3D(scalar * self.x, scalar * self.y, scalar * self.z)
    
    __rmul__ = __mul__
    
    def __getitem__(self, idx):
        """Index into vector components."""
        return Vec3D(self.x[idx], self.y[idx], self.z[idx])
    
    def dot(self, other):
        """Compute dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Compute cross product with another vector."""
        return Vec3D(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)
                    
    def rotx(self, angle):
        """Rotate around x-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return Vec3D(self.x,
                   c*self.y - s*self.z,
                   s*self.y + c*self.z)
    
    def roty(self, angle):
        """Rotate around y-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return Vec3D(c*self.x + s*self.z,
                   self.y,
                   -s*self.x + c*self.z)
    
    def rotz(self, angle):
        """Rotate around z-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return Vec3D(c*self.x - s*self.y,
                   s*self.x + c*self.y,
                   self.z)
    
    def rotate(self, angle, axis='z'):
        """
        Rotate around a specified axis.
        
        Args:
            angle: Rotation angle in radians
            axis: Axis to rotate around ('x', 'y', or 'z')
        
        Returns:
            Vec3D: New rotated 3D vector
            
        Raises:
            ValueError: If an invalid axis is specified
        """
        if axis.lower() == 'x':
            return self.rotx(angle)
        elif axis.lower() == 'y':
            return self.roty(angle)
        elif axis.lower() == 'z':
            return self.rotz(angle)
        else:
            raise ValueError(f"Invalid rotation axis: {axis}. Must be 'x', 'y', or 'z'")