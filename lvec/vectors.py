# vectors.py
from lvec.backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                        backend_sin, backend_cos, backend_atan2)
from .utils import (ensure_array, check_shapes, compute_pt, compute_p)
from .exceptions import ShapeError, InputError, BackendError

class Vec2D:
    """2D Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        x, y: Vector components
        _lib: Backend library ('np' or 'ak')
        _cache: Dictionary storing computed properties
        _version: Cache version counter
    """
    
    def __init__(self, x, y):
        try:
            self._x, self._y, self._lib = ensure_array(x, y)
            check_shapes(self._x, self._y, self._lib)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        self._cache = {}
        self._version = 0
        
    def clear_cache(self):
        """Clear the computed property cache."""
        self._cache.clear()

    def touch(self):
        """Invalidate cache by incrementing version."""
        self._version += 1
        self.clear_cache()
            
    def _get_cached(self, key, func):
        """Get cached value or compute and cache it."""
        entry = self._cache.get(key)
        if entry is None or entry['version'] != self._version:
            val = func()
            self._cache[key] = {'val': val, 'version': self._version}
            return val
        return entry['val']
    
    @property
    def x(self): return self._x
    
    @property
    def y(self): return self._y
    
    @property
    def r(self):
        """Magnitude of the vector."""
        return self._get_cached('r', 
                              lambda: backend_sqrt(self._x**2 + self._y**2, self._lib))
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: backend_atan2(self._y, self._x, self._lib))
    
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

class Vec3D:
    """3D Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        x, y, z: Vector components
        _lib: Backend library ('np' or 'ak')
        _cache: Dictionary storing computed properties
        _version: Cache version counter
    """
    
    def __init__(self, x, y, z):
        try:
            self._x, self._y, self._z, self._lib = ensure_array(x, y, z)
            check_shapes(self._x, self._y, self._z, self._lib)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        self._cache = {}
        self._version = 0
        
    def clear_cache(self):
        """Clear the computed property cache."""
        self._cache.clear()

    def touch(self):
        """Invalidate cache by incrementing version."""
        self._version += 1
        self.clear_cache()
            
    def _get_cached(self, key, func):
        """Get cached value or compute and cache it."""
        entry = self._cache.get(key)
        if entry is None or entry['version'] != self._version:
            val = func()
            self._cache[key] = {'val': val, 'version': self._version}
            return val
        return entry['val']
    
    @property
    def x(self): return self._x
    
    @property
    def y(self): return self._y
    
    @property
    def z(self): return self._z
    
    @property
    def r(self):
        """Magnitude of the vector."""
        return self._get_cached('r', 
                              lambda: backend_sqrt(self._x**2 + self._y**2 + self._z**2, self._lib))
    
    @property
    def rho(self):
        """Cylindrical radius."""
        return self._get_cached('rho',
                              lambda: backend_sqrt(self._x**2 + self._y**2, self._lib))
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: backend_atan2(self._y, self._x, self._lib))
    
    @property
    def theta(self):
        """Polar angle."""
        return self._get_cached('theta',
                              lambda: backend_atan2(self.rho, self._z, self._lib))
    
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