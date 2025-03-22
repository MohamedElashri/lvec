# lvec.py
import awkward as ak
from lvec.backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                        backend_sin, backend_cos, backend_atan2,
                        backend_sinh, backend_cosh, backend_where)
from .utils import (ensure_array, check_shapes, compute_p4_from_ptepm,
                   compute_pt, compute_p, compute_mass, compute_eta, compute_phi)
from .exceptions import ShapeError, InputError, BackendError, DependencyError
from .caching import PropertyCache
import numpy as np


class LVec:
    """
    Lorentz Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        px, py, pz: Momentum components
        E: Energy
        _lib: Backend library ('np' or 'ak')
        _cache: Property caching system for optimized property calculations
    """
    
    def __init__(self, px, py, pz, E):
        """
        Initialize Lorentz vector from components.
        
        Args:
            px, py, pz: Momentum components (float, list, ndarray, or ak.Array)
            E: Energy (float, list, ndarray, or ak.Array)
        
        Raises:
            ShapeError: If input arrays have inconsistent shapes
            InputError: If any input has invalid values
            BackendError: If there's an issue with the backend operations
        """
        try:
            self._px, self._py, self._pz, self._E, self._lib = ensure_array(px, py, pz, E)
            check_shapes(self._px, self._py, self._pz, self._E, self._lib)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        # Validate physics constraints
        if isinstance(self._E, (float, int)):
            if self._E < 0:
                raise InputError("E", self._E, "Energy must be non-negative")
        else:
            if self._lib == 'np' and (self._E < 0).any():
                raise InputError("E", "array", "energy values must be non-negative")
            elif self._lib == 'ak' and ak.any(self._E < 0):
                raise InputError("E", "array", "energy values must be non-negative")
                
        # Initialize the enhanced caching system
        self._cache = PropertyCache()
        
        # Register property dependencies
        self._cache.register_dependency('pt', ['px', 'py'])
        self._cache.register_dependency('p', ['px', 'py', 'pz'])
        self._cache.register_dependency('mass', ['px', 'py', 'pz', 'E'])
        self._cache.register_dependency('phi', ['px', 'py'])
        self._cache.register_dependency('eta', ['px', 'py', 'pz'])
        
        # Register intermediate calculation dependencies
        self._cache.register_dependency('px_squared', ['px'])
        self._cache.register_dependency('py_squared', ['py'])
        self._cache.register_dependency('pz_squared', ['pz'])
        self._cache.register_dependency('p_squared', ['px', 'py', 'pz'])
        self._cache.register_dependency('pt_squared', ['px', 'py'])
        
    @classmethod
    def from_p4(cls, px, py, pz, E):
        """Create from Cartesian components."""
        return cls(px, py, pz, E)
    
    @classmethod
    def from_ptepm(cls, pt, eta, phi, m):
        """Create from pt, eta, phi, mass."""
        # First convert to arrays and get the lib type
        pt, eta, phi, m, lib = ensure_array(pt, eta, phi, m)
        px, py, pz, E = compute_p4_from_ptepm(pt, eta, phi, m, lib)
        return cls(px, py, pz, E)
    
    @classmethod
    def from_ary(cls, ary_dict):
        """Create from dictionary with px, py, pz, E keys."""
        return cls(ary_dict["px"], ary_dict["py"], 
                  ary_dict["pz"], ary_dict["E"])
    
    @classmethod
    def from_vec(cls, vobj):
        """Create from another vector-like object with px, py, pz, E attributes."""
        return cls(vobj.px, vobj.py, vobj.pz, vobj.E)
    
    def clear_cache(self):
        """Clear the computed property cache."""
        self._cache.clear_cache()

    def touch(self):
        """
        Mark all components as modified, invalidating all cached properties.
        For backward compatibility with previous API.
        """
        self._cache.touch_component('px')
        self._cache.touch_component('py')
        self._cache.touch_component('pz')
        self._cache.touch_component('E')
            
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
    def px(self): 
        return self._px
    
    @property
    def py(self): 
        return self._py
    
    @property
    def pz(self): 
        return self._pz
    
    @property
    def E(self): 
        return self._E
    
    # Cached intermediate calculations for reuse
    def _px_squared(self):
        return self._get_intermediate('px_squared', lambda: self._px**2)
    
    def _py_squared(self):
        return self._get_intermediate('py_squared', lambda: self._py**2)
    
    def _pz_squared(self):
        return self._get_intermediate('pz_squared', lambda: self._pz**2)
    
    def _pt_squared(self):
        return self._get_intermediate('pt_squared', 
                                    lambda: self._px_squared() + self._py_squared())
    
    def _p_squared(self):
        return self._get_intermediate('p_squared', 
                                    lambda: self._pt_squared() + self._pz_squared())
    
    @property
    def pt(self):
        """Transverse momentum."""
        return self._get_cached('pt', 
                             lambda: backend_sqrt(self._pt_squared(), self._lib),
                             ['px', 'py'])
    
    @property
    def p(self):
        """Total momentum magnitude."""
        return self._get_cached('p', 
                             lambda: backend_sqrt(self._p_squared(), self._lib),
                             ['px', 'py', 'pz'])
    
    @property
    def mass(self):
        """Invariant mass."""
        def calc_mass():
            # E² - p² calculation optimized with intermediate results
            E_squared = self._E**2
            p_squared = self._p_squared()
            m_squared = E_squared - p_squared
            
            # Handle numerical precision issues (for physical particles, m² should be ≥ 0)
            if isinstance(m_squared, (float, int)):
                if m_squared < 0:
                    if abs(m_squared) < 1e-10:  # Small tolerance for numerical precision
                        m_squared = 0
                    # Large negative values are a problem
                    elif m_squared < -1e-6:
                        import warnings
                        warnings.warn(f"Mass calculation resulted in negative m²={m_squared}")
                        m_squared = abs(m_squared)
            else:
                # Array case
                if self._lib == 'np':
                    if (m_squared < 0).any():
                        # Check for numerically significant negative values
                        problematic = (m_squared < -1e-6)
                        if problematic.any():
                            import warnings
                            warnings.warn("Mass calculation resulted in negative m² values")
                        # Take absolute value to ensure physical result
                        m_squared = np.where(m_squared < 0, abs(m_squared), m_squared)
                elif self._lib == 'ak':
                    import awkward as ak
                    if ak.any(m_squared < 0):
                        problematic = (m_squared < -1e-6)
                        if ak.any(problematic):
                            import warnings
                            warnings.warn("Mass calculation resulted in negative m² values")
                        # Take absolute value with the appropriate backend
                        m_squared = ak.where(m_squared < 0, abs(m_squared), m_squared)
            
            return backend_sqrt(m_squared, self._lib)
        
        return self._get_cached('mass', calc_mass, ['px', 'py', 'pz', 'E'])
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                             lambda: backend_atan2(self._py, self._px, self._lib),
                             ['px', 'py'])
    
    @property
    def eta(self):
        """Pseudorapidity."""
        def calc_eta():
            p = self.p  # This will use our cached property
            pz = self._pz
            # Use a small epsilon to avoid division by zero
            epsilon = 1e-10
            
            if self._lib == 'np':
                # For NumPy, handle scalar and array cases
                if isinstance(p, (float, int)) and isinstance(pz, (float, int)):
                    if abs(p - abs(pz)) < epsilon:
                        # When p ≈ |pz|, the particle is along the beam axis
                        return float('inf') if pz >= 0 else float('-inf')
                    # Numerically stable formula for eta
                    return 0.5 * np.log((p + pz) / (p - pz))
                else:
                    # Array case with mask for special values
                    near_beam = (abs(p - abs(pz)) < epsilon)
                    # First compute the standard formula
                    result = 0.5 * np.log((p + pz) / (p - pz + epsilon))
                    # Then correct special cases (if any)
                    if near_beam.any():
                        result = np.where(near_beam & (pz >= 0), np.inf, result)
                        result = np.where(near_beam & (pz < 0), -np.inf, result)
                    return result
            else:
                # For Awkward arrays
                import awkward as ak
                # Define a small epsilon to avoid division by zero
                near_beam = (abs(p - abs(pz)) < epsilon)
                # Compute the standard formula
                result = 0.5 * backend_log((p + pz) / (p - pz + epsilon), self._lib)
                # Correct special cases
                if ak.any(near_beam):
                    result = ak.where(near_beam & (pz >= 0), np.inf, result)
                    result = ak.where(near_beam & (pz < 0), -np.inf, result)
                return result
                
        return self._get_cached('eta', calc_eta, ['px', 'py', 'pz'])
    
    def __add__(self, other):
        """Add two Lorentz vectors."""
        return LVec(self.px + other.px, self.py + other.py,
                   self.pz + other.pz, self.E + other.E)
    
    def __sub__(self, other):
        """Subtract two Lorentz vectors."""
        return LVec(self.px - other.px, self.py - other.py,
                   self.pz - other.pz, self.E - other.E)
    
    def __mul__(self, scalar):
        """Multiply by scalar."""
        return LVec(scalar * self.px, scalar * self.py,
                   scalar * self.pz, scalar * self.E)
    
    __rmul__ = __mul__
    
    def __getitem__(self, idx):
        """Index into vector components."""
        return LVec(self.px[idx], self.py[idx],
                   self.pz[idx], self.E[idx])
    
    def boost(self, bx, by, bz):
        """
        Apply Lorentz boost.
        
        Raises:
            InputError: If boost velocity is >= speed of light
            BackendError: If there's an issue with the backend calculations
            DependencyError: If required backend package is not available
        """
        try:
            b2 = bx*bx + by*by + bz*bz
            if isinstance(b2, (float, int)):
                if b2 >= 1:
                    raise InputError("boost velocity", f"√{b2}", "Must be < 1 (speed of light)")
            else:
                if self._lib == 'np' and (b2 >= 1).any():
                    raise InputError("boost velocity", "array", "All values must be < 1 (speed of light)")
                elif self._lib == 'ak':
                    try:
                        import awkward
                    except ImportError:
                        raise DependencyError("awkward", "pip install awkward")
                    if awkward.any(b2 >= 1):
                        raise InputError("boost velocity", "array", "All values must be < 1 (speed of light)")
            
            # ...existing boost implementation...
            gamma = 1.0 / backend_sqrt(1.0 - b2, self._lib)
            bp = bx*self.px + by*self.py + bz*self.pz
        
            # Modify the condition to work with arrays
            gamma2 = backend_where(b2 > 0, (gamma - 1.0) / b2, 0.0, self._lib)
        
            px = self.px + gamma2*bp*bx + gamma*bx*self.E
            py = self.py + gamma2*bp*by + gamma*by*self.E
            pz = self.pz + gamma2*bp*bz + gamma*bz*self.E
            E = gamma*(self.E + bp)
    
            return LVec(px, py, pz, E)
        except Exception as e:
            if isinstance(e, (InputError, DependencyError)):
                raise
            raise BackendError("boost", self._lib, str(e))
    
    def boostz(self, bz):
        """Apply boost along z-axis."""
        return self.boost(0, 0, bz)
    
    def rotx(self, angle):
        """Rotate around x-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return LVec(self.px,
                   c*self.py - s*self.pz,
                   s*self.py + c*self.pz,
                   self.E)
    
    def roty(self, angle):
        """Rotate around y-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return LVec(c*self.px + s*self.pz,
                   self.py,
                   -s*self.px + c*self.pz,
                   self.E)
    
    def rotz(self, angle):
        """Rotate around z-axis."""
        c, s = backend_cos(angle, self._lib), backend_sin(angle, self._lib)
        return LVec(c*self.px - s*self.py,
                   s*self.px + c*self.py,
                   self.pz,
                   self.E)
    
    def rotate(self, angle, axis='z'):
        """
        Rotate around a specified axis.
        
        Args:
            angle: Rotation angle in radians
            axis: Axis to rotate around ('x', 'y', or 'z')
        
        Returns:
            LVec: New rotated Lorentz vector
            
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
    
    def to_np(self):
        """Convert to NumPy arrays."""
        return {
            'px': to_np(self.px),
            'py': to_np(self.py),
            'pz': to_np(self.pz),
            'E': to_np(self.E)
        }
        
    def to_ak(self):
        """Convert to Awkward arrays."""
        return {
            'px': to_ak(self.px),
            'py': to_ak(self.py),
            'pz': to_ak(self.pz),
            'E': to_ak(self.E)
        }
        
    def to_p4(self):
        """Return components as tuple."""
        return self.px, self.py, self.pz, self.E
    
    def to_ptepm(self):
        """Return pt, eta, phi, mass representation."""
        return self.pt, self.eta, self.phi, self.mass
    
    def to_root_dict(self):
        """Convert to ROOT-compatible dictionary."""
        return {
            'fX': to_np(self.px),
            'fY': to_np(self.py),
            'fZ': to_np(self.pz),
            'fE': to_np(self.E)
        }