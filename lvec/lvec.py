# lvec.py
from lvec.backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                        backend_sin, backend_cos, backend_atan2,
                        backend_sinh, backend_cosh, backend_where, backend_any, validate_beta, own_components)
from .utils import ensure_array, check_shapes, compute_p4_from_ptepm
from .exceptions import ShapeError, InputError, BackendError, DependencyError
from .caching import PropertyCache
from .frame import Frame
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
    
    def __init__(self, px, py, pz, E, max_cache_size=None, default_ttl=None, *,
                 validate=False, cache_intermediates=False, cache_stats=True):
        """
        Initialize Lorentz vector from components.
        
        Args:
            px, py, pz: Momentum components (float, list, ndarray, or ak.Array)
            E: Energy (float, list, ndarray, or ak.Array)
            max_cache_size: Maximum number of entries in the cache (None for unlimited)
            default_ttl: Default time-to-live for cached values in seconds (None for no expiration)
            validate: Opt in to physical-particle validation (default False)
            cache_intermediates: Retain squared-component arrays for reuse (default False)
            cache_stats: Record cache hit/miss counters (default True)
        
        Raises:
            ShapeError: If input arrays have inconsistent shapes
            InputError: If any input has invalid values
            BackendError: If there's an issue with the backend operations
        """
        self._lib = 'unknown'
        try:
            self._px, self._py, self._pz, self._E, self._lib = ensure_array(*(float(x) if isinstance(x, np.number) else x for x in (px, py, pz, E)))
            check_shapes(self._px, self._py, self._pz, self._E, self._lib)
            self._px, self._py, self._pz, self._E = own_components(
                self._px, self._py, self._pz, self._E)
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", self._lib, str(e))
            
        # Initialize the enhanced caching system with memory optimization features
        self._cache = PropertyCache(max_size=max_cache_size, default_ttl=default_ttl,
                                    cache_intermediates=cache_intermediates,
                                    instrumentation=cache_stats)
        
        # Register property dependencies
        self._cache.register_dependency('pt', ['px', 'py'])
        self._cache.register_dependency('p', ['px', 'py', 'pz'])
        self._cache.register_dependency('mass2', ['px', 'py', 'pz', 'E'])
        self._cache.register_dependency('mass', ['px', 'py', 'pz', 'E'])
        self._cache.register_dependency('phi', ['px', 'py'])
        self._cache.register_dependency('eta', ['px', 'py', 'pz'])
        
        # Only retained intermediates need dependency bookkeeping.
        if cache_intermediates:
            self._cache.register_dependency('px_squared', ['px'])
            self._cache.register_dependency('py_squared', ['py'])
            self._cache.register_dependency('pz_squared', ['pz'])
            self._cache.register_dependency('p_squared', ['px', 'py', 'pz'])
            self._cache.register_dependency('pt_squared', ['px', 'py'])
        
        if validate:
            self.validate_particle()

    def is_physical(self, rtol=1e-12, atol=0.0):
        """Mask finite, future-directed timelike/lightlike particle momenta.

        ``atol`` has units of energy squared. The relative tolerance scales
        with max(E², |p|²); raw ``mass2`` is never clipped.
        """
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("Particle validation tolerances must be finite and non-negative")
        finite = (np.isfinite(self.px) & np.isfinite(self.py)
                  & np.isfinite(self.pz) & np.isfinite(self.E))
        with np.errstate(invalid='ignore', over='ignore'):
            p2 = self._p_squared()
            e2 = self.E**2
            tolerance = atol + rtol * np.maximum(e2, p2)
            return finite & (self.E >= 0) & ((e2 - p2) >= -tolerance)

    def validate_particle(self, rtol=1e-12, atol=0.0):
        """Raise InputError for invalid particle momenta; return self otherwise.

        Mathematical four-vectors may have negative energy or be spacelike.
        Use this opt-in check when the object represents physical particles.
        Missing Awkward values are preserved and excluded from validation.
        """
        if backend_any(~self.is_physical(rtol=rtol, atol=atol)):
            raise InputError("four-momentum", "particle", "Components must be finite, energy must be non-negative, and mass² must be non-negative within tolerance")
        return self

    @classmethod
    def from_p4(cls, px, py, pz, E, max_cache_size=None, default_ttl=None, *,
                 validate=False, cache_intermediates=False, cache_stats=True):
        """Create from Cartesian components."""
        return cls(px, py, pz, E, max_cache_size=max_cache_size, default_ttl=default_ttl,
                   validate=validate, cache_intermediates=cache_intermediates,
                   cache_stats=cache_stats)
    
    @classmethod
    def from_ptepm(cls, pt, eta, phi, m, max_cache_size=None, default_ttl=None, *,
                 validate=False, cache_intermediates=False, cache_stats=True):
        """Create from pt, eta, phi, mass."""
        # First convert to arrays and get the lib type
        pt, eta, phi, m, lib = ensure_array(*(float(x) if isinstance(x, np.number) else x for x in (pt, eta, phi, m)))
        pt, eta, phi, m = own_components(pt, eta, phi, m)
        if validate and backend_any((pt < 0) | (m < 0)):
            raise InputError("pt or mass", "negative", "Particle pt and mass must be non-negative")
        px, py, pz, E = compute_p4_from_ptepm(pt, eta, phi, m, lib)
        return cls(px, py, pz, E, max_cache_size=max_cache_size, default_ttl=default_ttl,
                   validate=validate, cache_intermediates=cache_intermediates,
                   cache_stats=cache_stats)
    
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
            
    def _get_cached(self, key, func, dependencies=None, ttl=None):
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Property or intermediate result name
            func: Function to compute the value if not cached
            dependencies: List of components this value depends on (if not already registered)
            ttl: Time-to-live in seconds for this specific value (overrides default)
            
        Returns:
            The cached or computed value
        """
        return self._cache.get_cached(key, func, dependencies, ttl)
    
    def _get_intermediate(self, key, func, ttl=None):
        """
        Get or compute an intermediate result for reuse across properties.
        
        Args:
            key: Intermediate result identifier
            func: Function to compute the result
            ttl: Time-to-live in seconds for this specific value (overrides default)
            
        Returns:
            The intermediate result
        """
        return self._cache.get_intermediate(key, func, ttl)
        
    def set_ttl(self, property_name, ttl_seconds):
        """
        Set or update the time-to-live for a specific property.
        
        Args:
            property_name: Name of the property to set TTL for (e.g., 'pt', 'mass')
            ttl_seconds: Time-to-live in seconds (None for no expiration)
        """
        self._cache.set_ttl(property_name, ttl_seconds)
        
    def clear_expired(self):
        """
        Remove all expired properties from the cache.
        
        Returns:
            int: Number of expired properties removed
        """
        return self._cache.clear_expired()
    
    # Cache instrumentation properties
    @property
    def cache_stats(self):
        """Get comprehensive statistics about the cache performance."""
        return self._cache.get_stats()
    
    @property
    def cache_hit_ratio(self):
        """Get the overall cache hit ratio as a float between 0 and 1."""
        return self._cache.get_hit_ratio()
    
    @property
    def cache_size(self):
        """Get the current number of items in the cache."""
        return len(self._cache)
    
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
    def mass2(self):
        """Signed invariant mass squared with metric (+, -, -, -)."""
        return self._get_cached('mass2', lambda: self._E**2 - self._p_squared(),
                                ['px', 'py', 'pz', 'E'])

    @property
    def mass(self):
        """Signed invariant mass: sign(m²) sqrt(abs(m²)).

        A negative result denotes a spacelike four-vector. No arbitrary
        dimensionful tolerance or absolute value hides the sign of m².
        """
        def calc_mass():
            m2 = self._E**2 - self._p_squared()
            return np.sign(m2) * np.sqrt(abs(m2))
        return self._get_cached('mass', calc_mass, ['px', 'py', 'pz', 'E'])

    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                             lambda: backend_atan2(self._py, self._px, self._lib),
                             ['px', 'py'])
    
    @property
    def eta(self):
        """Pseudorapidity, asinh(pz/pt).

        Beam-axis vectors have signed infinity. The zero spatial vector has
        eta=0 by convention. Missing Awkward entries remain missing.
        """
        def calc_eta():
            pt = np.hypot(self._px, self._py)
            # Select a safe denominator before division, so zero vectors do
            # not produce runtime warnings on either array backend.
            safe_pt = backend_where(pt == 0, 1.0, pt, self._lib)
            eta = np.arcsinh(self._pz / safe_pt)
            beam_eta = backend_where(self._pz == 0, 0.0,
                                     backend_where(self._pz > 0, np.inf, -np.inf, self._lib),
                                     self._lib)
            return backend_where(pt == 0, beam_eta, eta, self._lib)
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
            # Normalize array-like boost components without changing vector
            # component broadcasting rules.
            bx, by, bz, boost_lib = ensure_array(*(float(x) if isinstance(x, np.number) else x for x in (bx, by, bz)))
            bx, by, bz = own_components(bx, by, bz)
            b2 = validate_beta(bx, by, bz)
            gamma = 1.0 / backend_sqrt(1.0 - b2, boost_lib)
            bp = bx*self.px + by*self.py + bz*self.pz
            # Equivalent to (gamma-1)/beta², stable at beta=0 and tiny beta.
            gamma2 = gamma * gamma / (gamma + 1.0)

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
    
    def to_frame(self, frame):
        """
        Transform this 4-vector to the specified reference frame.

        This is equivalent to performing a boost by the negative
        of the frame's velocity.

        Parameters
        ----------
        frame : Frame
            The target frame to transform into.

        Returns
        -------
        LVec
            The LVec transformed to the target frame.
        """
        return self.boost(-frame.beta_x, -frame.beta_y, -frame.beta_z)

    def transform_frame(self, current_frame, target_frame):
        """
        Transform this vector from current_frame to target_frame.

        This performs a two-step transformation:
        1. Transform from current_frame back to the "universal rest frame"
        2. Transform from the "universal rest frame" to the target_frame

        Parameters
        ----------
        current_frame : Frame
            The frame in which this LVec is currently expressed.
        target_frame : Frame
            The frame to which to transform.

        Returns
        -------
        LVec
            The new LVec in the target frame.
        """
        # Step 1: First boost from current_frame back to the universal rest frame
        # Undo the passive transform by an active boost +beta_current.
        unboosted = self.boost(current_frame.beta_x,
                              current_frame.beta_y,
                              current_frame.beta_z)
        
        # Step 2: Now boost into the target_frame
        # Apply the passive transform with -beta_target.
        final = unboosted.boost(-target_frame.beta_x,
                              -target_frame.beta_y,
                              -target_frame.beta_z)
        
        return final
    
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
