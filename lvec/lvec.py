# lvec.py
from lvec.backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                        backend_atan2,
                        backend_sinh, backend_cosh, backend_where, backend_any, validate_beta, own_components, freeze_array, normalize_scalar)
from .utils import (ensure_array, check_shapes, compute_p4_from_ptepm,
                    scaled_invariant, invariant_result, awkward_invariant, euclidean_result, dot_result)
from .exceptions import ShapeError, InputError, BackendError, DependencyError
from .caching import PropertyCache
from .frame import Frame
from ._adaptive import rotation_pair, rotation_coefficients, boost_components, adaptive_sum
import numpy as np
from ._protocol import PlanarAPI, SpatialAPI, LorentzAPI


class LVec(LorentzAPI):
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
        self._initialize_components(px, py, pz, E)
        self._initialize_cache(max_cache_size, default_ttl,
                               cache_intermediates=cache_intermediates,
                               cache_stats=cache_stats)
        if validate:
            self.validate_particle()

    def _initialize_components(self, px, py, pz, E, *, trusted=False):
        """Normalize components; only internal results may bypass snapshots."""
        self._lib = 'unknown'
        try:
            self._px, self._py, self._pz, self._E, self._lib = ensure_array(*(normalize_scalar(x) for x in (px, py, pz, E)))
            components = (self._px, self._py, self._pz, self._E)
            if trusted and self._lib == 'np':
                # Callers supply fresh arithmetic results or immutable owned
                # components. Reject complex/object results before promotion.
                if any(isinstance(x, np.ndarray) and x.dtype.kind not in 'biuf'
                       for x in components):
                    raise TypeError("Vector components must be real numeric values")
                components = tuple(freeze_array(np.asarray(x, dtype=np.float64))
                                   if isinstance(x, np.ndarray) else float(x)
                                   for x in components)
            else:
                # Awkward ufunc results can reuse index/offset buffers from an
                # external scalar/angle array. Keep a snapshot of those too.
                components = own_components(*components)
            self._px, self._py, self._pz, self._E = components
        except Exception as e:
            if isinstance(e, ShapeError):
                raise
            raise BackendError("initialization", getattr(self, "_lib", "unknown"), str(e))
            
    def _initialize_cache(self, max_cache_size=None, default_ttl=None, *,
                          cache_intermediates=False, cache_stats=True):
        """Initialize lazy derived-property bookkeeping for either factory."""
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
        
    @classmethod
    def _from_owned(cls, px, py, pz, E, max_cache_size=None, default_ttl=None,
                    *, validate=False, cache_intermediates=False, cache_stats=True,
                    _trusted_awkward=False):
        """Construct from fresh results or existing immutable owned buffers.

        Private trusted path: never pass caller-owned mutable arrays here.
        Public construction always takes independent component snapshots.

        _trusted_awkward additionally requires a producer that proves all four
        layouts agree and every buffer is fresh or belongs to an owned input.
        General indexing and array-valued rotation/scaling retain snapshots:
        their result structure may alias an external selector or parameter.
        """
        if cls is not LVec:
            # Preserve custom subclass initialization behavior.
            return cls(px, py, pz, E, max_cache_size=max_cache_size,
                       default_ttl=default_ttl, validate=validate,
                       cache_intermediates=cache_intermediates, cache_stats=cache_stats)
        result = cls.__new__(cls)
        components = (px, py, pz, E)
        if _trusted_awkward and all(is_ak(value) for value in components):
            result._lib = 'ak'
            result._px, result._py, result._pz, result._E = tuple(
                freeze_array(value) for value in components)
        else:
            result._initialize_components(*components, trusted=True)
        result._initialize_cache(max_cache_size, default_ttl,
                                 cache_intermediates=cache_intermediates,
                                 cache_stats=cache_stats)
        if validate:
            result.validate_particle()
        return result

    def is_physical(self, rtol=1e-12, atol=0.0):
        """Mask finite, future-directed timelike/lightlike particle momenta.

        ``atol`` has units of energy squared. The relative tolerance scales
        with max(E², |p|²); raw ``mass2`` is never clipped.
        """
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("Particle validation tolerances must be finite and non-negative")
        finite = (np.isfinite(self.px) & np.isfinite(self.py)
                  & np.isfinite(self.pz) & np.isfinite(self.E))
        if rtol == 0:
            norm = self.mass2
            allowed = norm >= -atol
            if atol == 0:
                # A negative squared invariant may round to negative zero.
                allowed = allowed & ~np.signbit(norm)
            return finite & (self.E >= 0) & allowed
        with np.errstate(invalid='ignore', over='ignore', under='ignore', divide='ignore'):
            scale, norm = scaled_invariant(self.E, self.px, self.py, self.pz, self._lib)
            denominator = backend_where(scale == 0, 1., scale, self._lib)
            e2 = (self.E / denominator)**2
            p2 = ((self.px / denominator)**2 + (self.py / denominator)**2
                  + (self.pz / denominator)**2)
            tolerance = (atol / denominator) / denominator + rtol * np.maximum(e2, p2)
            return finite & (self.E >= 0) & (norm >= -tolerance)

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
        pt, eta, phi, m, lib = ensure_array(*(normalize_scalar(x) for x in (pt, eta, phi, m)))
        pt, eta, phi, m = own_components(pt, eta, phi, m)
        if validate and backend_any((pt < 0) | (m < 0)):
            raise InputError("pt or mass", "negative", "Particle pt and mass must be non-negative")
        px, py, pz, E = compute_p4_from_ptepm(pt, eta, phi, m, lib)
        return cls._from_owned(px, py, pz, E, max_cache_size=max_cache_size, default_ttl=default_ttl,
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
        if all(hasattr(vobj,name) for name in ('px','py','pz','E')):
            return cls(vobj.px, vobj.py, vobj.pz, vobj.E)
        return cls(vobj.x, vobj.y, vobj.z, vobj.t)
    
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
        return self._get_intermediate('px_squared', lambda: np.square(self._px))
    
    def _py_squared(self):
        return self._get_intermediate('py_squared', lambda: np.square(self._py))
    
    def _pz_squared(self):
        return self._get_intermediate('pz_squared', lambda: np.square(self._pz))
    
    def _pt_squared(self):
        return self._get_intermediate('pt_squared', 
                                    lambda: self._px_squared() + self._py_squared())
    
    def _p_squared(self):
        return self._get_intermediate('p_squared', 
                                    lambda: self._pt_squared() + self._pz_squared())
    
    @property
    def pt(self):
        """Transverse momentum."""
        def calculate():
            if is_np(self._px):
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    return euclidean_result(self._pt_squared(), self._px, self._py)
            return np.hypot(self._px, self._py)
        return self._get_cached('pt', calculate, ['px', 'py'])
    
    @property
    def p(self):
        """Total momentum magnitude."""
        def calculate():
            if is_np(self._px):
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    return euclidean_result(self._p_squared(), self._px, self._py, self._pz)
            return np.hypot(np.hypot(self._px, self._py), self._pz)
        return self._get_cached('p', calculate, ['px', 'py', 'pz'])
    
    @property
    def mass2(self):
        """Signed invariant mass squared with metric (+, -, -, -)."""
        def calculate():
            if self._lib == 'ak' and not self._cache._cache_intermediates:
                return awkward_invariant(*self.to_p4(), squared=True)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                m2 = np.square(self._E) - self._p_squared()
                return invariant_result(m2, self.E, self.px, self.py, self.pz,
                                        self._lib, squared=True)
        return self._get_cached('mass2', calculate, ['px', 'py', 'pz', 'E'])

    @property
    def mass(self):
        """Signed invariant mass: sign(m²) sqrt(abs(m²)).

        A negative result denotes a spacelike four-vector. No arbitrary
        dimensionful tolerance or absolute value hides the sign of m².
        """
        def calc_mass():
            if self._lib == 'ak' and not self._cache._cache_intermediates:
                return awkward_invariant(*self.to_p4())
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                m2 = np.square(self._E) - self._p_squared()
                return invariant_result(m2, self.E, self.px, self.py, self.pz, self._lib)
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
            with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
                ratio = self._pz / safe_pt
                eta = np.arcsinh(ratio)
                exceptional = np.isinf(ratio) & np.isfinite(self._pz) & (pt > 0)
                if backend_any(exceptional):
                    fallback = np.sign(self._pz) * (np.log(abs(self._pz))
                                                    - np.log(safe_pt) + np.log(2.))
                    eta = backend_where(exceptional, fallback, eta, self._lib)
            beam_eta = backend_where(self._pz == 0, 0.0,
                                     backend_where(self._pz > 0, np.inf, -np.inf, self._lib),
                                     self._lib)
            return backend_where(pt == 0, beam_eta, eta, self._lib)
        return self._get_cached('eta', calc_eta, ['px', 'py', 'pz'])

    def __add__(self, other):
        """Add two Lorentz vectors."""
        self._check_other(other)
        return LVec._from_owned(self.px + other.px, self.py + other.py,
                   self.pz + other.pz, self.E + other.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec and type(other) is LVec)
    
    def __sub__(self, other):
        """Subtract two Lorentz vectors."""
        self._check_other(other)
        return LVec._from_owned(self.px - other.px, self.py - other.py,
                   self.pz - other.pz, self.E - other.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec and type(other) is LVec)
    
    def __mul__(self, scalar):
        """Multiply by scalar."""
        if isinstance(scalar,PlanarAPI):
            raise TypeError("Use dot or @ for vector products")
        return LVec._from_owned(scalar * self.px, scalar * self.py,
                   scalar * self.pz, scalar * self.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec
                   and type(scalar) in (int, float, bool))
    
    __rmul__ = __mul__
    
    def __getitem__(self, idx):
        """Index into vector components."""
        return LVec._from_owned(self.px[idx], self.py[idx],
                   self.pz[idx], self.E[idx])
    
    def boost(self, bx=None, by=None, bz=None, *, booster=None):
        """
        Apply Lorentz boost.
        
        Raises:
            InputError: If boost velocity is >= speed of light
            BackendError: If there's an issue with the backend calculations
            DependencyError: If required backend package is not available
        """
        if booster is not None:
            if bx is not None or by is not None or bz is not None:
                raise TypeError("Specify a booster or three beta components")
            bx=booster
        if by is None and bz is None and isinstance(bx,SpatialAPI):
            return self.boost_p4(bx) if isinstance(bx,LorentzAPI) else self.boost_beta3(bx)
        if bx is None or by is None or bz is None:
            raise TypeError("Specify a 3D/4D booster or all three beta components")
        try:
            # Normalize array-like boost components without changing vector
            # component broadcasting rules.
            bx, by, bz, boost_lib = ensure_array(*(normalize_scalar(x) for x in (bx, by, bz)))
            bx, by, bz = own_components(bx, by, bz)
            _, delta = validate_beta(bx, by, bz, return_delta=True)
            gamma = 1.0 / backend_sqrt(delta, boost_lib)
            px, py, pz, E = boost_components(self.to_p4(), (bx, by, bz), gamma)

            # Beta components were snapshotted above; result metadata can only
            # refer to those owned buffers, this collection, or fresh outputs.
            return LVec._from_owned(px, py, pz, E,
                       _trusted_awkward=self._lib == 'ak' and type(self) is LVec)
        except Exception as e:
            if isinstance(e, (InputError, DependencyError)):
                raise
            raise BackendError("boost", self._lib, str(e))
    
    def boostz(self, bz):
        """Apply boost along z-axis."""
        return self.boost(0, 0, bz)
    
    def rotx(self, angle):
        """Rotate around x-axis."""
        c, s = rotation_coefficients(angle)
        py, pz = rotation_pair(self.py, self.pz, c, s)
        return LVec._from_owned(self.px, py, pz, self.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec and np.isscalar(angle))

    def roty(self, angle):
        """Rotate around y-axis."""
        c, s = rotation_coefficients(angle)
        px, pz = rotation_pair(self.px, self.pz, c, -s)
        return LVec._from_owned(px, self.py, pz, self.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec and np.isscalar(angle))

    def rotz(self, angle):
        """Rotate around z-axis."""
        c, s = rotation_coefficients(angle)
        px, py = rotation_pair(self.px, self.py, c, s)
        return LVec._from_owned(px, py, self.pz, self.E,
                   _trusted_awkward=self._lib == 'ak' and type(self) is LVec and np.isscalar(angle))

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

    @property
    def pt2(self):
        """Squared transverse momentum, without a square root."""
        def calculate():
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                return self._pt_squared()
        return self._get_cached('pt2', calculate, ['px', 'py'])

    @property
    def p2(self):
        """Squared spatial momentum."""
        def calculate():
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                return self._p_squared()
        return self._get_cached('p2', calculate, ['px', 'py', 'pz'])

    def dot(self, other):
        """Minkowski scalar product with metric (+, -, -, -), in common units."""
        if not isinstance(other, LVec):
            raise TypeError("other must be an LVec")
        if other is self:
            return self.mass2
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            result = (self.E * other.E - self.px * other.px
                      - self.py * other.py - self.pz * other.pz)
            lib = 'ak' if self._lib == 'ak' or other._lib == 'ak' else 'np'
            return dot_result(result, self.to_p4(), other.to_p4(), lib)

    @property
    def rapidity(self):
        """Longitudinal rapidity y = atanh(pz/E).

        Lightlike beam-axis vectors have signed infinite rapidity. The zero
        four-vector and vectors with |pz| > |E| have NaN rapidity. Missing
        Awkward values remain missing.
        """
        def calculate():
            # Subtract logarithms instead of forming their ratio. In particular,
            # E-pz retains precision that would be lost by first forming pz/E.
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                energy = abs(self.E)
                longitudinal = np.sign(self.E) * self.pz
                forward = energy + longitudinal
                backward = energy - longitudinal
                # For overflowed sums, divide each term before adding. The
                # common factor cancels in the logarithmic difference.
                overflow = np.isinf(forward) | np.isinf(backward)
                forward = backend_where(overflow, energy / 2 + longitudinal / 2,
                                        forward, self._lib)
                backward = backend_where(overflow, energy / 2 - longitudinal / 2,
                                         backward, self._lib)
                return .5 * (np.log(forward) - np.log(backward))
        return self._get_cached('rapidity', calculate, ['pz', 'E'])

    @property
    def beta(self):
        """Speed |p|/E in units of c; NaN when E <= 0.

        Spacelike vectors can have beta > 1; this property does not reinterpret
        them as particles. Missing Awkward values remain missing.
        """
        def calculate():
            with np.errstate(divide='ignore', invalid='ignore'):
                momentum = np.hypot(np.hypot(self.px, self.py), self.pz)
                return backend_where(self.E > 0, np.divide(momentum, self.E),
                                     np.nan, self._lib)
        return self._get_cached('beta', calculate, ['px', 'py', 'pz', 'E'])

    @property
    def gamma(self):
        """Lorentz factor E/sqrt(E²-|p|²) for future timelike vectors.

        Positive-energy lightlike vectors have infinite gamma. Spacelike and
        nonpositive-energy vectors have NaN gamma; missing values stay missing.
        """
        def calculate():
            momentum = np.hypot(np.hypot(self.px, self.py), self.pz)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Factoring (E-p)*(E+p)/E**2 avoids overflowing squared
                # components and preserves the representable E-p difference.
                inverse_gamma2 = (np.divide(self.E - momentum, self.E)
                                  * (1 + np.divide(momentum, self.E)))
                factor = np.sqrt(backend_where(inverse_gamma2 >= 0,
                                              inverse_gamma2, np.nan, self._lib))
                return backend_where(self.E > 0, np.divide(1., factor),
                                     np.nan, self._lib)
        return self._get_cached('gamma', calculate, ['px', 'py', 'pz', 'E'])

    def delta_phi(self, other):
        """Signed self.phi - other.phi wrapped to [-pi, pi), in radians."""
        if not isinstance(other, PlanarAPI):
            raise TypeError("other must be a numerical vector")
        return (self.phi - other.phi + np.pi) % (2 * np.pi) - np.pi

    def delta_r2(self, other, *, rapidity=False):
        """Squared angular separation: delta_eta² + delta_phi².

        Set rapidity=True for delta_y² + delta_phi², which is invariant under
        common longitudinal boosts of physical vectors. Beam-axis inf-inf
        differences and other undefined coordinates return NaN.
        """
        if not isinstance(other, SpatialAPI) or (rapidity and not isinstance(other,LorentzAPI)):
            raise TypeError("other must have the required spatial/temporal coordinates")
        with np.errstate(invalid='ignore'):
            longitudinal = ((self.rapidity - other.rapidity) if rapidity
                            else (self.eta - other.eta))
            return longitudinal**2 + self.delta_phi(other)**2

    def delta_r(self, other, *, rapidity=False):
        """Angular separation; see delta_r2 for eta and rapidity conventions."""
        return np.sqrt(self.delta_r2(other, rapidity=rapidity))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, *, initial=None, where=None):
        """Sum four-momentum components along an array axis.

        axis=None sums the whole collection; axis=-1 sums particles within each
        event in a one-level jagged array. Empty collections yield zero vectors.
        Awkward missing particles are skipped; missing events remain missing
        when reducing their particle axis. All components must share masks.
        """
        self._validate_reduction(dtype, out, initial, where)
        components = [adaptive_sum(c, axis=axis, keepdims=keepdims)
                      for c in self.to_p4()]
        if type(self) is LVec and self._lib == 'ak':
            return LVec._from_owned(*components, _trusted_awkward=True)
        return type(self)._from_owned(*components)

    @classmethod
    def from_ak_record(cls, records):
        """Create from an Awkward Array/Record with px, py, pz, E fields."""
        try:
            import awkward as ak
        except ImportError as exc:
            raise DependencyError('awkward') from exc
        if not isinstance(records, (ak.Array, ak.Record)):
            raise TypeError("records must be an Awkward Array or Record")
        required = {'px', 'py', 'pz', 'E'}
        generic={'x','y','z','t'}
        fields=set(ak.fields(records))
        if required.issubset(fields):
            names=('px','py','pz','E')
        elif generic.issubset(fields):
            names=('x','y','z','t')
        else:
            raise InputError('records', ak.fields(records),
                             'Fields px, py, pz, E (or x, y, z, t) are required')
        return cls(*(records[field] for field in names))

    def to_ak_record(self,*,with_behavior=False):
        """Export px, py, pz, E as Awkward records, preserving jagged structure.

        Scalars produce an ak.Record. Array inputs produce an ak.Array with
        missing particles represented by missing records. to_ak() retains its
        existing dictionary return type.
        """
        try:
            import awkward as ak
        except ImportError as exc:
            raise DependencyError('awkward') from exc
        fields = dict(zip(('px', 'py', 'pz', 'E'), self.to_p4()))
        options={}
        if with_behavior:
            from .awkward import behavior
            options={'with_name':'LVec4D','behavior':behavior}
        if self._lib == 'np' and all(np.ndim(c) == 0 for c in self.to_p4()):
            return ak.zip(fields,**options)
        return ak.zip(fields, optiontype_outside_record=True,**options)

    def boost_to_rest(self, reference=None):
        """Boost to the rest frame of reference (self when omitted).

        reference must have finite components, positive energy, and timelike
        four-momentum at every present element. Lightlike/zero/spacelike systems
        have no rest frame and raise InputError. Missing elements stay missing.
        """
        reference = self if reference is None else reference
        if not isinstance(reference, LVec):
            raise TypeError("reference must be an LVec")
        momentum = np.hypot(np.hypot(reference.px, reference.py), reference.pz)
        invalid = reference.E <= momentum
        for component in reference.to_p4():
            invalid = invalid | ~np.isfinite(component)
        if reference._lib == 'ak':
            import awkward as ak
            has_invalid = bool(ak.any(invalid))
        else:
            has_invalid = bool(np.any(invalid))
        if has_invalid:
            raise InputError('reference', 'four-momentum',
                             'A rest frame requires finite, positive-energy timelike momentum')
        return self.boost(-reference.px / reference.E,
                          -reference.py / reference.E,
                          -reference.pz / reference.E)

    def _spatial_cosine(self, other):
        if not isinstance(other, LVec):
            raise TypeError("other must be an LVec")
        # Normalizing separately avoids overflow in |p1|*|p2| and dot products.
        norm1 = np.hypot(np.hypot(self.px, self.py), self.pz)
        norm2 = np.hypot(np.hypot(other.px, other.py), other.pz)
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine = sum(np.divide(a, norm1) * np.divide(b, norm2)
                         for a, b in zip(self.to_p4()[:3], other.to_p4()[:3]))
        return np.minimum(1.0, np.maximum(-1.0, cosine))

    def opening_angle(self, other):
        """Spatial opening angle in radians [0, pi] in the current frame.

        Zero-momentum vectors have undefined direction and yield NaN. Missing
        Awkward values stay missing. This spatial angle is frame dependent.
        """
        return self._spatial_angle(other)

    def _spatial_angle(self, other, *, opposite=False):
        """atan2 keeps collinear angles that would disappear in acos(dot)."""
        if not isinstance(other, LVec):
            raise TypeError("other must be an LVec")
        directions = []
        with np.errstate(divide='ignore', invalid='ignore', under='ignore'):
            for vector in (self, other):
                scale = np.maximum(np.maximum(abs(vector.px), abs(vector.py)), abs(vector.pz))
                xyz = tuple(np.divide(c, scale) for c in vector.to_p4()[:3])
                norm = np.hypot(np.hypot(xyz[0], xyz[1]), xyz[2])
                directions.append(tuple(c / norm for c in xyz))
            a, b = directions
            cross = (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0])
            sine = np.hypot(np.hypot(cross[0], cross[1]), cross[2])
            cosine = sum(x*y for x, y in zip(a, b))
            return np.arctan2(sine, -cosine if opposite else cosine)

    def helicity_angle(self, parent, reference):
        """Daughter helicity angle in radians [0, pi].

        self is the daughter, parent its decaying parent, and reference the
        parent's mother, all expressed in the same frame. In the parent's rest
        frame, measure the daughter's direction relative to the *opposite* of
        the reference momentum. This fixes the otherwise ambiguous sign of the
        helicity axis. A zero axis/daughter direction yields NaN; an invalid
        parent rest frame raises InputError.
        """
        if not isinstance(parent, LVec) or not isinstance(reference, LVec):
            raise TypeError("parent and reference must be LVec objects")
        daughter_rest = self.boost_to_rest(parent)
        reference_rest = reference.boost_to_rest(parent)
        return daughter_rest._spatial_angle(reference_rest, opposite=True)
