# lvec.py
from lvec.backends import (is_ak, is_np, to_ak, to_np, 
                      backend_sin, backend_cos, backend_sqrt, backend_log, 
                      backend_atan2, backend_sinh, backend_cosh)
from .utils import (ensure_array, check_shapes, compute_p4_from_ptepm,
                   compute_pt, compute_p, compute_mass, compute_eta, compute_phi)
from .exceptions import ShapeError

class LVec:
    """
    Lorentz Vector class supporting both NumPy and Awkward array backends.
    
    Attributes:
        px, py, pz: Momentum components
        E: Energy
        _lib: Backend library ('np' or 'ak')
        _cache: Dictionary storing computed properties
        _version: Cache version counter
    """
    
    def __init__(self, px, py, pz, E):
        """
        Initialize Lorentz vector from components.
        
        Args:
            px, py, pz: Momentum components (float, list, ndarray, or ak.Array)
            E: Energy (float, list, ndarray, or ak.Array)
        """
        self._px, self._py, self._pz, self._E, self._lib = ensure_array(px, py, pz, E)
        check_shapes(self._px, self._py, self._pz, self._E, self._lib)  # _lib is passed as the last argument
        self._cache = {}
        self._version = 0
        
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
        self._cache.clear()

    def touch(self):
        """Invalidate cache by incrementing version."""
        self._version += 1
        self.clear_cache()  # Just call it, don't add parentheses
            
    def _get_cached(self, key, func):
        """Get cached value or compute and cache it."""
        entry = self._cache.get(key)
        if entry is None or entry['version'] != self._version:
            val = func()
            self._cache[key] = {'val': val, 'version': self._version}
            return val
        return entry['val']
    
    @property
    def px(self): return self._px
    
    @property
    def py(self): return self._py
    
    @property
    def pz(self): return self._pz
    
    @property
    def E(self): return self._E
    
    @property
    def pt(self):
        """Transverse momentum."""
        return self._get_cached('pt', 
                              lambda: compute_pt(self._px, self._py, self._lib))
    
    @property
    def p(self):
        """Total momentum magnitude."""
        return self._get_cached('p', 
                              lambda: compute_p(self._px, self._py, self._pz, self._lib))
    
    @property
    def mass(self):
        """Invariant mass."""
        return self._get_cached('mass',
                              lambda: compute_mass(self._E, self.p, self._lib))
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: compute_phi(self._px, self._py, self._lib))
    
    @property
    def eta(self):
        """Pseudorapidity."""
        return self._get_cached('eta',
                              lambda: compute_eta(self.p, self._pz, self._lib))
    
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
        """Apply Lorentz boost."""
        b2 = bx*bx + by*by + bz*bz
        gamma = 1.0 / backend_sqrt(1.0 - b2, self._lib)
        bp = bx*self.px + by*self.py + bz*self.pz
        gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0
        
        px = self.px + gamma2*bp*bx + gamma*bx*self.E
        py = self.py + gamma2*bp*by + gamma*by*self.E
        pz = self.pz + gamma2*bp*bz + gamma*bz*self.E
        E = gamma*(self.E + bp)
        
        return LVec(px, py, pz, E)
    
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
        """Return dictionary compatible with ROOT."""
        return {
            'fX': to_np(self.px),
            'fY': to_np(self.py),
            'fZ': to_np(self.pz),
            'fE': to_np(self.E)
        }