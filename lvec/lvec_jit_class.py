import numpy as np
from .lvec_jit import (
    compute_px_from_ptepm, compute_py_from_ptepm,
    compute_pz_from_ptepm, compute_E_from_ptepm,
    compute_pt_vec, compute_p_vec, compute_mass_vec,
    compute_eta_vec, compute_phi_vec,
    boost_px_vec, boost_py_vec, boost_pz_vec, boost_E_vec
)
from .exceptions import ShapeError, InputError, BackendError

class LVecJIT:
    """
    JIT-optimized Lorentz Vector class using Numba.
    Only supports NumPy backend for maximum JIT performance.
    """
    
    def __init__(self, px, py, pz, E):
        """
        Initialize JIT-optimized Lorentz vector from components.
        
        Args:
            px, py, pz: Momentum components (float or ndarray)
            E: Energy (float or ndarray)
        """
        # Convert inputs to numpy arrays
        self._px = np.asarray(px, dtype=np.float64)
        self._py = np.asarray(py, dtype=np.float64)
        self._pz = np.asarray(pz, dtype=np.float64)
        self._E = np.asarray(E, dtype=np.float64)
        
        # Check shapes
        shapes = {arr.shape for arr in [self._px, self._py, self._pz, self._E]}
        if len(shapes) > 1:
            raise ShapeError("Inconsistent input shapes")
            
        # Validate physics constraints
        if np.any(self._E < 0):
            raise InputError("E", "array", "All energy values must be non-negative")
            
        self._cache = {}
        self._version = 0
        
    @classmethod
    def from_p4(cls, px, py, pz, E):
        """Create from Cartesian components."""
        return cls(px, py, pz, E)
    
    @classmethod
    def from_ptepm(cls, pt, eta, phi, m):
        """Create from pt, eta, phi, mass."""
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        
        px = compute_px_from_ptepm(pt, eta, phi, m)
        py = compute_py_from_ptepm(pt, eta, phi, m)
        pz = compute_pz_from_ptepm(pt, eta, phi, m)
        E = compute_E_from_ptepm(pt, eta, phi, m)
        return cls(px, py, pz, E)
    
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
                              lambda: compute_pt_vec(self._px, self._py))
    
    @property
    def p(self):
        """Total momentum magnitude."""
        return self._get_cached('p', 
                              lambda: compute_p_vec(self._px, self._py, self._pz))
    
    @property
    def mass(self):
        """Invariant mass."""
        return self._get_cached('mass',
                              lambda: compute_mass_vec(self._E, self.p))
    
    @property
    def phi(self):
        """Azimuthal angle."""
        return self._get_cached('phi',
                              lambda: compute_phi_vec(self._px, self._py))
    
    @property
    def eta(self):
        """Pseudorapidity."""
        return self._get_cached('eta',
                              lambda: compute_eta_vec(self.p, self._pz))
    
    def __add__(self, other):
        """Add two Lorentz vectors."""
        return LVecJIT(self.px + other.px, self.py + other.py,
                      self.pz + other.pz, self.E + other.E)
    
    def __sub__(self, other):
        """Subtract two Lorentz vectors."""
        return LVecJIT(self.px - other.px, self.py - other.py,
                      self.pz - other.pz, self.E - other.E)
    
    def __mul__(self, scalar):
        """Multiply by scalar."""
        return LVecJIT(scalar * self.px, scalar * self.py,
                      scalar * self.pz, scalar * self.E)
    
    __rmul__ = __mul__
    
    def __getitem__(self, idx):
        """Index into vector components."""
        return LVecJIT(self.px[idx], self.py[idx],
                      self.pz[idx], self.E[idx])
    
    def boost(self, bx, by, bz):
        """Apply Lorentz boost with JIT optimization."""
        bx = np.asarray(bx, dtype=np.float64)
        by = np.asarray(by, dtype=np.float64)
        bz = np.asarray(bz, dtype=np.float64)
        
        px_new = boost_px_vec(self.px, self.py, self.pz, self.E, bx, by, bz)
        py_new = boost_py_vec(self.px, self.py, self.pz, self.E, bx, by, bz)
        pz_new = boost_pz_vec(self.px, self.py, self.pz, self.E, bx, by, bz)
        E_new = boost_E_vec(self.px, self.py, self.pz, self.E, bx, by, bz)
        return LVecJIT(px_new, py_new, pz_new, E_new)
