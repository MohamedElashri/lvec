import numpy as np
import awkward as ak
from typing import Tuple, Union, Optional
from functools import lru_cache

def _get_backend(arr):
    """Determine the backend of the input array."""
    if isinstance(arr, ak.Array):
        return ak
    return np

def _to_array(backend, arr):
    """Convert input to appropriate array type."""
    if backend is ak:
        return ak.Array(arr) if not isinstance(arr, ak.Array) else arr
    return np.asarray(arr)

def _compute_p4_from_ptepm_opt(pt, eta, phi, m):
    """Optimized conversion from pt, eta, phi, mass to px, py, pz, E."""
    backend = _get_backend(pt)
    
    # Pre-compute trig functions once - works for both backends
    cos_phi = np.cos(phi)  # Use numpy for math functions
    sin_phi = np.sin(phi)
    sinh_eta = np.sinh(eta)
    
    # Compute components using standard operators (backend-agnostic)
    px = pt * cos_phi
    py = pt * sin_phi
    pz = pt * sinh_eta
    
    # Efficient p^2 calculation
    p_squared = px * px + py * py + pz * pz
    E = np.sqrt(p_squared + m * m)  # Use numpy for math functions
    
    return px, py, pz, E

def _compute_pt_opt(px, py):
    """Optimized transverse momentum calculation."""
    return np.sqrt(px * px + py * py)  # Use numpy for math functions

def _compute_p_opt(px, py, pz):
    """Optimized total momentum calculation."""
    return np.sqrt(px * px + py * py + pz * pz)  # Use numpy for math functions

def _compute_mass_opt(E, p):
    """Optimized mass calculation with improved numerical stability."""
    backend = _get_backend(E)
    m_squared = E * E - p * p
    # Handle numerical instabilities near zero in a backend-agnostic way
    return np.sqrt(backend.where(m_squared < 0, 0, m_squared))

def _compute_eta_opt(p, pz):
    """Optimized pseudorapidity calculation."""
    backend = _get_backend(p)
    
    # Handle special cases in a backend-agnostic way
    return backend.where(
        p == pz,
        float('inf'),
        backend.where(
            p == -pz,
            float('-inf'),
            np.arctanh(pz/p)  # Use numpy for math functions
        )
    )

def _compute_phi_opt(px, py):
    """Optimized azimuthal angle calculation."""
    return np.arctan2(py, px)  # Use numpy for math functions

def _boost_vector_opt(px, py, pz, E, bx, by, bz):
    """Optimized Lorentz boost with backend-agnostic operations."""
    backend = _get_backend(px)
    
    # Compute b^2 and validate
    b2 = bx * bx + by * by + bz * bz
    if backend.any(b2 >= 1):
        raise ValueError("Boost velocity must be < 1 (speed of light)")
    
    # Pre-compute gamma factors
    gamma = 1.0 / np.sqrt(1.0 - b2)
    gamma2 = backend.where(b2 > 0, (gamma - 1.0) / b2, 0.0)
    
    # Compute dot product
    bp = bx * px + by * py + bz * pz
    
    # Compute boosted components
    px_new = px + gamma2 * bp * bx + gamma * bx * E
    py_new = py + gamma2 * bp * by + gamma * by * E
    pz_new = pz + gamma2 * bp * bz + gamma * bz * E
    E_new = gamma * (E + bp)
    
    return px_new, py_new, pz_new, E_new

class LVecOpt:
    """Backend-agnostic optimized Lorentz Vector implementation."""
    def __init__(self, px, py, pz, E):
        # Determine backend from input
        self._backend = _get_backend(px)
        
        # Convert inputs to appropriate array type if needed
        self._px = _to_array(self._backend, px)
        self._py = _to_array(self._backend, py)
        self._pz = _to_array(self._backend, pz)
        self._E = _to_array(self._backend, E)
        self._cache = {}

    @classmethod
    def from_ptepm(cls, pt, eta, phi, m):
        """Create from pt, eta, phi, mass."""
        px, py, pz, E = _compute_p4_from_ptepm_opt(pt, eta, phi, m)
        return cls(px, py, pz, E)

    def _invalidate_cache(self):
        """Clear cached properties."""
        for attr in list(self.__dict__.keys()):
            if not attr.startswith('_'):
                delattr(self, attr)

    @property
    def pt(self):
        """Transverse momentum."""
        return _compute_pt_opt(self._px, self._py)

    @property
    def p(self):
        """Total momentum."""
        return _compute_p_opt(self._px, self._py, self._pz)

    @property
    def mass(self):
        """Invariant mass."""
        return _compute_mass_opt(self._E, self.p)

    @property
    def eta(self):
        """Pseudorapidity."""
        return _compute_eta_opt(self.p, self._pz)

    @property
    def phi(self):
        """Azimuthal angle."""
        return _compute_phi_opt(self._px, self._py)

    def boost(self, bx, by, bz):
        """Apply Lorentz boost."""
        px_new, py_new, pz_new, E_new = _boost_vector_opt(
            self._px, self._py, self._pz, self._E,
            _to_array(self._backend, bx),
            _to_array(self._backend, by),
            _to_array(self._backend, bz)
        )
        return LVecOpt(px_new, py_new, pz_new, E_new)
