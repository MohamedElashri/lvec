import numpy as np
from numba import jit, float64, vectorize
from typing import Tuple

# JIT-optimized core computations
@jit(nopython=True)
def _compute_p4_from_ptepm(pt: float64, eta: float64, phi: float64, m: float64) -> Tuple[float64, float64, float64, float64]:
    """JIT-optimized conversion from pt, eta, phi, mass to px, py, pz, E."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px*px + py*py + pz*pz + m*m)
    return px, py, pz, E

@jit(nopython=True)
def _compute_pt(px: float64, py: float64) -> float64:
    """JIT-optimized transverse momentum calculation."""
    return np.sqrt(px*px + py*py)

@jit(nopython=True)
def _compute_p(px: float64, py: float64, pz: float64) -> float64:
    """JIT-optimized total momentum calculation."""
    return np.sqrt(px*px + py*py + pz*pz)

@jit(nopython=True)
def _compute_mass(E: float64, p: float64) -> float64:
    """JIT-optimized mass calculation."""
    return np.sqrt(np.abs(E*E - p*p))  # abs to handle floating point errors

@jit(nopython=True)
def _compute_eta(p: float64, pz: float64) -> float64:
    """JIT-optimized pseudorapidity calculation."""
    if p == pz:  # Forward direction
        return np.inf
    elif p == -pz:  # Backward direction
        return -np.inf
    return np.arctanh(pz/p)

@jit(nopython=True)
def _compute_phi(px: float64, py: float64) -> float64:
    """JIT-optimized azimuthal angle calculation."""
    return np.arctan2(py, px)

@jit(nopython=True)
def _boost_vector(px: float64, py: float64, pz: float64, E: float64,
                 bx: float64, by: float64, bz: float64) -> Tuple[float64, float64, float64, float64]:
    """JIT-optimized Lorentz boost."""
    b2 = bx*bx + by*by + bz*bz
    if b2 >= 1:
        raise ValueError("Boost velocity must be < 1 (speed of light)")
        
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = bx*px + by*py + bz*pz
    gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0
    
    px_new = px + gamma2*bp*bx + gamma*bx*E
    py_new = py + gamma2*bp*by + gamma*by*E
    pz_new = pz + gamma2*bp*bz + gamma*bz*E
    E_new = gamma*(E + bp)
    
    return px_new, py_new, pz_new, E_new

# Separate vectorized functions for each component
@vectorize([float64(float64, float64, float64, float64)])
def compute_px_from_ptepm(pt, eta, phi, m):
    return pt * np.cos(phi)

@vectorize([float64(float64, float64, float64, float64)])
def compute_py_from_ptepm(pt, eta, phi, m):
    return pt * np.sin(phi)

@vectorize([float64(float64, float64, float64, float64)])
def compute_pz_from_ptepm(pt, eta, phi, m):
    return pt * np.sinh(eta)

@vectorize([float64(float64, float64, float64, float64)])
def compute_E_from_ptepm(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.sqrt(px*px + py*py + pz*pz + m*m)

@vectorize([float64(float64, float64)])
def compute_pt_vec(px, py):
    return _compute_pt(px, py)

@vectorize([float64(float64, float64, float64)])
def compute_p_vec(px, py, pz):
    return _compute_p(px, py, pz)

@vectorize([float64(float64, float64)])
def compute_mass_vec(E, p):
    return _compute_mass(E, p)

@vectorize([float64(float64, float64)])
def compute_eta_vec(p, pz):
    return _compute_eta(p, pz)

@vectorize([float64(float64, float64)])
def compute_phi_vec(px, py):
    return _compute_phi(px, py)

# Separate vectorized functions for boost components
@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def boost_px_vec(px, py, pz, E, bx, by, bz):
    px_new, _, _, _ = _boost_vector(px, py, pz, E, bx, by, bz)
    return px_new

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def boost_py_vec(px, py, pz, E, bx, by, bz):
    _, py_new, _, _ = _boost_vector(px, py, pz, E, bx, by, bz)
    return py_new

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def boost_pz_vec(px, py, pz, E, bx, by, bz):
    _, _, pz_new, _ = _boost_vector(px, py, pz, E, bx, by, bz)
    return pz_new

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def boost_E_vec(px, py, pz, E, bx, by, bz):
    _, _, _, E_new = _boost_vector(px, py, pz, E, bx, by, bz)
    return E_new
