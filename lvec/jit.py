"""
JIT compilation support for performance-critical operations in LVec.

This module provides Numba-accelerated versions of core computation functions
while maintaining compatibility with LVec's caching system. Functions gracefully 
fall back to non-JIT versions when Numba is not available.
"""

import numpy as np
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Configuration
JIT_ENABLED = HAS_NUMBA  # Global flag to enable/disable JIT
OPTIMIZE_NUMBA = True    # Apply additional optimizations specific to Numba

# Function to check if JIT is enabled and available
def is_jit_available():
    """Check if JIT compilation is available and enabled."""
    return HAS_NUMBA and JIT_ENABLED

def enable_jit(enable=True):
    """Enable or disable JIT compilation globally."""
    global JIT_ENABLED
    if enable and not HAS_NUMBA:
        import warnings
        warnings.warn("Numba not installed, JIT compilation not available")
        return False
    JIT_ENABLED = enable
    return JIT_ENABLED

# ------------------- JIT implementations of core computation functions -------------------

if HAS_NUMBA:
    # Numba-optimized functions for NumPy arrays
    
    @nb.njit(cache=True)
    def _jit_compute_pt_np(px, py):
        """JIT-optimized pt calculation for NumPy arrays."""
        return np.sqrt(px*px + py*py)
        
    @nb.njit(cache=True)
    def _jit_compute_p_np(px, py, pz):
        """JIT-optimized momentum calculation for NumPy arrays."""
        return np.sqrt(px*px + py*py + pz*pz)
        
    @nb.njit(cache=True)
    def _jit_compute_mass_np(E, px, py, pz):
        """JIT-optimized mass calculation for NumPy arrays."""
        p_squared = px*px + py*py + pz*pz
        m_squared = E*E - p_squared
        
        # Handle numerical precision issues that can lead to small negative mass-squared values
        # For physical particles, m_squared should be positive
        abs_m_squared = np.abs(m_squared)
        return np.sqrt(abs_m_squared)
    
    @nb.njit(cache=True)
    def _jit_compute_eta_np(p, pz):
        """JIT-optimized eta calculation for NumPy arrays."""
        epsilon = 1e-10  # Small constant to avoid division by zero
        return 0.5 * np.log((p + pz) / (p - pz + epsilon))
    
    @nb.njit(cache=True)
    def _jit_compute_phi_np(px, py):
        """JIT-optimized phi calculation for NumPy arrays."""
        return np.arctan2(py, px)
    
    @nb.njit(cache=True)
    def _jit_compute_p4_from_ptepm_np(pt, eta, phi, m):
        """JIT-optimized conversion from pt,eta,phi,m to p4 for NumPy arrays."""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        p = np.sqrt(px*px + py*py + pz*pz)
        E = np.sqrt(p*p + m*m)
        return px, py, pz, E

# ------------------- JIT-aware wrapper functions -------------------

# Helper for the non-JIT implementations
def _get_original_compute_pt():
    """Get the original compute_pt function without JIT."""
    from lvec.utils import _compute_pt_original
    return _compute_pt_original

def _get_original_compute_p():
    """Get the original compute_p function without JIT."""
    from lvec.utils import _compute_p_original
    return _compute_p_original

def _get_original_compute_mass():
    """Get the original compute_mass function without JIT."""
    from lvec.utils import _compute_mass_original
    return _compute_mass_original

def _get_original_compute_eta():
    """Get the original compute_eta function without JIT."""
    from lvec.utils import _compute_eta_original
    return _compute_eta_original

def _get_original_compute_phi():
    """Get the original compute_phi function without JIT."""
    from lvec.utils import _compute_phi_original
    return _compute_phi_original

def _get_original_compute_p4_from_ptepm():
    """Get the original compute_p4_from_ptepm function without JIT."""
    from lvec.utils import _compute_p4_from_ptepm_original
    return _compute_p4_from_ptepm_original

def jit_compute_pt(px, py, lib):
    """
    JIT-optimized pt calculation with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    px, py : scalar or array-like
        X and Y components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Transverse momentum with the same type as input
    """
    # Only use JIT for NumPy arrays, fall back for scalars and awkward arrays
    if lib == 'np' and is_jit_available() and not isinstance(px, (float, int)):
        result = _jit_compute_pt_np(px, py)
        # Convert to scalar if both inputs were scalars (shouldn't happen but just in case)
        if isinstance(px, (float, int)) and isinstance(py, (float, int)):
            return float(result)
        return result
    else:
        # Let the original function handle awkward arrays and scalars
        return _get_original_compute_pt()(px, py, lib)

def jit_compute_p(px, py, pz, lib):
    """
    JIT-optimized momentum calculation with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    px, py, pz : scalar or array-like
        X, Y, and Z components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Total momentum with the same type as input
    """
    if lib == 'np' and is_jit_available() and not isinstance(px, (float, int)):
        result = _jit_compute_p_np(px, py, pz)
        # Convert to scalar if all inputs were scalars
        if isinstance(px, (float, int)) and isinstance(py, (float, int)) and isinstance(pz, (float, int)):
            return float(result)
        return result
    else:
        # Let the original function handle awkward arrays and scalars
        return _get_original_compute_p()(px, py, pz, lib)

def jit_compute_mass(E, px, py, pz, lib):
    """
    JIT-optimized mass calculation with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    E : scalar or array-like
        Energy
    px, py, pz : scalar or array-like
        Momentum components
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Mass with the same type as input
    """
    if lib == 'np' and is_jit_available() and not isinstance(E, (float, int)):
        result = _jit_compute_mass_np(E, px, py, pz)
        # Convert to scalar if all inputs were scalars
        if isinstance(E, (float, int)) and isinstance(px, (float, int)) and isinstance(py, (float, int)) and isinstance(pz, (float, int)):
            return float(result)
        return result
    else:
        # For awkward arrays, we need a different approach since we calculate
        # mass differently in the original function
        from lvec.utils import compute_p
        p = compute_p(px, py, pz, lib)
        return _get_original_compute_mass()(E, p, lib)

def jit_compute_eta(p, pz, lib):
    """
    JIT-optimized eta calculation with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    p : scalar or array-like
        Total momentum
    pz : scalar or array-like
        Z component of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Pseudorapidity with the same type as input
    """
    if lib == 'np' and is_jit_available() and not isinstance(p, (float, int)):
        result = _jit_compute_eta_np(p, pz)
        # Convert to scalar if both inputs were scalars
        if isinstance(p, (float, int)) and isinstance(pz, (float, int)):
            return float(result)
        return result
    else:
        # Let the original function handle awkward arrays and scalars
        return _get_original_compute_eta()(p, pz, lib)

def jit_compute_phi(px, py, lib):
    """
    JIT-optimized phi calculation with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    px, py : scalar or array-like
        X and Y components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Azimuthal angle with the same type as input
    """
    if lib == 'np' and is_jit_available() and not isinstance(px, (float, int)):
        result = _jit_compute_phi_np(px, py)
        # Convert to scalar if both inputs were scalars
        if isinstance(px, (float, int)) and isinstance(py, (float, int)):
            return float(result)
        return result
    else:
        # Let the original function handle awkward arrays and scalars
        return _get_original_compute_phi()(px, py, lib)

def jit_compute_p4_from_ptepm(pt, eta, phi, m, lib):
    """
    JIT-optimized conversion from pt,eta,phi,m to p4 with backend awareness.
    
    Falls back to non-JIT implementation for non-NumPy backends or when JIT is disabled.
    
    Parameters
    ----------
    pt : scalar or array-like
        Transverse momentum
    eta : scalar or array-like
        Pseudorapidity
    phi : scalar or array-like
        Azimuthal angle
    m : scalar or array-like
        Mass
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    tuple
        (px, py, pz, E) with the same type as input
    """
    if lib == 'np' and is_jit_available() and not isinstance(pt, (float, int)):
        return _jit_compute_p4_from_ptepm_np(pt, eta, phi, m)
    else:
        # Let the original function handle awkward arrays and scalars
        return _get_original_compute_p4_from_ptepm()(pt, eta, phi, m, lib)
