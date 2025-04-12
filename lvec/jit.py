"""
JIT compilation support for performance-critical operations in LVec.

This module provides Numba-accelerated versions of core computation functions
while maintaining compatibility with LVec's caching system. Functions gracefully 
fall back to non-JIT versions when Numba is not available.
"""

import numpy as np
import time
import warnings
from functools import lru_cache

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Configuration
JIT_ENABLED = HAS_NUMBA  # Global flag to enable/disable JIT
OPTIMIZE_NUMBA = True    # Apply additional optimizations specific to Numba
PARALLEL = True          # Enable parallel execution where possible
FASTMATH = True          # Enable fast math optimizations (may reduce precision slightly)

# Adaptive batch processing
# Initial threshold - will be adjusted based on runtime performance
BATCH_THRESHOLD = 100_000
MIN_BATCH_THRESHOLD = 10_000
MAX_BATCH_THRESHOLD = 1_000_000
BATCH_THRESHOLD_ADJUSTMENT_FACTOR = 1.5  # How much to adjust threshold up/down

# Performance tracking for adaptive threshold
_performance_history = {}

# Function to check if JIT is available and enabled
def is_jit_available():
    """Check if JIT compilation is available and enabled."""
    return HAS_NUMBA and JIT_ENABLED

def enable_jit(enable=True):
    """Enable or disable JIT compilation globally."""
    global JIT_ENABLED
    if enable and not HAS_NUMBA:
        warnings.warn("Numba not installed, JIT compilation not available")
        return False
    JIT_ENABLED = enable
    return JIT_ENABLED

# ------------------- JIT implementations of core computation functions -------------------

if HAS_NUMBA:
    # Numba-optimized functions for NumPy arrays
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_pt_np(px, py):
        """JIT-optimized pt calculation for NumPy arrays."""
        return np.sqrt(px*px + py*py)
        
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_p_np(px, py, pz):
        """JIT-optimized momentum calculation for NumPy arrays."""
        return np.sqrt(px*px + py*py + pz*pz)
        
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_mass_np(E, px, py, pz):
        """JIT-optimized mass calculation for NumPy arrays."""
        p_squared = px*px + py*py + pz*pz
        m_squared = E*E - p_squared
        
        # Handle numerical precision issues that can lead to small negative mass-squared values
        # For physical particles, m_squared should be positive
        abs_m_squared = np.abs(m_squared)
        return np.sqrt(abs_m_squared)
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_eta_np(p, pz):
        """JIT-optimized eta calculation for NumPy arrays."""
        epsilon = 1e-10  # Small constant to avoid division by zero
        return 0.5 * np.log((p + pz) / (p - pz + epsilon))
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_phi_np(px, py):
        """JIT-optimized phi calculation for NumPy arrays."""
        return np.arctan2(py, px)
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_compute_p4_from_ptepm_np(pt, eta, phi, m):
        """JIT-optimized conversion from pt,eta,phi,m to p4 for NumPy arrays."""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        p = np.sqrt(px*px + py*py + pz*pz)
        E = np.sqrt(p*p + m*m)
        return px, py, pz, E
    
    # Batch processing functions for large arrays
    @nb.njit(cache=True, fastmath=FASTMATH, parallel=PARALLEL)
    def _jit_batch_compute_pt_np(px, py):
        """Batch-optimized pt calculation for large NumPy arrays."""
        result = np.empty_like(px)
        n = len(px)
        for i in nb.prange(n):
            result[i] = np.sqrt(px[i]*px[i] + py[i]*py[i])
        return result
    
    @nb.njit(cache=True, fastmath=FASTMATH, parallel=PARALLEL)
    def _jit_batch_compute_p_np(px, py, pz):
        """Batch-optimized momentum calculation for large NumPy arrays."""
        result = np.empty_like(px)
        n = len(px)
        for i in nb.prange(n):
            result[i] = np.sqrt(px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i])
        return result
    
    @nb.njit(cache=True, fastmath=FASTMATH, parallel=PARALLEL)
    def _jit_batch_compute_mass_np(E, px, py, pz):
        """Batch-optimized mass calculation for large NumPy arrays."""
        result = np.empty_like(E)
        n = len(E)
        for i in nb.prange(n):
            p_squared = px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]
            m_squared = E[i]*E[i] - p_squared
            # Handle numerical precision issues
            abs_m_squared = abs(m_squared)
            result[i] = np.sqrt(abs_m_squared)
        return result
    
    # Vectorized versions that use NumPy's optimized operations directly
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_vectorized_compute_pt_np(px, py):
        """Vectorized pt calculation using NumPy operations."""
        return np.sqrt(px**2 + py**2)
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_vectorized_compute_p_np(px, py, pz):
        """Vectorized momentum calculation using NumPy operations."""
        return np.sqrt(px**2 + py**2 + pz**2)
    
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_vectorized_compute_mass_np(E, px, py, pz):
        """Vectorized mass calculation using NumPy operations."""
        p_squared = px**2 + py**2 + pz**2
        m_squared = E**2 - p_squared
        return np.sqrt(np.abs(m_squared))
    
    # Optimized phi calculation that might perform better with JIT
    @nb.njit(cache=True, fastmath=FASTMATH)
    def _jit_optimized_phi_np(px, py):
        """Optimized phi calculation that avoids arctan2 for common cases."""
        result = np.empty_like(px)
        n = len(px)
        
        for i in range(n):
            x, y = px[i], py[i]
            
            # Handle special cases
            if x == 0.0:
                if y > 0.0:
                    result[i] = np.pi/2
                elif y < 0.0:
                    result[i] = -np.pi/2
                else:
                    result[i] = 0.0
                continue
                
            # Use arctan2 for general case
            result[i] = np.arctan2(y, x)
            
        return result

# Function to determine the optimal implementation based on array size
def _select_optimal_implementation(array_size, operation):
    """
    Select the optimal implementation based on array size and operation.
    
    Parameters
    ----------
    array_size : int
        Size of the input array
    operation : str
        Operation type ('pt', 'p', 'mass', 'eta', 'phi')
        
    Returns
    -------
    str
        Implementation type ('vectorized', 'batch', 'standard')
    """
    # For very small arrays, use standard implementation
    if array_size < MIN_BATCH_THRESHOLD:
        return 'standard'
    
    # For medium-sized arrays, use vectorized implementation
    if array_size < BATCH_THRESHOLD:
        return 'vectorized'
    
    # For large arrays, use batch implementation
    return 'batch'

# Adaptive threshold adjustment
def _update_batch_threshold(operation, implementation, array_size, execution_time):
    """
    Update the batch threshold based on performance data.
    
    Parameters
    ----------
    operation : str
        Operation type ('pt', 'p', 'mass', 'eta', 'phi')
    implementation : str
        Implementation used ('vectorized', 'batch', 'standard')
    array_size : int
        Size of the input array
    execution_time : float
        Execution time in seconds
    """
    global BATCH_THRESHOLD
    
    # Only track performance for arrays near the threshold
    if not (MIN_BATCH_THRESHOLD <= array_size <= MAX_BATCH_THRESHOLD):
        return
    
    # Initialize performance history for this operation if needed
    if operation not in _performance_history:
        _performance_history[operation] = {'vectorized': [], 'batch': []}
    
    # Add performance data
    if implementation in ('vectorized', 'batch'):
        _performance_history[operation][implementation].append((array_size, execution_time))
    
    # Only adjust threshold if we have enough data points
    if (len(_performance_history[operation]['vectorized']) >= 3 and 
        len(_performance_history[operation]['batch']) >= 3):
        
        # Calculate average performance for each implementation
        vec_sizes = [s for s, _ in _performance_history[operation]['vectorized']]
        vec_times = [t for _, t in _performance_history[operation]['vectorized']]
        batch_sizes = [s for s, _ in _performance_history[operation]['batch']]
        batch_times = [t for _, t in _performance_history[operation]['batch']]
        
        # Find average performance per element
        if vec_sizes and vec_times:
            vec_perf = sum(t/s for s, t in zip(vec_sizes, vec_times)) / len(vec_sizes)
        else:
            vec_perf = float('inf')
            
        if batch_sizes and batch_times:
            batch_perf = sum(t/s for s, t in zip(batch_sizes, batch_times)) / len(batch_sizes)
        else:
            batch_perf = float('inf')
        
        # Adjust threshold based on which is faster
        if vec_perf < batch_perf:
            # Vectorized is faster, increase threshold
            new_threshold = min(MAX_BATCH_THRESHOLD, 
                               int(BATCH_THRESHOLD * BATCH_THRESHOLD_ADJUSTMENT_FACTOR))
        else:
            # Batch is faster, decrease threshold
            new_threshold = max(MIN_BATCH_THRESHOLD, 
                               int(BATCH_THRESHOLD / BATCH_THRESHOLD_ADJUSTMENT_FACTOR))
        
        # Update threshold and reset history
        if new_threshold != BATCH_THRESHOLD:
            BATCH_THRESHOLD = new_threshold
            _performance_history[operation] = {'vectorized': [], 'batch': []}

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
        array_size = getattr(px, 'size', len(px) if hasattr(px, '__len__') else 0)
        
        # Select implementation based on array size
        start_time = time.time()
        implementation = _select_optimal_implementation(array_size, 'pt')
        
        if implementation == 'batch':
            result = _jit_batch_compute_pt_np(px, py)
        elif implementation == 'vectorized':
            result = _jit_vectorized_compute_pt_np(px, py)
        else:
            result = _jit_compute_pt_np(px, py)
            
        # Update adaptive threshold
        execution_time = time.time() - start_time
        _update_batch_threshold('pt', implementation, array_size, execution_time)
            
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
        array_size = getattr(px, 'size', len(px) if hasattr(px, '__len__') else 0)
        
        # Select implementation based on array size
        start_time = time.time()
        implementation = _select_optimal_implementation(array_size, 'p')
        
        if implementation == 'batch':
            result = _jit_batch_compute_p_np(px, py, pz)
        elif implementation == 'vectorized':
            result = _jit_vectorized_compute_p_np(px, py, pz)
        else:
            result = _jit_compute_p_np(px, py, pz)
            
        # Update adaptive threshold
        execution_time = time.time() - start_time
        _update_batch_threshold('p', implementation, array_size, execution_time)
            
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
        array_size = getattr(E, 'size', len(E) if hasattr(E, '__len__') else 0)
        
        # Select implementation based on array size
        start_time = time.time()
        implementation = _select_optimal_implementation(array_size, 'mass')
        
        if implementation == 'batch':
            result = _jit_batch_compute_mass_np(E, px, py, pz)
        elif implementation == 'vectorized':
            result = _jit_vectorized_compute_mass_np(E, px, py, pz)
        else:
            result = _jit_compute_mass_np(E, px, py, pz)
            
        # Update adaptive threshold
        execution_time = time.time() - start_time
        _update_batch_threshold('mass', implementation, array_size, execution_time)
            
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
        # Convert to scalar if all inputs were scalars
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
        array_size = getattr(px, 'size', len(px) if hasattr(px, '__len__') else 0)
        
        # For phi, we'll try the optimized implementation for large arrays
        if array_size > BATCH_THRESHOLD:
            result = _jit_optimized_phi_np(px, py)
        else:
            result = _jit_compute_phi_np(px, py)
            
        # Convert to scalar if all inputs were scalars
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

# Pre-compilation function to eliminate first-run overhead
def precompile_jit_functions():
    """
    Pre-compile all JIT functions to eliminate first-run compilation overhead.
    
    This function should be called once at module import time or when JIT is enabled.
    """
    if not is_jit_available():
        return False
    
    try:
        # Create small test arrays
        size = 10
        px = np.ones(size)
        py = np.ones(size)
        pz = np.ones(size)
        E = np.ones(size) * 2
        p = np.ones(size) * np.sqrt(3)
        pt = np.ones(size) * np.sqrt(2)
        eta = np.zeros(size)
        phi = np.zeros(size)
        m = np.ones(size)
        
        # Pre-compile standard functions
        _ = _jit_compute_pt_np(px, py)
        _ = _jit_compute_p_np(px, py, pz)
        _ = _jit_compute_mass_np(E, px, py, pz)
        _ = _jit_compute_eta_np(p, pz)
        _ = _jit_compute_phi_np(px, py)
        _ = _jit_compute_p4_from_ptepm_np(pt, eta, phi, m)
        
        # Pre-compile vectorized functions
        _ = _jit_vectorized_compute_pt_np(px, py)
        _ = _jit_vectorized_compute_p_np(px, py, pz)
        _ = _jit_vectorized_compute_mass_np(E, px, py, pz)
        
        # Pre-compile batch functions
        _ = _jit_batch_compute_pt_np(px, py)
        _ = _jit_batch_compute_p_np(px, py, pz)
        _ = _jit_batch_compute_mass_np(E, px, py, pz)
        
        # Pre-compile optimized phi
        _ = _jit_optimized_phi_np(px, py)
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to pre-compile JIT functions: {e}")
        return False

# Attempt to pre-compile if JIT is available
if HAS_NUMBA and JIT_ENABLED:
    precompile_jit_functions()
