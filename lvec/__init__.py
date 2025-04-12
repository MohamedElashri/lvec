# __init__.py
from .lvec import LVec
from .vectors import Vec2D as Vector2D, Vec3D as Vector3D
from .exceptions import ShapeError
from .frame import Frame

# Import JIT functions and settings
try:
    from .jit import (enable_jit, is_jit_available, precompile_jit_functions,
                     BATCH_THRESHOLD, MIN_BATCH_THRESHOLD, MAX_BATCH_THRESHOLD)
except ImportError:
    # Create dummy functions if JIT is not available
    def enable_jit(enable=True):
        """Dummy function when Numba is not available."""
        import warnings
        warnings.warn("Numba not installed, JIT compilation not available")
        return False
        
    def is_jit_available():
        """Check if JIT compilation is available."""
        return False
        
    def precompile_jit_functions():
        """Dummy function when Numba is not available."""
        return False
        
    # Dummy threshold values
    BATCH_THRESHOLD = 100_000
    MIN_BATCH_THRESHOLD = 10_000
    MAX_BATCH_THRESHOLD = 1_000_000

def get_jit_batch_threshold():
    """
    Get the current batch processing threshold for JIT operations.
    
    Returns
    -------
    int
        Current threshold value. Arrays larger than this will use parallel batch processing.
    """
    return BATCH_THRESHOLD

__all__ = ['LVec', 'Vector2D', 'Vector3D', 'Frame', 'ShapeError', 
           'enable_jit', 'is_jit_available', 'precompile_jit_functions',
           'get_jit_batch_threshold']
__version__ = '0.1.4'
