# __init__.py
from .lvec import LVec
from .vectors import Vec2D as Vector2D, Vec3D as Vector3D
from .exceptions import ShapeError
from .frame import Frame

# Import JIT functions and settings
try:
    from .jit import enable_jit, is_jit_available
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

__all__ = ['LVec', 'Vector2D', 'Vector3D', 'Frame', 'ShapeError', 'enable_jit', 'is_jit_available']
__version__ = '0.1.4'
