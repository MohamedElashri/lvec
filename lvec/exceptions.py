# exceptions.py
class LVecError(Exception):
    """Base exception class for LVec package."""
    pass

class ShapeError(LVecError):
    """Raised when array shapes are inconsistent."""
    pass

class DependencyError(LVecError):
    """Raised when required dependencies are not available."""
    pass