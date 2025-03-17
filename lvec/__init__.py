# __init__.py
from .lvec import LVec
from .exceptions import LVecError, ShapeError, DependencyError

__all__ = ['LVec', 'LVecError', 'ShapeError', 'DependencyError']
__version__ = '0.1.2'