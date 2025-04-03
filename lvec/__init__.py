# __init__.py
from .lvec import LVec
from .vectors import Vec2D as Vector2D, Vec3D as Vector3D
from .exceptions import ShapeError

__all__ = ['LVec', 'Vector2D', 'Vector3D', 'ShapeError']
__version__ = '0.1.4'