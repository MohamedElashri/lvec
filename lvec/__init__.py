# __init__.py
from .lvec import LVec
from .vectors import Vec2D as Vector2D, Vec3D as Vector3D
from .exceptions import ShapeError
from .frame import Frame
from ._coordinates import obj, array, arr, Array, awk, zip, install_factories

Vector4D = LVec
Momentum2D = Vector2D
Momentum3D = Vector3D
Momentum4D = LVec
for _class in (Vector2D, Vector3D, LVec):
    install_factories(_class, _class._dimension)

__all__ = ['LVec', 'Vector2D', 'Vector3D', 'Vector4D', 'Momentum2D',
           'Momentum3D', 'Momentum4D', 'Frame', 'ShapeError',
           'obj', 'array', 'arr', 'Array', 'awk', 'zip']
__version__ = '0.2.0'
