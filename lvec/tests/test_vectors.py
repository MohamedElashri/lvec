import pytest
import numpy as np
import awkward as ak
from lvec.vectors import Vec2D, Vec3D
from lvec.exceptions import ShapeError, InputError, BackendError

def test_vec2d_scalar():
    """Test 2D vector with scalar inputs"""
    v = Vec2D(3.0, 4.0)
    assert v.x == 3.0
    assert v.y == 4.0
    assert v.r == 5.0  # 3-4-5 triangle
    assert np.isclose(v.phi, np.arctan2(4.0, 3.0))

def test_vec2d_numpy():
    """Test 2D vector with numpy arrays"""
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 3.0])
    v = Vec2D(x, y)
    assert np.all(v.x == x)
    assert np.all(v.y == y)
    assert np.all(np.isclose(v.r, np.sqrt(x**2 + y**2)))

def test_vec2d_awkward():
    """Test 2D vector with awkward arrays"""
    x = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    y = ak.Array([[2.0, 3.0], [4.0, 5.0]])
    v = Vec2D(x, y)
    assert ak.all(v.x == x)
    assert ak.all(v.y == y)

def test_vec2d_operations():
    """Test 2D vector operations"""
    v1 = Vec2D(1.0, 2.0)
    v2 = Vec2D(2.0, 3.0)
    v3 = v1 + v2
    assert v3.x == 3.0
    assert v3.y == 5.0
    
    v4 = v2 - v1
    assert v4.x == 1.0
    assert v4.y == 1.0
    
    v5 = 2 * v1
    assert v5.x == 2.0
    assert v5.y == 4.0

def test_vec3d_scalar():
    """Test 3D vector with scalar inputs"""
    v = Vec3D(1.0, 2.0, 2.0)
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 2.0
    assert v.r == 3.0
    assert v.rho == np.sqrt(5.0)
    assert np.isclose(v.theta, np.arctan2(np.sqrt(5.0), 2.0))

def test_vec3d_numpy():
    """Test 3D vector with numpy arrays"""
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 3.0])
    z = np.array([3.0, 4.0])
    v = Vec3D(x, y, z)
    assert np.all(v.x == x)
    assert np.all(v.y == y)
    assert np.all(v.z == z)
    assert np.all(np.isclose(v.r, np.sqrt(x**2 + y**2 + z**2)))

def test_vec3d_awkward():
    """Test 3D vector with awkward arrays"""
    x = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    y = ak.Array([[2.0, 3.0], [4.0, 5.0]])
    z = ak.Array([[3.0, 4.0], [5.0, 6.0]])
    v = Vec3D(x, y, z)
    assert ak.all(v.x == x)
    assert ak.all(v.y == y)
    assert ak.all(v.z == z)

def test_vec3d_operations():
    """Test 3D vector operations"""
    v1 = Vec3D(1.0, 2.0, 3.0)
    v2 = Vec3D(2.0, 3.0, 4.0)
    v3 = v1 + v2
    assert v3.x == 3.0
    assert v3.y == 5.0
    assert v3.z == 7.0
    
    v4 = v2 - v1
    assert v4.x == 1.0
    assert v4.y == 1.0
    assert v4.z == 1.0
    
    v5 = 2 * v1
    assert v5.x == 2.0
    assert v5.y == 4.0
    assert v5.z == 6.0