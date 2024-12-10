import pytest
import numpy as np
import awkward as ak
from lvec import LVec, ShapeError

def test_init():
    # Test scalar inputs
    v = LVec(1.0, 2.0, 3.0, 4.0)
    assert v.px == 1.0
    assert v.E == 4.0
    
    # Test numpy arrays
    data = np.array([[1, 2], [3, 4]])
    v = LVec(data[:, 0], data[:, 0], data[:, 1], data[:, 1])
    assert np.all(v.px == data[:, 0])
    
    # Test awkward arrays
    data = ak.Array([[1, 2], [3, 4]])
    v = LVec(data[:, 0], data[:, 0], data[:, 1], data[:, 1])
    assert ak.all(v.px == data[:, 0])

def test_shape_error():
    with pytest.raises(ShapeError):
        LVec([1, 2], [1], [1, 2], [1, 2])

def test_arithmetic():
    # Test addition
    v1 = LVec(1.0, 2.0, 3.0, 4.0)
    v2 = LVec(2.0, 3.0, 4.0, 5.0)
    v3 = v1 + v2
    assert v3.px == 3.0
    assert v3.py == 5.0
    assert v3.pz == 7.0
    assert v3.E == 9.0
    
    # Test scalar multiplication
    v4 = v1 * 2
    assert v4.px == 2.0
    assert v4.py == 4.0
    assert v4.pz == 6.0
    assert v4.E == 8.0

def test_properties():
    # Test derived properties and caching
    v = LVec(3.0, 4.0, 0.0, 7.0)
    
    # Test pt
    assert v.pt == 5.0  # 3-4-5 triangle
    
    # Test p
    assert v.p == 5.0  # pz = 0
    
    # Test mass
    expected_mass = np.sqrt(7.0**2 - 5.0**2)
    assert abs(v.mass - expected_mass) < 1e-10
    
    # Test phi
    assert abs(v.phi - np.arctan2(4.0, 3.0)) < 1e-10

def test_caching():
    v = LVec(1.0, 1.0, 1.0, 2.0)
    
    # Access pt to cache it
    initial_pt = v.pt
    
    # Verify it's cached
    assert 'pt' in v._cache
    assert v._cache['pt']['version'] == v._version
    
    # Touch and verify cache is invalidated
    v.touch()
    assert 'pt' not in v._cache

def test_conversions():
    # Test NumPy conversion
    v = LVec(1.0, 2.0, 3.0, 4.0)
    np_dict = v.to_np()
    assert isinstance(np_dict['px'], np.ndarray)
    assert np_dict['px'] == 1.0
    
    # Test Awkward conversion
    ak_dict = v.to_ak()
    assert isinstance(ak_dict['px'], ak.Array)
    assert ak.to_numpy(ak_dict['px']) == 1.0
    
    # Test pt, eta, phi, mass conversion
    ptepm = v.to_ptepm()
    pt, eta, phi, mass = ptepm
    assert isinstance(pt, (float, np.ndarray, ak.Array))
    
    # Test ROOT dictionary conversion
    root_dict = v.to_root_dict()
    assert 'fX' in root_dict
    assert isinstance(root_dict['fX'], np.ndarray)
    assert root_dict['fX'] == 1.0

def test_from_constructors():
    # Test from_p4
    v1 = LVec.from_p4(1.0, 2.0, 3.0, 4.0)
    assert v1.px == 1.0
    
    # Test from_ptepm
    pt, eta, phi, m = 5.0, 0.0, 0.0, 1.0
    v2 = LVec.from_ptepm(pt, eta, phi, m)
    assert abs(v2.pt - pt) < 1e-10
    assert abs(v2.mass - m) < 1e-10

def test_transformations():
    # Test rotations
    v = LVec(1.0, 0.0, 0.0, 2.0)
    v_rot = v.rotz(np.pi/2)
    assert abs(v_rot.px) < 1e-10
    assert abs(v_rot.py - 1.0) < 1e-10
    
    # Test boost
    v = LVec(0.0, 0.0, 0.0, 1.0)
    bz = 0.5  # beta = 0.5
    v_boost = v.boostz(bz)
    gamma = 1/np.sqrt(1 - bz**2)
    assert abs(v_boost.E - gamma) < 1e-10
    assert abs(v_boost.pz - gamma*bz) < 1e-10