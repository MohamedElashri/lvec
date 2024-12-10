import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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

