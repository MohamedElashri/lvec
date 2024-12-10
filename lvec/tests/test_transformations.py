import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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
