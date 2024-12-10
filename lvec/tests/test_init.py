import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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

