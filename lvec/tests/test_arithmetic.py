import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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

