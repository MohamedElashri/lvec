import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

def test_from_constructors():
    # Test from_p4
    v1 = LVec.from_p4(1.0, 2.0, 3.0, 4.0)
    assert v1.px == 1.0
    
    # Test from_ptepm
    pt, eta, phi, m = 5.0, 0.0, 0.0, 1.0
    v2 = LVec.from_ptepm(pt, eta, phi, m)
    assert abs(v2.pt - pt) < 1e-10
    assert abs(v2.mass - m) < 1e-10

