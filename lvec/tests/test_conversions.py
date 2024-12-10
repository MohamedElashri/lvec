import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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

