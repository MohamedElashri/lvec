import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

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

