import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

def test_caching():
    v = LVec(1.0, 1.0, 1.0, 2.0)
    
    # Access pt to cache it
    initial_pt = v.pt
    
    # Verify it's cached 
    assert 'pt' in v._cache._values
    
    # Store the value for comparison
    stored_pt_value = v._cache._values['pt']
    
    # Touch and verify cache is invalidated
    v.touch()  # This will call touch_component on all components
    
    # In the new cache system, touching the components should clear the cached value
    assert 'pt' not in v._cache._values
    
    # Access pt again and verify it's recalculated (but same value)
    recalculated_pt = v.pt
    assert recalculated_pt == initial_pt
    assert 'pt' in v._cache._values
    
    # Verify the value is cached again
    assert v._cache._values['pt'] is not None

def test_fine_grained_cache_invalidation():
    """Test the fine-grained cache invalidation feature of the new PropertyCache system."""
    v = LVec(1.0, 1.0, 1.0, 2.0)
    
    # Access multiple properties to cache them
    pt_value = v.pt
    phi_value = v.phi
    eta_value = v.eta
    mass_value = v.mass
    
    # Verify all are cached
    assert 'pt' in v._cache._values
    assert 'phi' in v._cache._values
    assert 'eta' in v._cache._values
    assert 'mass' in v._cache._values
    
    # Modify only px component - should invalidate pt, phi, eta, mass but not pz-only dependent properties
    v._cache.touch_component('px')
    
    # Properties dependent on px should be removed from cache
    assert 'pt' not in v._cache._values
    assert 'phi' not in v._cache._values
    assert 'eta' not in v._cache._values
    assert 'mass' not in v._cache._values
    
    # Access pt again and verify only pt is recalculated and cached again
    new_pt = v.pt
    assert 'pt' in v._cache._values
    
    # Other properties shouldn't be in cache until accessed
    assert 'phi' not in v._cache._values
    assert 'eta' not in v._cache._values
    assert 'mass' not in v._cache._values
