import pytest
import numpy as np
from lvec import LVec

def test_boostz():
    """Test boost along z-axis."""
    # Test particle at rest
    v = LVec(0.0, 0.0, 0.0, 1.0)
    bz = 0.5  # beta = 0.5
    v_boost = v.boostz(bz)
    
    # Calculate expected values
    gamma = 1/np.sqrt(1 - bz**2)
    assert abs(v_boost.E - gamma) < 1e-10
    assert abs(v_boost.pz - gamma*bz) < 1e-10
    assert abs(v_boost.px) < 1e-10
    assert abs(v_boost.py) < 1e-10

def test_boost_3d():
    """Test general 3D boost."""
    # Test with arbitrary boost vector
    v = LVec(1.0, 2.0, 3.0, 10.0)
    bx, by, bz = 0.1, 0.2, 0.3
    v_boost = v.boost(bx, by, bz)
    
    # Calculate gamma factor
    b2 = bx**2 + by**2 + bz**2
    gamma = 1/np.sqrt(1 - b2)
    
    # Original 4-momentum dot product should be invariant
    m2_before = v.E**2 - (v.px**2 + v.py**2 + v.pz**2)
    m2_after = v_boost.E**2 - (v_boost.px**2 + v_boost.py**2 + v_boost.pz**2)
    assert abs(m2_before - m2_after) < 1e-10

def test_boost_batch():
    """Test boost with array inputs."""
    # Test with batch of particles
    px = np.array([1.0, 2.0, 3.0])
    py = np.array([0.0, 0.0, 0.0])
    pz = np.array([0.0, 0.0, 0.0])
    E = np.array([2.0, 3.0, 4.0])
    
    v = LVec(px, py, pz, E)
    bz = 0.5
    v_boost = v.boostz(bz)
    
    # Check shapes are preserved
    assert v_boost.px.shape == px.shape
    assert v_boost.py.shape == py.shape
    assert v_boost.pz.shape == pz.shape
    assert v_boost.E.shape == E.shape

def test_boost_limits():
    """Test boost with edge cases."""
    v = LVec(1.0, 0.0, 0.0, 2.0)
    
    # Test small boost
    b_small = 1e-8
    v_small = v.boostz(b_small)
    assert abs(v_small.E - v.E) < 1e-7
    assert abs(v_small.pz - 0.0) < 1e-7
    
    # Test boost close to speed of light
    b_large = 0.9999
    v_large = v.boostz(b_large)
    gamma = 1/np.sqrt(1 - b_large**2)
    assert v_large.E > v.E  # Energy should increase
    assert abs(v_large.mass - v.mass) < 1e-10  # Mass should be invariant

def test_boost_invariant_mass():
    """Test that mass is invariant under boosts."""
    # Test with massive particle
    px, py, pz = 1.0, 2.0, 3.0
    E = np.sqrt(px**2 + py**2 + pz**2 + 5.0)  # mass = 5
    v = LVec(px, py, pz, E)
    
    # Try different boost vectors
    boosts = [
        (0.1, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.3),
        (0.1, 0.2, 0.3),
    ]
    
    initial_mass = v.mass
    for bx, by, bz in boosts:
        v_boost = v.boost(bx, by, bz)
        assert abs(v_boost.mass - initial_mass) < 1e-10

def test_boost_composition():
    """Test that composed boosts work correctly."""
    v = LVec(1.0, 0.0, 0.0, 2.0)
    
    # Apply two successive boosts
    b1 = 0.3
    b2 = 0.4
    v_boost1 = v.boostz(b1)
    v_boost2 = v_boost1.boostz(b2)
    
    # Calculate combined boost
    beta_combined = (b1 + b2)/(1 + b1*b2)
    v_combined = v.boostz(beta_combined)
    
    # Results should match within numerical precision
    assert abs(v_boost2.E - v_combined.E) < 1e-10
    assert abs(v_boost2.pz - v_combined.pz) < 1e-10