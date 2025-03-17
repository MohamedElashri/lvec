import pytest
import numpy as np
import awkward as ak
from lvec import LVec
from lvec.exceptions import ShapeError, InputError, BackendError, DependencyError

def test_init_scalar():
    """Test initialization with scalar inputs"""
    v = LVec(1.0, 2.0, 3.0, 4.0)
    assert v.px == 1.0
    assert v.E == 4.0

def test_init_numpy():
    """Test initialization with numpy arrays"""
    data = np.array([[1, 2], [3, 4]])
    v = LVec(data[:, 0], data[:, 0], data[:, 1], data[:, 1])
    assert np.all(v.px == data[:, 0])

def test_init_awkward():
    """Test initialization with awkward arrays"""
    data = ak.Array([[1, 2], [3, 4]])
    v = LVec(data[:, 0], data[:, 0], data[:, 1], data[:, 1])
    assert ak.all(v.px == data[:, 0])

def test_shape_errors():
    """Test shape validation errors"""
    # Test mismatched numpy shapes
    with pytest.raises(ShapeError) as excinfo:
        LVec(np.array([1, 2]), np.array([1]), np.array([1]), np.array([1]))
    assert "Inconsistent array shapes" in str(excinfo.value)
    assert "(2,)" in excinfo.value.shapes[0]

    # Test mismatched awkward lengths
    with pytest.raises(ShapeError) as excinfo:
        LVec(ak.Array([1, 2]), ak.Array([1]), ak.Array([1]), ak.Array([1]))
    assert "Inconsistent array lengths" in str(excinfo.value)

def test_input_validation():
    """Test input value validation"""
    # Test negative energy
    with pytest.raises(InputError) as excinfo:
        LVec(1.0, 1.0, 1.0, -1.0)
    assert "Energy must be non-negative" in str(excinfo.value)

    # Test array with negative energy
    with pytest.raises(InputError) as excinfo:
        LVec(np.array([1.0]), np.array([1.0]), 
             np.array([1.0]), np.array([-1.0]))
    assert "energy values must be non-negative" in str(excinfo.value)

def test_boost_validation():
    """Test boost operation validation"""
    v = LVec(1.0, 1.0, 1.0, 2.0)
    
    # Test invalid boost velocity
    with pytest.raises(InputError) as excinfo:
        v.boost(0.7, 0.7, 0.7)  # sqrt(0.7^2 * 3) > 1
    assert "boost velocity" in str(excinfo.value)
    assert "Must be < 1" in str(excinfo.value)

    # Test array boost with invalid velocities
    with pytest.raises(InputError) as excinfo:
        v.boost(np.array([0.7, 0.1]), np.array([0.7, 0.1]), 
                np.array([0.7, 0.1]))
    assert "All values must be < 1" in str(excinfo.value)

def test_backend_errors():
    """Test backend operation errors"""
    # Test invalid conversion
    with pytest.raises(BackendError) as excinfo:
        LVec({"invalid": "data"}, 1.0, 1.0, 1.0)
    assert "initialization" in str(excinfo.value)

    # Test invalid boost operation
    v = LVec(1.0, 1.0, 1.0, 2.0)
    with pytest.raises(BackendError) as excinfo:
        v.boost({"invalid": "data"}, 0, 0)
    assert "boost" in str(excinfo.value)
