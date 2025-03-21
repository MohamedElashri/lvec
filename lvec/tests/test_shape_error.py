import pytest

from lvec import LVec, Vector2D, ShapeError
import numpy as np
import awkward as ak

def test_shape_error():
    with pytest.raises(ShapeError):
        LVec([1, 2], [1], [1, 2], [1, 2])

def test_jagged_awkward_shape_error():
    """Test that jagged awkward arrays with inconsistent inner dimensions raise a ShapeError."""
    # Create jagged arrays with inconsistent inner dimensions
    x = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]])
    y = ak.Array([[2.0, 3.0], [4.0, 5.0]])  # Second inner array has different length
    
    with pytest.raises(ShapeError):
        Vector2D(x, y)
        
def test_jagged_awkward_consistent_inner_dimensions():
    """Test that jagged awkward arrays with consistent inner dimensions work correctly."""
    # Create jagged arrays with consistent inner dimensions
    x = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    y = ak.Array([[2.0, 3.0], [4.0, 5.0]])  # All inner arrays have the same length
    
    # This should not raise an error
    vec = Vector2D(x, y)
    assert ak.all(vec.x == x)
    assert ak.all(vec.y == y)
