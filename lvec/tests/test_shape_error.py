import pytest

from lvec import LVec, ShapeError
import numpy as np
import awkward as ak

def test_shape_error():
    with pytest.raises(ShapeError):
        LVec([1, 2], [1], [1, 2], [1, 2])

