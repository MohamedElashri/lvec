"""Physics, structure, and ownership contracts for Awkward optimizations."""

import numpy as np
import pytest

from lvec import LVec, ShapeError
from lvec.utils import check_shapes


@pytest.fixture
def ak():
    return pytest.importorskip('awkward')


@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_plain_jagged_structure_uses_exact_offsets_and_accepts_slices(ak, monkeypatch, index_dtype):
    def array(offsets, values):
        return ak.Array(ak.contents.ListOffsetArray(
            ak.index.Index(np.asarray(offsets, dtype=index_dtype)),
            ak.contents.NumpyArray(np.asarray(values, dtype=float))))

    left = array([0, 2, 2, 3], [1., 2., 3., 100.])
    shifted = array([2, 4, 4, 5], [100., 100., 4., 5., 6., 100.])
    wrong = array([0, 1, 2, 3], [1., 2., 3.])
    monkeypatch.setattr(ak, 'almost_equal', lambda *a, **k: pytest.fail('generic structural comparison'))
    check_shapes(left, shifted, 'ak')
    check_shapes(left[1:], shifted[1:], 'ak')
    with pytest.raises(ShapeError):
        check_shapes(left, wrong, 'ak')
    with pytest.raises(ShapeError):
        LVec(left, wrong, left, left)


def test_shape_validation_matches_independent_nested_structure_oracle(ak):
    def structure(value):
        if isinstance(value, list):
            return tuple(structure(x) for x in value)
        return None if value is None else 'number'

    rng = np.random.default_rng(20261102)
    samples = [[], [[], []], [None, []], [[1., None], []],
               [[[1.], []], None], [[[]], []]]
    for _ in range(40):
        sample = []
        for _ in range(int(rng.integers(1, 8))):
            sample.append(None if rng.random() < .2 else
                          [None if rng.random() < .2 else float(rng.normal())
                           for _ in range(int(rng.integers(0, 6)))])
        samples.append(sample)
    for index, left in enumerate(samples):
        for right in (left, samples[(index + 1) % len(samples)]):
            a, b = ak.Array(left), ak.Array(right)
            if structure(left) == structure(right):
                check_shapes(a, b, 'ak')
            else:
                with pytest.raises(ShapeError):
                    check_shapes(a, b, 'ak')


def test_general_layouts_retain_structural_validation(ak):
    ordinary = ak.Array([[1., 2.], [], [3.]])
    layout = ordinary.layout
    lists = ak.contents.ListArray(layout.offsets[:-1], layout.offsets[1:], layout.content)
    check_shapes(ak.Array(lists), ordinary, 'ak')
    regular = ak.Array(np.ones((3, 2)))
    check_shapes(regular, ak.Array([[4., 5.], [6., 7.], [8., 9.]]), 'ak')
    for left, right in [([[1., None], []], [[None, 1.], []]),
                        ([None, [1.]], [[1.], None]),
                        ([[[1.], None]], [[None, [1.]]])]:
        with pytest.raises(ShapeError):
            check_shapes(ak.Array(left), ak.Array(right), 'ak')
