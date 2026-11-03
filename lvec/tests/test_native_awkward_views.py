"""Native records keep view semantics while using the shared physics kernels."""
import numpy as np
import pytest

from lvec import LVec, Vector2D, Vector3D, ShapeError
from lvec.exceptions import BackendError, InputError

ak = pytest.importorskip('awkward')
from lvec.awkward import zip as native_zip, behavior
from .test_awkward_fast_paths import shaped


@pytest.mark.parametrize('dimension,cls', [(2, Vector2D), (3, Vector3D), (4, LVec)])
@pytest.mark.parametrize('kind', ['regular', 'lists', 'sliced', 'list_array', 'indexed',
                                 'byte_mask', 'bit_mask', 'missing', 'deep'])
def test_native_arithmetic_preserves_values_and_missing_structure(dimension, cls, kind):
    x = np.arange(1., 9.)
    columns = [shaped(ak, factor*x, kind) for factor in (1., 2., 3., 5.)[:dimension]]
    names = ('px', 'py', 'pz', 'E') if dimension == 4 else ('x', 'y', 'z')[:dimension]
    native = native_zip(dict(zip(names, columns)))
    owned = cls(*columns)
    expected = [owned+owned, owned-owned, owned*2., owned/2.]
    actual = [native+native, native-native, native*2., native/2.]
    for result, reference in zip(actual, expected):
        for name, column in zip(names, reference._components()):
            assert ak.to_list(result[name]) == ak.to_list(column)
    assert cls.from_ak_record(ak.sum(native, axis=-1)).allclose(owned.sum(axis=-1))


@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_native_arithmetic_does_not_copy_application_metadata(dimension):
    class Context:
        def __deepcopy__(self, memo):
            raise AssertionError('Native operations must not deep-copy application context')

    names = ('px', 'py', 'pz', 'E') if dimension == 4 else ('x', 'y', 'z')[:dimension]
    record = native_zip({name: ak.Array([float(i), float(i+1)], attrs={'context': Context()})
                         for i, name in enumerate(names, 1)})
    context = Context()
    record.behavior = dict(record.behavior, application_context=context)
    assert record[ak.fields(record)[0]].behavior['application_context'] is context
    for result in (record+record, record-record, record*2., -record):
        assert len(result) == 2
    # Scalar records also use the same arithmetic without an owned detour.
    assert (record[0]+record[0])[names[0]] == 2.


def test_native_view_stays_live_after_properties_and_arithmetic():
    x = np.array([3., 4.])
    record = native_zip({'px': x, 'py': np.zeros(2), 'pz': np.zeros(2), 'E': np.full(2, 13.)})
    before = ak.to_numpy(record.mass).copy()
    _ = record+record
    assert x.flags.writeable
    x[:] = [5., 12.]
    np.testing.assert_allclose(record.mass, [12., 5.])
    assert not np.array_equal(record.mass, before)


def external(fields):
    return ak.zip(fields, depth_limit=1, with_name='LVec4D', behavior=behavior)


@pytest.mark.parametrize('dtype', [np.int16, np.float32])
def test_other_real_dtypes_keep_normalization(dtype):
    record = external({name: np.array([value], dtype=dtype)
                       for name, value in zip(('px', 'py', 'pz', 'E'), (3, 0, 0, 5))})
    np.testing.assert_allclose(record.mass, [4.])


def test_record_name_does_not_bypass_per_field_shape_and_mask_checks():
    for px, py in [(ak.Array([[1.], []]), ak.Array([[0., 0.], []])),
                   (ak.Array([1., None]), ak.Array([None, 0.])),
                   (np.ones((2, 2)), np.ones((2, 3)))]:
        record = external({'px': px, 'py': py, 'pz': py, 'E': py})
        with pytest.raises(ShapeError):
            _ = record.mass


@pytest.mark.parametrize('value', [ak.Array(['bad']), ak.Array([1., 'bad']), np.array([1+2j])])
def test_record_name_does_not_bypass_real_numeric_validation(value):
    record = external({'px': value, 'py': [0.], 'pz': [0.], 'E': [5.]})
    with pytest.raises(BackendError):
        _ = record.mass


def test_rest_frame_domain_rejection_survives_native_fast_path():
    particle = native_zip({'px': [1.], 'py': [0.], 'pz': [0.], 'E': [2.]})
    null = native_zip({'px': [1.], 'py': [0.], 'pz': [0.], 'E': [1.]})
    with pytest.raises(InputError):
        particle.boostCM_of_p4(null)


@pytest.mark.parametrize('values', [[], [[], []], [None, None], [[None], [], None]])
def test_native_empty_and_all_missing_fields_preserve_structure(values):
    record = native_zip({name: ak.Array(values) for name in ('px', 'py', 'pz', 'E')})
    assert ak.to_list(record.mass) == values
    assert ak.to_list((record+record).mass) == values


@pytest.mark.parametrize('dimension', [2, 3, 4])
@pytest.mark.parametrize('operation', [np.add, np.subtract])
@pytest.mark.parametrize('result_kind', ['nonnumeric', 'different_shape'])
def test_custom_component_arithmetic_cannot_change_numeric_or_shape_contract(dimension, operation, result_kind):
    names = ('x', 'y', 'z', 't')[:dimension]
    record = native_zip({name: ak.Array([float(i), float(i+1)])
                         for i, name in enumerate(names, 1)})
    tagged = ak.with_parameter(record.x, '__array__', 'custom_component')
    tagged = ak.with_parameter(tagged, '__list__', 'custom_component')
    record = ak.with_field(record, tagged, 'x')

    def custom(left, right):
        if result_kind == 'nonnumeric':
            return ak.Array(['invalid']*len(left))
        return ak.Array([[1., 2.] for _ in range(len(left))])

    record.behavior = dict(record.behavior)
    record.behavior[operation, 'custom_component', 'custom_component'] = custom
    with pytest.raises((BackendError, ShapeError, TypeError)):
        operation(record, record)


@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_custom_real_results_and_scalar_array_factors_are_promoted(dimension):
    names = ('x', 'y', 'z', 't')[:dimension]
    record = native_zip({name: ak.Array([float(i), float(i+1)])
                         for i, name in enumerate(names, 1)})
    tagged = ak.with_parameter(record.x, '__array__', 'custom_component')
    tagged = ak.with_parameter(tagged, '__list__', 'custom_component')
    record = ak.with_field(record, tagged, 'x')
    record.behavior = dict(record.behavior)
    record.behavior[np.add, 'custom_component', 'custom_component'] = (
        lambda left, right: ak.Array(np.full(len(left), 2., dtype=np.float32)))
    result = record+record
    field = 'px' if dimension == 4 else 'x'
    assert ak.to_numpy(result[field]).dtype == np.dtype('float64')
    assert ak.to_list(result[field]) == [2., 2.]
    scaled = record[0].scale(np.array([2., 3.], dtype=np.float32))
    assert ak.to_numpy(scaled['px' if dimension == 4 else 'x']).dtype == np.dtype('float64')
