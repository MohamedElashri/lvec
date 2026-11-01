"""Numerical and ownership contracts for scalar-parameter transform kernels."""
import math
from fractions import Fraction

import numpy as np
import pytest

from lvec import LVec, Vector2D, Vector3D
from lvec.exceptions import BackendError


def reference_dot(point,row):
    value=sum(Fraction.from_float(float(a))*Fraction.from_float(float(b))
              for a,b in zip(point,row))
    try:return float(value)
    except OverflowError:return math.copysign(math.inf,-1. if value<0 else 1.)


@pytest.mark.parametrize('size',[1,8,9,65539])
def test_constant_matrix_range_recovery_across_block_boundaries(size):
    rng=np.random.default_rng(720261102)
    data=rng.normal(size=(3,size))
    tiny=np.nextafter(0.,1.)
    fixtures=[(1e308,1e308,1e308),(tiny,tiny,0.),
              (1.,np.nextafter(1.,0.),1e-300),(1e308,1e308,1e-300)]
    positions=sorted(set(i%size for i in (0,7,8,65535,65536,65538)))
    for i,position in enumerate(positions):data[:,position]=fixtures[i%len(fixtures)]
    matrix=np.array([[1.,1.,-1.],[.5,.5,0.],[1.,-1.,1.]])
    source=LVec(*data,np.full(size,5.))
    with np.errstate(all='raise'):
        result=source.transform3D(matrix)
    for position in positions:
        for output,row in zip(result._components()[:3],matrix):
            expected=reference_dot(data[:,position],row)
            if expected==0:
                assert output[position]==0.
            elif np.isinf(expected):assert output[position]==expected
            else:assert output[position]==pytest.approx(expected,rel=4e-15,abs=0.)
    assert np.array_equal(result.E,source.E)
    for c in result._components():assert not c.flags.writeable


@pytest.mark.parametrize('cls,n',[(Vector2D,2),(Vector3D,3),(LVec,4)])
def test_constant_matrices_preserve_shape_and_nonfinite_ieee_rules(cls,n):
    values=[np.array([[1.,0.,-0.],[np.inf,np.nan,-2.]]) for _ in range(n)]
    source=cls(*values)
    matrix=np.eye(n)
    with np.errstate(invalid='ignore'):
        result=getattr(source,f'transform{n}D')(matrix)
        expected=[sum(matrix[i,j]*values[j] for j in range(n)) for i in range(n)]
    for actual,wanted in zip(result._components(),expected):
        np.testing.assert_array_equal(actual,wanted)
    ordinary=cls(*(np.arange(12.).reshape(3,4)+i for i in range(n)))[::-1,::2]
    transformed=getattr(ordinary,f'transform{n}D')(matrix)
    assert transformed.allclose(ordinary)
    assert transformed.shape==(3,2)


def test_mutable_scalar_parameters_and_coefficient_cache_eviction():
    v=LVec(1.,2.,3.,8.)
    angle=np.array(.2)
    first=v.rotate_euler(angle,.3,.4)
    angle[...]=.7
    second=v.rotate_euler(angle,.3,.4)
    assert not first.allclose(second)
    expected=v.rotateZ(-.2).rotateX(-.3).rotateZ(-.4)
    assert first.allclose(expected,rtol=1e-14)
    for i in range(260):v.rotate_euler(.001*i,.3,.4)
    assert v.rotate_euler(.2,.3,.4).allclose(first,rtol=1e-14)
    coefficient=np.array(2.)
    matrix={'xx':coefficient,'xy':0.,'yx':0.,'yy':1.}
    result=v.transform2D(matrix)
    coefficient[...]=9.
    assert result.x==2.
    assert v.transform2D(matrix).x==9.


def test_zero_signs_and_subnormal_diagonal_products():
    tiny=np.nextafter(0.,1.)
    source=LVec(np.array([0.,-0.,tiny,-tiny]),0.,0.,1.)
    result=source.transform4D(np.diag([-.5,1.,1.,1.]))
    np.testing.assert_array_equal(result.x,[0.,0.,-0.,0.])
    np.testing.assert_array_equal(np.signbit(result.x),[False,False,True,False])


def test_scalar_shape_and_subclass_initialization_are_preserved():
    v=LVec(1.,2.,3.,8.).rotate_euler(.1,.2,.3)
    assert isinstance(v.px,float) and v.shape==()
    class Custom(LVec):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.initialized=True
    result=Custom(np.ones(2),0.,0.,3.).transform4D(np.eye(4))
    assert type(result) is Custom and result.initialized
    with pytest.raises(ValueError):LVec(1.,0.,0.,2.).transform3D(1.)


def test_numpy_vector_with_awkward_boost_parameters_selects_output_backend():
    ak=pytest.importorskip('awkward')
    v=LVec(1.,0.,0.,3.)
    result=v.boost(ak.Array([[.1,None],[],[.2]]),0.,0.)
    assert result._lib=='ak'
    assert ak.to_list(ak.is_none(result.E,axis=-1))==[[False,True],[],[False]]
    for beta,actual in zip((.1,.2),ak.to_list(ak.flatten(result.E,axis=None))):
        assert actual==pytest.approx((3.+beta)/math.sqrt(1-beta*beta))


@pytest.mark.parametrize('nested',[False,True])
def test_owned_plain_awkward_snapshots_external_data_and_structure(nested):
    ak=pytest.importorskip('awkward')
    offsets=np.array([0,2,4],dtype=np.int64)
    data=np.array([1.,2.,3.,4.])
    array=ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets),ak.contents.NumpyArray(data)))
    if nested:array=ak.unflatten(array,1)
    source=LVec(array,array,array,array*4)
    before=[ak.to_list(c) for c in source._components()]
    data[:]=9.;offsets[:]=[0,1,4]
    assert [ak.to_list(c) for c in source._components()]==before
    for c in source._components():
        _,_,buffers=ak.to_buffers(c)
        assert all(not b.flags.writeable for b in buffers.values())


def test_native_cartesian_factory_views_are_live_and_owned_conversion_snapshots():
    ak=pytest.importorskip('awkward')
    from lvec.awkward import zip as record_zip,Array
    x=np.array([1.,2.])
    records=record_zip(dict(px=ak.Array(x),py=0.,pz=0.,E=5.))
    owned=LVec.from_ak_record(records)
    original=ak.to_list(records.mass)
    assert x.flags.writeable
    x[:]=[2.,3.]
    assert ak.to_list(records.mass)!=original
    np.testing.assert_array_equal(ak.to_numpy(owned.px),[1.,2.])
    from_array=Array(ak.zip({'x':x,'y':np.zeros(2)}))
    assert from_array.allclose(Vector2D(x,0.))
    with pytest.raises(BackendError,match='real numeric'):
        record_zip(dict(x=ak.Array([1.+2.j]),y=0.))
