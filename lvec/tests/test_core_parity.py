"""Independent contracts for the numerical vector API and migration surface."""
import numpy as np
import pytest

from lvec import LVec, Vector2D, Vector3D


@pytest.mark.parametrize('cls,n', [(Vector2D,2),(Vector3D,3),(LVec,4)])
def test_algebra_and_native_numpy(cls, n):
    values = [np.array([[i+1., i+2.], [i+3., i+4.]]) for i in range(n)]
    a, b = cls(*values), cls(*(v.copy() for v in values))
    np.testing.assert_array_equal(a == b, np.ones((2,2), dtype=bool))
    assert not np.any(a != b)
    assert np.allclose(a, b) and a.allclose(b)
    assert np.all(np.isclose(a, b))
    assert np.all((np.multiply(2., a) / 2) == a)
    assert np.all(-a == a.scale(-1))
    assert +a is a
    np.testing.assert_allclose(a @ b, a.dot(b))
    expected = sum(v*v for v in values) if n < 4 else values[3]**2-sum(v*v for v in values[:3])
    np.testing.assert_allclose(np.square(a), expected)
    np.testing.assert_allclose(a**2, expected)
    for i,c in enumerate(np.sum(a, axis=1, keepdims=True)._components()):
        np.testing.assert_allclose(c, values[i].sum(axis=1, keepdims=True))
    assert np.concatenate([a,b]).shape == (4,2)
    assert np.stack([a,b]).shape == (2,2,2)
    assert a.reshape(4).shape == (4,)
    assert len(a) == 2
    assert np.count_nonzero(a) == 4
    with pytest.raises(TypeError, match='out'):
        np.add(a,b,out=a)
    with pytest.raises(TypeError):
        np.sum(a, dtype=np.float32)


@pytest.mark.parametrize('left,right',[(2,3),(3,2),(2,4),(4,2),(3,4),(4,3)])
def test_mixed_dimensions_reject(left,right):
    classes={2:Vector2D,3:Vector3D,4:LVec}
    a=classes[left](*([1.]*left));b=classes[right](*([1.]*right))
    for op in (lambda:a+b,lambda:a-b,lambda:a.dot(b),lambda:a==b):
        with pytest.raises(TypeError):op()


def test_component_comparison_nan_and_signed_zero():
    a=Vector2D([0.,np.nan,np.inf],[-0.,1.,2.])
    b=Vector2D([-0.,np.nan,np.inf],[0.,1.,2.])
    np.testing.assert_array_equal(a==b,[True,False,True])
    assert np.all(a.isclose(b,equal_nan=True))
    assert not a.allclose(b)
    assert a != 1


def test_awkward_algebra_and_reduction():
    ak=pytest.importorskip('awkward')
    a=Vector3D(ak.Array([[1.,None],[],None,[2.,3.]]),0.,1.)
    assert ak.to_list(a==a)==[[True,None],[],None,[True,True]]
    assert ak.to_list(np.negative(a).x)==[[-1.,None],[],None,[-2.,-3.]]
    assert ak.to_list(a.sum(axis=-1).x)==[1.,0.,None,5.]
    assert a.allclose(a)


@pytest.mark.parametrize('dimension',[2,3,4])
@pytest.mark.parametrize('backend',['scalar','numpy','awkward'])
def test_all_coordinate_families_and_projections(dimension,backend):
    import lvec
    from lvec._coordinates import COORDINATE_FAMILIES
    values={'x':3.,'y':4.,'rho':5.,'phi':np.arctan2(4.,3.),
            'z':5.,'theta':np.pi/4,'eta':np.arcsinh(1.),'t':10.,'tau':np.sqrt(50.)}
    def wrap(value):
        if backend=='numpy':return np.full((2,3),value)
        if backend=='awkward':
            ak=pytest.importorskip('awkward')
            return ak.Array([[value,None],[],None,[value]])
        return value
    expected=lvec.obj(**{k:wrap(values[k]) for k in ('x','y','z','t')[:dimension]})
    for name,fields in COORDINATE_FAMILIES[dimension]:
        v=lvec.obj(**{key:wrap(values[key]) for key in fields})
        assert v.allclose(expected,rtol=1e-13,atol=1e-13)
        cls={2:Vector2D,3:Vector3D,4:LVec}[dimension]
        assert getattr(cls,'from_'+name)(*(wrap(values[k]) for k in fields)).allclose(expected)
        for target in (2,3,4):
            projected=getattr(v,f'to_{target}D')()
            assert projected._dimension==target
            for i,c in enumerate(projected._components()):
                if i<dimension:
                    other=v._components()[i]
                    if backend=='awkward':
                        ak=pytest.importorskip('awkward')
                        assert ak.almost_equal(c,other,rtol=1e-13,atol=1e-13)
                    else: np.testing.assert_allclose(c,other,rtol=1e-13,atol=1e-13)
    if dimension>=3:
        np.testing.assert_allclose(float(lvec.obj(x=3.,y=4.,z=5.).eta),np.arcsinh(1.))


def test_coordinate_aliases_signed_mass_and_exports():
    import lvec
    v=lvec.obj(pt=2.,eta=0.,phi=0.,mass=-1.)
    assert v.mass2==pytest.approx(-1.)
    assert LVec.from_ptepm(2.,0.,0.,-1.).mass2==pytest.approx(1.)
    assert LVec.from_coordinates(px=3.,py=4.,pz=5.,energy=10.).allclose(LVec(3.,4.,5.,10.))
    assert Vector2D(3.,4.).to_3D(eta=np.arcsinh(1.)).allclose(Vector3D(3.,4.,5.))
    v=LVec(3.,4.,5.,10.)
    assert lvec.array(v.to_numpy()).allclose(v)
    assert v.to_ptphietamass().allclose(v)
    assert v.to_beta3().allclose(Vector3D(.3,.4,.5))
    assert v.Mt2==pytest.approx(75.)
    assert v.Mt==pytest.approx(np.sqrt(75.))
    assert v.Et==pytest.approx(10./np.sqrt(2.))
    huge=LVec(0.,0.,1e308,1.5e308)
    assert np.isfinite(huge.Mt)
    assert huge.Mt==pytest.approx(np.sqrt(1.25)*1e308)
    for coords in (dict(x=1.,y=2.,px=1.),dict(x=1.),dict(x=1.,y=2.,t=3.),dict(x=1.,y=2.,z=3.,eta=1.)):
        with pytest.raises(TypeError):lvec.obj(**coords)


def test_explicit_awkward_constructor_broadcast():
    import lvec
    ak=pytest.importorskip('awkward')
    v=lvec.zip(dict(px=ak.Array([[1.,2.],[],[3.]]),py=0.,pz=0.,E=ak.Array([10.,20.,30.])))
    assert ak.to_list(v.E)==[[10.,10.],[],[30.]]
    assert lvec.Array([{'pt':2.,'phi':0.}]).allclose(Vector2D([2.],[0.]))
