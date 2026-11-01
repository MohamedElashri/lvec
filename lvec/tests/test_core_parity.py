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
