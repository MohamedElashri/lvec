"""Fused parameter rotations: independent matrices, physics and ownership."""
import itertools
import math

import numpy as np
import pytest

from lvec import LVec, Vector3D
from .test_transform_optimization import reference_dot


ORDERS=('zxz','xyx','yzy','zyz','xzx','yxy','xzy','xyz','yxz','yzx','zyx','zxy')


def elementary(axis,angle):
    c,s=math.cos(angle),math.sin(angle)
    if axis=='x':return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    if axis=='y':return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


@pytest.mark.parametrize('order',ORDERS)
def test_particle_euler_matches_independent_matrices_and_spatial_norm(order):
    rng=np.random.default_rng(20261102)
    xyz=rng.normal(size=(3,3,7))
    angles=rng.normal(size=(3,3,7))
    v=LVec(*xyz,np.full((3,7),7.))
    result=v.rotate_euler(*angles,order=order)
    for i,j in itertools.product(range(3),range(7)):
        matrix=np.eye(3)
        for axis,angle in zip(order[::-1],angles[:,i,j]):
            matrix=elementary(axis,-angle)@matrix
        np.testing.assert_allclose([c[i,j] for c in result._components()[:3]],matrix@xyz[:,i,j],rtol=2e-14,atol=2e-15)
    np.testing.assert_allclose(result.mag2,v.mag2,rtol=3e-15)
    np.testing.assert_array_equal(result.E,v.E)


@pytest.mark.parametrize('size',[1,8,9,65539])
def test_particle_axis_range_and_block_boundaries(size):
    rng=np.random.default_rng(41)
    xyz=rng.normal(size=(3,size))
    axis=rng.normal(size=(3,size))
    angle=np.full(size,.7)
    points=sorted({0,7%size,8%size,65536%size,(size-1)})
    fixtures=[(1e308,1e308,1e308),(1e308,-1e308,1.),
              (np.nextafter(0.,1.),0.,0.),(1.,-1.,1e-300),(0.,0.,0.)]
    for i,index in enumerate(points):
        xyz[:,index]=fixtures[i]
        axis[:,index]=(1.,1.,1.)
        angle[index]=2*math.pi/3
    v=LVec(*xyz,np.full(size,5.))
    with np.errstate(all='raise'):
        result=v.rotate_axis(Vector3D(*axis),angle)
    # Independent coefficient matrix, exact product/sum reference for the
    # supplied binary64 coordinates and rounded rotation coefficients.
    for index in points:
        a=np.ones(3)/math.sqrt(3)
        c,s=math.cos(angle[index]),math.sin(angle[index])
        skew=np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
        matrix=c*np.eye(3)+(1-c)*np.outer(a,a)+s*skew
        for out,row in zip(result._components()[:3],matrix):
            expected=reference_dot(xyz[:,index],row)
            assert out[index]==pytest.approx(expected,rel=3e-14,abs=2*np.nextafter(0.,1.))


def test_particle_axis_extreme_directions_and_zero_rejection():
    x=np.tile([1e308,np.nextafter(0.,1.),1.,0.],3)
    y=np.tile([1e308,0.,1e-300,1.],3)
    z=np.tile([1e308,0.,0.,0.],3)
    v=LVec(np.arange(12.)+1.,2.,3.,30.)
    with np.errstate(all='raise'):
        result=v.rotate_axis(Vector3D(x,y,z),np.full(12,.4))
    for i in range(12):
        expected=v[i].rotate_axis(Vector3D(x[i],y[i],z[i]),.4)
        assert result[i].allclose(expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(result.mag2,v.mag2,rtol=2e-15)
    with pytest.raises(ValueError,match='nonzero'):
        v.rotate_axis(Vector3D(np.zeros(12),0.,0.),np.ones(12))


def test_numpy_broadcast_and_nonfinite_rotation_parameters():
    v=LVec(np.arange(6.).reshape(2,3),1.,2.,8.)
    result=v.rotate_euler(np.array([[.1],[.2]]),.3,.4)
    assert result.shape==(2,3)
    for i in range(2):assert result[i].allclose(v[i].rotate_euler(.1*(i+1),.3,.4))
    with np.errstate(invalid='ignore'):
        result=v.rotate_euler(np.full((2,3),np.inf),.3,.4)
    assert np.all(np.isnan(result.x)) and np.all(np.isnan(result.z))
    assert np.array_equal(result.E,v.E)


def test_awkward_parameter_masks_broadcast_and_snapshot_structure():
    ak=pytest.importorskip('awkward')
    offsets=np.array([0,2,2,4],dtype=np.int64)
    indices=np.array([0,-1,1,2],dtype=np.int64)
    data=np.array([.1,.2,.3])
    angles=ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets),
        ak.contents.IndexedOptionArray(ak.index.Index64(indices),ak.contents.NumpyArray(data))))
    source=LVec(1.,2.,3.,8.)
    result=source.rotate_euler(angles,.3,.4)
    before=[ak.to_list(c) for c in result._components()]
    assert ak.to_list(ak.is_none(result.x,axis=-1))==[[False,True],[],[False,False]]
    expected=[source.rotate_euler(a,.3,.4) for a in data]
    for i,wanted in enumerate(expected):
        for actual,component in zip(result._components(),wanted._components()):
            assert ak.to_numpy(ak.flatten(actual,axis=None))[i]==pytest.approx(component)
    assert offsets.flags.writeable and indices.flags.writeable and data.flags.writeable
    offsets[:]=[0,1,2,4];indices[:]=[0,1,-1,2];data[:]=.9
    assert [ak.to_list(c) for c in result._components()]==before
    for component in result._components():
        assert all(not b.flags.writeable for b in ak.to_buffers(component)[2].values())
    v=LVec(ak.Array([[1.,2.],[],[3.]]),0.,1.,8.)
    angle=ak.Array([.1,.2,.3])
    rotated=v.rotate_axis(Vector3D(0.,0.,1.),angle)
    for event,a in zip((0,2),(.1,.3)):
        assert rotated[event].allclose(v[event].rotateZ(a))
    missing=ak.Array([None,.2,.3])
    assert ak.to_list(v.rotate_euler(missing,.3,.4).x)[0] is None
    with pytest.raises(ValueError,match='nonzero'):
        v.rotate_axis(Vector3D(ak.Array([[0.,1.],[],[1.]]),0.,0.),ak.Array([[None,.1],[],[.2]]))
