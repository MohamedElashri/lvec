"""Independent checks for additional parameterized numerical kernels."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from lvec import LVec
from lvec.exceptions import InputError

# These tests exercise recovery, which the default fast policy skips.
pytestmark = pytest.mark.usefixtures('accurate_policy')


def gamma_reference(p,e,g):
    with localcontext() as ctx:
        ctx.prec=800
        q,t,gd=map(Decimal.from_float,(float(p),float(e),float(g)))
        u=(gd*gd-1).sqrt().copy_sign(gd);gd=abs(gd)
        return float(gd*q+u*t),float(gd*t+u*q)


@pytest.mark.parametrize('size',[1,8,9,32769])
@pytest.mark.parametrize('scalar',[False,True])
def test_gamma_boost_subnormal_recovery_and_block_boundaries(size,scalar):
    tiny=float(np.nextafter(0.,1.))
    p=np.full(size,1.);e=np.full(size,3.);g=np.full(size,1.2)
    indices=sorted({0,size-1,7%size,8%size,32768%size})
    for i in indices:p[i],e[i]=tiny,2*tiny
    if not scalar:
        for i in indices[1:]:p[i],e[i],g[i]=1e308,1e308,-2.
    v=LVec(p,0.,0.,e)
    with np.errstate(all='raise'):
        result=v.boostX(gamma=1.2 if scalar else g)
    for i in indices:
        expected=gamma_reference(p[i],e[i],g[i])
        assert result.px[i]==expected[0]
        assert result.E[i]==expected[1]
    assert np.all(result.py==0.) and np.all(result.pz==0.)


def test_gamma_scalar_subnormal_physical_counterexample():
    tiny=float(np.nextafter(0.,1.))
    v=LVec(tiny,0.,0.,2*tiny)
    result=v.boostX(gamma=1.2)
    assert result.px==3*tiny
    assert (result.px,result.E)==gamma_reference(tiny,2*tiny,1.2)


@pytest.mark.parametrize('qx,px,energy',[(float(np.nextafter(0.,1.)),0.,4.),
                                      (5*float(np.nextafter(0.,1.)),1e-300,1e12)])
def test_reference_boost_does_not_lose_subnormal_proper_velocity(qx,px,energy):
    reference=LVec(qx,0.,3.,5.)
    result=LVec(px,0.,0.,energy).boost_p4(reference)
    with localcontext() as ctx:
        ctx.prec=800
        q=Decimal.from_float(qx);p=Decimal.from_float(px);e=Decimal.from_float(energy)
        mass=(Decimal(16)-q*q).sqrt()
        g=Decimal(5)/mass;u=q/mass;uz=Decimal(3)/mass
        dot=u*p;factor=e+dot/(g+1)
        expected=(float(p+u*factor),0.,float(uz*factor),float(g*e+dot))
    assert result.to_p4()==expected


@pytest.mark.parametrize('scale',[float(np.nextafter(0.,1.)),2*float(np.nextafter(0.,1.)),
                                 float(np.finfo(float).tiny)/8,1e-200,1e-100,1.,1e100,1e200,1e307])
@pytest.mark.parametrize('backend',['scalar','numpy','awkward'])
def test_reference_boost_is_independent_of_reference_scale(scale,backend):
    v=LVec(.1,-.2,.3,2.)
    coordinates=(scale,2*scale,3*scale,5*scale)
    if backend=='numpy':coordinates=tuple(np.full(9,c) for c in coordinates)
    elif backend=='awkward':
        ak=pytest.importorskip('awkward')
        coordinates=tuple(ak.Array([[c]*9,[],None,[None,c]]) for c in coordinates)
    reference=LVec(*coordinates)
    result=v.boost_p4(reference)
    with localcontext() as ctx:
        ctx.prec=100
        mass=Decimal(11).sqrt();g=Decimal(5)/mass
        u=tuple(Decimal(i)/mass for i in (1,2,3))
        xyz=tuple(map(Decimal.from_float,(.1,-.2,.3)));e=Decimal(2)
        dot=sum(a*b for a,b in zip(u,xyz));factor=e+dot/(g+1)
        expected=tuple(float(p+a*factor) for p,a in zip(xyz,u))+(float(g*e+dot),)
    def present(c):
        return ak.to_numpy(ak.flatten(c,axis=None)) if backend=='awkward' else c
    for actual,wanted in zip(result.to_p4(),expected):
        np.testing.assert_allclose(present(actual),wanted,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(present(result.mass2),v.mass2,rtol=5e-14,atol=5e-14)


@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('size',[1,8,9,32769])
def test_reference_boost_recovery_from_original_inputs(size,direction):
    data=np.tile([[.1],[.2],[.3],[2.]],(1,size))
    for i in {0,size-1}:data[:,i]=[0.,0.,1e308,1e308]
    v=LVec(*data);reference=LVec(0.,0.,3.,5.)
    with np.errstate(all='raise'):
        result=v.boost_p4(reference) if direction==1 else v.boostCM_of_p4(reference)
    with localcontext() as ctx:
        ctx.prec=800
        p=Decimal.from_float(1e308)
        wanted_z=float(Decimal('1.25')*p+direction*Decimal('.75')*p)
        wanted_e=float(Decimal('1.25')*p+direction*Decimal('.75')*p)
    for i in {0,size-1}:
        assert result.pz[i]==wanted_z and result.E[i]==wanted_e


def test_array_reference_boosts_preserve_invariants_and_inverse():
    rng=np.random.default_rng(20261103)
    p=rng.normal(size=(3,32769));q=rng.normal(size=(3,32769))
    v=LVec(*p,np.sqrt(np.sum(p*p,axis=0)+4.))
    ref=LVec(*q,np.sqrt(np.sum(q*q,axis=0)+9.))
    result=v.boost_p4(ref)
    np.testing.assert_allclose(result.mass2,v.mass2,rtol=2e-13,atol=2e-13)
    restored=result.boostCM_of_p4(ref)
    for actual,wanted in zip(restored.to_p4(),v.to_p4()):
        np.testing.assert_allclose(actual,wanted,rtol=2e-13,atol=2e-13)


def hamilton(a,b):
    s,x,y,z=a;t,u,v,w=b
    return (s*t-x*u-y*v-z*w,s*u+x*t+y*w-z*v,
            s*v-x*w+y*t+z*u,s*w+x*v-y*u+z*t)


@pytest.mark.parametrize('size',[1,8,9,32769])
def test_quaternion_arrays_against_hamilton_products(size):
    rng=np.random.default_rng(23)
    points=rng.normal(size=(3,size));quaternion=rng.normal(size=(4,size))
    v=LVec(*points,np.full(size,10.))
    result=v.rotate_quaternion(*quaternion)
    norm2=np.sum(quaternion*quaternion,axis=0)
    np.testing.assert_allclose(result.p,norm2*v.p,rtol=2e-14,atol=2e-14)
    for i in {0,size-1,8%size,32768%size}:
        q=tuple(quaternion[:,i]);conj=(q[0],-q[1],-q[2],-q[3])
        expected=hamilton(hamilton(q,(0.,*points[:,i])),conj)[1:]
        np.testing.assert_allclose([c[i] for c in result.to_p4()[:3]],expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(result.E,v.E)


def test_awkward_boost_validation_and_parameter_snapshot():
    ak=pytest.importorskip('awkward')
    v=LVec(ak.Array([[1.,None],[],None,[2.]]),0.,0.,4.)
    g=ak.Array([[1.2,None],[],None,[1.3]])
    boosted=v.boostX(gamma=g)
    assert ak.to_list(ak.is_none(boosted.E,axis=-1))==[[False,True],[],None,[False]]
    for bad in (.5,np.inf,np.nan):
        with pytest.raises(InputError):v.boostX(gamma=ak.Array([[bad,None],[],None,[1.3]]))
    with pytest.raises(InputError):v.boost_p4(LVec(ak.Array([[5.,None],[],None,[1.]]),0.,0.,2.))
    ref=LVec(0.,0.,.3,2.)
    assert v.boost_p4(ref).boostCM_of_p4(ref).allclose(v)
    offsets=np.array([0,2,2,4],dtype=np.int64)
    data=np.array([1.2,1.3,1.4,1.5])
    external=ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets),ak.contents.NumpyArray(data)))
    result=LVec(1.,0.,0.,3.).boostX(gamma=external)
    before=[ak.to_list(c) for c in result.to_p4()]
    offsets[:]=[0,1,2,4];data[:]=2.
    assert [ak.to_list(c) for c in result.to_p4()]==before


def test_native_transform_results_preserve_views_and_owned_conversion():
    ak=pytest.importorskip('awkward')
    from lvec.awkward import zip as record_zip
    x=np.array([1.,2.]);energy=np.array([4.,5.])
    records=record_zip(dict(px=ak.Array(x),py=0.,pz=0.,E=ak.Array(energy)))
    rotated=records.rotate_euler(.1,.2,.3)
    owned=LVec.from_ak_record(rotated)
    before=ak.to_list(rotated.x)
    assert x.flags.writeable and energy.flags.writeable
    x[:]=7.;energy[:]=10.
    assert ak.to_list(rotated.x)==before
    assert ak.to_list(rotated.E)==[10.,10.]
    assert ak.to_list(owned.E)==[4.,5.]
    assert owned.allclose(owned.rotate_quaternion(1.,0.,0.,0.))


def test_quaternion_awkward_masks_and_native_values():
    ak=pytest.importorskip('awkward')
    q=ak.Array([[.5,None],[],None,[.5]])
    v=LVec(1.,2.,3.,8.)
    result=v.rotate_quaternion(q,q,q,q)
    assert ak.to_list(result.x)==[[3.,None],[],None,[3.]]
    assert ak.to_list(result.y)==[[1.,None],[],None,[1.]]
    assert ak.to_list(result.z)==[[2.,None],[],None,[2.]]


def test_quaternion_fusion_recovers_overflowed_intermediates():
    data=np.full(32769,1e308)
    v=LVec(data,data,data,data)
    q=np.full(data.shape,.5)
    with np.errstate(all='raise'):
        result=v.rotate_quaternion(q,q,q,q)
    for component in result.to_p4():np.testing.assert_array_equal(component,data)


@pytest.mark.parametrize('awkward',[False,True])
def test_variable_matrix_recovery_across_large_block_boundaries(awkward):
    from .test_transform_optimization import reference_dot
    size=131073
    data=np.random.default_rng(72).normal(size=(4,size))
    factor=1.+np.arange(size,dtype=float)/size*.01
    indices=(0,32767,32768,65535,65536,131071,131072)
    for index in indices:data[:,index]=(1e308,1e308,1e308,1.)
    if awkward:
        ak=pytest.importorskip('awkward')
        components=tuple(ak.unflatten(c,[65536,0,65537]) for c in data)
        factors=ak.unflatten(factor,[65536,0,65537])
    else:components=data;factors=factor
    matrix={a+b:float(a==b) for a in 'xyzt' for b in 'xyzt'}
    matrix.update(xx=factors,xy=1.,xz=-1.)
    with np.errstate(all='raise'):
        result=LVec(*components).transform4D(matrix)
    x=ak.to_numpy(ak.flatten(result.x,axis=None)) if awkward else result.x
    for index in indices:
        assert x[index]==reference_dot(data[:,index],(factor[index],1.,-1.,0.))


@pytest.mark.parametrize('layout',['byte_mask','indexed','option_index','sliced_list'])
def test_gamma_validation_ignores_hidden_invalid_values(layout):
    ak=pytest.importorskip('awkward')
    data=np.array([np.nan,1.2,1.3,.5])
    content=ak.contents.NumpyArray(data)
    if layout=='byte_mask':
        node=ak.contents.ByteMaskedArray(ak.index.Index8(np.array([0,1,1,0],dtype=np.int8)),content,valid_when=True)
    elif layout=='indexed':
        node=ak.contents.IndexedArray(ak.index.Index64(np.array([1,2],dtype=np.int64)),content)
    elif layout=='option_index':
        node=ak.contents.IndexedOptionArray(ak.index.Index64(np.array([-1,1,2,-1],dtype=np.int64)),content)
    else:
        node=ak.contents.ListOffsetArray(ak.index.Index64(np.array([1,3],dtype=np.int64)),content)
    gamma=ak.Array(node)
    result=LVec(1.,0.,0.,3.).boostX(gamma=gamma)
    assert ak.to_list(ak.ones_like(result.E))==ak.to_list(ak.ones_like(gamma))
    expected=np.array([1.2,1.3])*3+np.sqrt(np.array([1.2,1.3])**2-1)
    np.testing.assert_allclose(ak.to_numpy(ak.flatten(result.E,axis=None)),expected,rtol=2e-14)
    assert data.flags.writeable
    reference=LVec(gamma,0.,0.,4.)
    boosted=LVec(1.,0.,0.,3.).boost_p4(reference)
    assert ak.to_list(ak.ones_like(boosted.E))==ak.to_list(ak.ones_like(gamma))
    np.testing.assert_allclose(ak.to_numpy(ak.flatten(boosted.mass2,axis=None)),8.,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('data',[[],[None],[[]],[[None],[],None]])
def test_gamma_parameters_without_visible_values(data):
    ak=pytest.importorskip('awkward')
    gamma=ak.Array(data)
    result=LVec(1.,0.,0.,3.).boostX(gamma=gamma)
    assert ak.to_list(ak.ones_like(result.E))==ak.to_list(ak.ones_like(gamma))
