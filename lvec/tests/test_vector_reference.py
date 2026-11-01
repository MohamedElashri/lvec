"""Differential reference checks on the shared, well-defined numerical domain.

Independent physics/edge-case oracles live in test_core_parity and the existing
physics suites. vector is an optional test reference, never a runtime dependency.
"""
import numpy as np
import pytest

vector=pytest.importorskip('vector')
import lvec

# vector 1.9 changes structured dtypes in place during flavor conversions.
# Keep this upstream NumPy 2.5 deprecation separate from LVec's warning checks.
pytestmark=pytest.mark.filterwarnings(
    'ignore:Setting the dtype on a NumPy array has been deprecated:DeprecationWarning:vector.backends.numpy')


def same(a,b):
    try:
        import awkward as ak
    except ImportError:
        ak=None
    if ak is not None and isinstance(a,ak.Array):
        assert ak.almost_equal(a,b,rtol=2e-12,atol=2e-13,check_parameters=False,check_regular=False)
    else:
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13,equal_nan=True)


def components(a,b):
    for name in ('x','y','z','t')[:a._dimension]:same(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('dimension',[2,3,4])
@pytest.mark.parametrize('backend',['scalar','numpy','awkward'])
def test_mathematical_reference(dimension,backend):
    rng=np.random.default_rng(20261101)
    raw=[rng.normal(size=37) for _ in range(3)]
    raw.append(np.hypot(np.hypot(raw[0],raw[1]),np.hypot(raw[2],2.)))
    def wrap(value):
        if backend=='scalar':return float(value[0])
        if backend=='numpy':return value.reshape(37,1)
        ak=pytest.importorskip('awkward')
        data=ak.to_list(ak.unflatten(value,[2,0,3,32]));data[0][1]=None;data[2]=None
        return ak.Array(data)
    fields=dict(zip(('x','y','z','t')[:dimension],map(wrap,raw[:dimension])))
    a=lvec.obj(**fields)
    factory=vector.obj if backend=='scalar' else (vector.array if backend=='numpy' else vector.zip)
    b=factory(**fields) if backend=='scalar' else factory(fields)
    properties=['x','y','rho','rho2','phi','px','py','pt','pt2']
    if dimension>=3:properties+=['z','theta','eta','costheta','cottheta','mag','mag2','p','p2','pz','pseudorapidity']
    if dimension==4:properties+=['t','t2','tau','tau2','E','E2','mass','mass2','beta','gamma','rapidity','Et','Et2','Mt','Mt2']
    # Generic vector flavors lack momentum aliases. Compare those to a
    # momentum flavor with identical Cartesian components.
    momentum_fields=dict(zip(('px','py','pz','E')[:dimension],map(wrap,raw[:dimension])))
    momentum=factory(**momentum_fields) if backend=='scalar' else factory(momentum_fields)
    for name in properties:same(getattr(a,name),getattr(momentum,name))
    for factor in (-2.,.3):
        components(a*factor,b*factor)
        components(a/factor,b/factor)
        components(a.scale2D(factor),b.scale2D(factor))
        if dimension>=3:components(a.scale3D(factor),b.scale3D(factor))
    components(a.unit(),b.unit())
    same(a.dot(a),b.dot(b));same(a**2,b**2)
    same(a.deltaphi(a),b.deltaphi(b))
    same(a.is_parallel(a),b.is_parallel(b))
    same(a.is_antiparallel(-a),b.is_antiparallel(-b))
    components(a.rotateZ(.4),b.rotateZ(.4))
    m2={'xx':1.2,'xy':.1,'yx':-.2,'yy':.8}
    components(a.transform2D(m2),b.transform2D(m2))
    if dimension>=3:
        components(a.cross(a.rotateZ(.7)),b.to_3D().cross(b.rotateZ(.7).to_3D()))
        same(a.deltaangle(a.rotateY(.3)),b.deltaangle(b.rotateY(.3)))
        same(a.deltaR(a.rotateY(.3)),b.deltaR(b.rotateY(.3)))
        same(a.deltaR2(a.rotateY(.3)),b.deltaR2(b.rotateY(.3)))
        same(a.deltaeta(a.rotateY(.3)),b.deltaeta(b.rotateY(.3)))
        components(a.rotateX(.3),b.rotateX(.3));components(a.rotateY(.3),b.rotateY(.3))
        axis=lvec.Vector3D(1.,2.,3.);reference_axis=vector.obj(x=1.,y=2.,z=3.)
        components(a.rotate_axis(axis,.2),b.rotate_axis(reference_axis,.2))
        for order in ('zxz','xyx','yzy','zyz','xzx','yxy','xzy','xyz','yxz','yzx','zyx','zxy'):
            components(a.rotate_euler(.2,.3,.4,order=order),b.rotate_euler(.2,.3,.4,order=order))
        components(a.rotate_nautical(.2,.3,.4),b.rotate_nautical(.2,.3,.4))
        components(a.rotate_quaternion(.5,.2,.3,.4),b.rotate_quaternion(.5,.2,.3,.4))
        m3={i+j:(1. if i==j else .1) for i in 'xyz' for j in 'xyz'}
        components(a.transform3D(m3),b.transform3D(m3))
    if dimension==4:
        beta=lvec.Vector3D(.2,.1,-.3);refbeta=vector.obj(x=.2,y=.1,z=-.3)
        p4=lvec.LVec(.2,.1,-.3,1.);refp4=vector.obj(x=.2,y=.1,z=-.3,t=1.)
        components(a.boost_beta3(beta),b.boost_beta3(refbeta))
        components(a.boost_p4(p4),b.boost_p4(refp4))
        components(a.boostCM_of_p4(p4),b.boostCM_of_p4(refp4))
        components(a.boostCM_of_beta3(beta),b.boostCM_of_beta3(refbeta))
        components(a.to_beta3(),b.to_beta3())
        for axis in 'XYZ':
            for gamma in (-2.,2.):components(getattr(a,'boost'+axis)(gamma=gamma),getattr(b,'boost'+axis)(gamma=gamma))
        same(a.deltaRapidityPhi(a.rotateY(.3)),b.deltaRapidityPhi(b.rotateY(.3)))
        same(a.deltaRapidityPhi2(a.rotateY(.3)),b.deltaRapidityPhi2(b.rotateY(.3)))
        same(a.is_timelike(),b.is_timelike());same(a.is_spacelike(),b.is_spacelike());same(a.is_lightlike(),b.is_lightlike())
        m4={i+j:(1. if i==j else .1) for i in 'xyzt' for j in 'xyzt'}
        components(a.transform4D(m4),b.transform4D(m4))


@pytest.mark.parametrize('dimension',[2,3,4])
def test_coordinate_constructors_and_conversion_reference(dimension):
    from lvec._coordinates import COORDINATE_FAMILIES
    values={'x':3.,'y':4.,'rho':5.,'phi':.4,'z':2.,'theta':.7,'eta':.8,'t':10.,'tau':2.}
    for name,fields in COORDINATE_FAMILIES[dimension]:
        coords={key:values[key] for key in fields}
        a=lvec.obj(**coords);b=vector.obj(**coords)
        components(a,b)
        for target,families in COORDINATE_FAMILIES.items():
            for conversion,_ in families:components(getattr(a,'to_'+conversion)(),getattr(b,'to_'+conversion)())
    if dimension==4:
        for mass in (-2.,0.,2.):
            a=lvec.obj(pt=5.,phi=.4,eta=.8,mass=mass)
            b=vector.obj(pt=5.,phi=.4,eta=.8,mass=mass)
            components(a,b)


def test_native_records_against_reference():
    ak=pytest.importorskip('awkward')
    fields={'px':ak.Array([[1.,None],[],None,[2.,3.]]),'py':0.,'pz':0.,'E':5.}
    a=lvec.zip(fields).to_ak_record(with_behavior=True)
    # Compare complete optional four-vectors, not records whose individual
    # fields have different missing masks (a distinct, documented policy).
    b=vector.zip({name:a[name] for name in fields})
    same(a.mass,b.mass)
    same((a+a).mass,(b+b).mass)
    same(ak.sum(a,axis=-1).mass,ak.sum(b,axis=-1).mass)
    same(ak.count_nonzero(a,axis=-1),ak.count_nonzero(b,axis=-1))
